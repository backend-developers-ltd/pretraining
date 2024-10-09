import asyncio
import argparse
import json

import math
from transformers import AutoTokenizer, logging as hf_logging
from taoverse.utilities import utils
from taoverse.model.data import ModelId
import typing
import functools
import statistics
import pickle
import base64

import logging
import numpy as np
import torch
from taoverse.model.storage.disk.disk_model_store import DiskModelStore
from taoverse.model.storage.hugging_face.hugging_face_model_store import (
    HuggingFaceModelStore,
)

from taoverse.model.data import Model
from taoverse.model.utils import get_hash_of_two_strings
from taoverse.model.competition.data import Competition, ModelConstraints
from taoverse.model.model_updater import MinerMisconfiguredError
import constants
import sys

torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition_pickle", type=str, required=True)
    parser.add_argument("--pack_samples", type=bool, default=False)

    parser.add_argument("--hotkeys", type=str, nargs="+", required=True)
    parser.add_argument("--uids", type=str, nargs="+", required=True)
    parser.add_argument("--metadata_ids", type=str, nargs="+", required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device name.",
    )

    parser.add_argument(
        "--pages_per_eval",
        type=int,
        required=True,
        help="Number of pages used to eval each step. If not specified, it will be automatically set.",
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument(
        "--model_dir",
        default="/app/model-store/",
        help="Where to store downloaded models",
    )
    parser.add_argument(
        "--enable_logging",
        action="store_true",
        help="Enable detailed logging messages.",
    )

    return parser.parse_args()


args = parse_arguments()

hf_logging.set_verbosity_error()
logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG if args.enable_logging else logging.FATAL
)
logger = logging.getLogger(__name__)


def _validate_parameters(
    base_model, eps_soft, eps_soft_percent_threshold, eps_hard, print_vals=False
) -> bool:
    """
    Validate that parameters of a model are below the specified epsilon limits.

    Parameters:
        base_model (transformers.PreTrainedModel): The base model instance.
        num_layers (int): Number of layers in the model to inspect.
        eps_soft (float): Calculate the percentage of layers above this norm
        eps_soft_percent_threshold (float): Threshold of percentage above eps_soft that will trigger a detection
        eps_hard (float): Hard limit for any norm
    """

    exceed_counts = {
        "q_proj": 0,
        "k_proj": 0,
        "v_proj": 0,
        "o_proj": 0,
        "up_proj": 0,
        "down_proj": 0,
    }
    total_counts = {
        "q_proj": 0,
        "k_proj": 0,
        "v_proj": 0,
        "o_proj": 0,
        "up_proj": 0,
        "down_proj": 0,
    }
    if print_vals:
        avg_norms = {
            "q_proj": 0.0,
            "k_proj": 0.0,
            "v_proj": 0.0,
            "o_proj": 0.0,
            "up_proj": 0.0,
            "down_proj": 0.0,
        }
        max_norms = {
            "q_proj": 0.0,
            "k_proj": 0.0,
            "v_proj": 0.0,
            "o_proj": 0.0,
            "up_proj": 0.0,
            "down_proj": 0.0,
        }

    for layer in base_model.model.layers:
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weight_norm = getattr(layer.self_attn, proj).weight.norm().item()
            if weight_norm > eps_hard:
                return False
            elif weight_norm > eps_soft:
                exceed_counts[proj] += 1
            total_counts[proj] += 1
            if print_vals:
                avg_norms[proj] += weight_norm
                max_norms[proj] = max(max_norms[proj], weight_norm)

        # up_proj and down_proj are in the mlp layer
        for proj in ["up_proj", "down_proj"]:
            weight_norm = getattr(layer.mlp, proj).weight.norm().item()
            if weight_norm > eps_hard:
                return False
            elif weight_norm > eps_soft:
                exceed_counts[proj] += 1
            total_counts[proj] += 1
            if print_vals:
                avg_norms[proj] += weight_norm
                max_norms[proj] = max(max_norms[proj], weight_norm)

    # Calculating and printing percentages
    percentages = [exceed_counts[proj] / total_counts[proj] for proj in exceed_counts]

    if print_vals:
        for key, value in total_counts.items():
            avg_norms[key] = avg_norms[key] / value
        print(avg_norms)
        print(max_norms)
        print(percentages)

    return statistics.fmean(percentages) <= eps_soft_percent_threshold


def verify_model_satisfies_parameters(
    model: Model, model_constraints: ModelConstraints
) -> bool:
    if not model_constraints:
        logger.trace(f"No competition found for {model.id.competition_id}")
        return False

    # Check that the parameter count of the model is within allowed bounds.
    parameter_size = sum(p.numel() for p in model.pt_model.parameters())

    if (
        parameter_size > model_constraints.max_model_parameter_size
        or parameter_size < model_constraints.min_model_parameter_size
    ):
        logger.debug(
            f"Model {model.id.name} does not satisfy constraints for competition {model.id.competition_id}"
        )
        logger.debug(f"Number of model parameters is {parameter_size}")
        logger.debug(
            f"Max parameters allowed is {model_constraints.max_model_parameter_size}"
        )
        logger.debug(
            f"Min parameters allowed is {model_constraints.min_model_parameter_size}"
        )
        return False

    # Make sure it's an allowed architecture.
    if type(model.pt_model) not in model_constraints.allowed_architectures:
        return False

    # Check parameters are sane if specified
    if model_constraints.norm_validation_constraints is not None:
        return _validate_parameters(
            model.pt_model,
            model_constraints.norm_validation_constraints.norm_eps_soft,
            model_constraints.norm_validation_constraints.norm_eps_soft_percent_threshold,
            model_constraints.norm_validation_constraints.norm_eps_hard,
        )

    return True


# Function to load a pickled object from a base64 string
def load_pickled_object_from_string(pickled_string):
    return pickle.loads(base64.b64decode(pickled_string))


def run():
    competition: Competition = load_pickled_object_from_string(args.competition_pickle)
    hotkeys = args.hotkeys
    uids = args.uids
    metadata_ids = [ModelId.from_compressed_str(m) for m in args.metadata_ids]

    # download model

    # Setup a RemoteModelStore
    remote_store = HuggingFaceModelStore()

    # Setup a LocalModelStore
    local_store = DiskModelStore(base_dir=args.model_dir)

    # Get the dataloader for this competition
    SubsetDataLoader = constants.DATASET_BY_COMPETITION_ID[competition.id]
    logger.debug(f"Dataset in use: {SubsetDataLoader.name}.")

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        competition.constraints.tokenizer, cache_dir=args.model_dir
    )

    pack_samples = False

    logger.debug(f"Sample packing is set to: {pack_samples}.")
    logger.debug(f"Number of pages per evaluation step is: {args.pages_per_eval}")

    dataloader = SubsetDataLoader(
        batch_size=args.batch_size,
        sequence_length=competition.constraints.sequence_length,
        num_pages=args.pages_per_eval,
        tokenizer=tokenizer,
        pack_samples=pack_samples,
    )
    batches = list(dataloader)

    # Prepare evaluation.
    competition.constraints.kwargs["device_map"] = "auto"

    uid_losses = {}
    for uid_i, hotkey, metadata_id in zip(uids, hotkeys, metadata_ids):
        losses: typing.List[float] = [math.inf for _ in range(len(batches))]
        try:
            logger.info(
                f"Evaluating uid: {uid_i} / hotkey: {hotkey} with metadata_id: {metadata_id}."
            )

            # Get the local path based on the local store to download to (top level hotkey path)
            path = local_store.get_path(hotkey)

            # Otherwise we need to download the new model based on the metadata.
            model = asyncio.run(
                remote_store.download_model(metadata_id, path, competition.constraints)
            )
            logger.info(f"Downloaded model to cuda: {path}")

            # Check that the hash of the downloaded content matches.
            # This is only useful for SN9's legacy competition before multi-competition support
            # was introduced. Securing hashes was optional. In modern competitions, `hash` is
            # always None, and only `secure_hash` is used.
            if model.id.hash != metadata_id.hash:
                # Check that the hash of the downloaded content matches.
                secure_hash = get_hash_of_two_strings(model.id.hash, hotkey)
                if secure_hash != metadata_id.secure_hash:
                    raise MinerMisconfiguredError(
                        hotkey,
                        f"Hash of content downloaded from hugging face does not match chain metadata_id. {metadata_id}",
                    )

            if not verify_model_satisfies_parameters(model, competition.constraints):
                raise MinerMisconfiguredError(
                    hotkey,
                    f"Model does not satisfy parameters for competition {competition.id}",
                )
            logger.info(f"Model {model.id.name} verified successfully.")

            # Setup a LocalModelStore
            local_store = DiskModelStore(base_dir=args.model_dir)
            logger.info(f"Set up local model store.")

            logger.info("Computing losses")
            # Run each computation in a subprocess so that the GPU is reset between each model.
            losses = utils.run_in_subprocess(
                functools.partial(
                    compute_losses,
                    model.pt_model,
                    batches,
                    args.device,
                    tokenizer.eos_token_id,
                    pack_samples,
                ),
                ttl=400,
                mode="spawn",
            )
            del model

        except Exception as e:
            logger.error(
                f"Error in eval loop: {e}. Setting losses for uid: {uid_i} to infinity."
            )

        uid_losses[uid_i] = losses
        average_model_loss = sum(losses) / len(losses)
        logger.debug(
            f"Computed model losses for uid:{uid_i} with average loss: {average_model_loss}"
        )

    print(json.dumps(uid_losses))


def check_for_reasonable_output(
    model, input1: torch.Tensor, input2: torch.Tensor, pad_token_id: int
) -> bool:
    """Checks that a model generates reasonable outputs for two given inputs.

    Args:
        model (torch.nn.Module): The model for which outputs are to be checked. Already loaded to device.
        input1 (torch.Tensor]): Tokenized input1 to check. Already loaded to device.
        input2 (torch.Tensor]): Tokenized input2 to check. Already loaded to device.
        pad_token_id (int): Pad token id for the tokenizer used to generate inputs 1 and 2.

    Returns:
        bool: If the model generates reasonable outputs.
    """
    # Generate 20 tokens of output from the model for each prompt.
    output_length = 20
    # Only take the last 20 tokens since otherwise we also get the prompt ids.
    generate_id1s = model.generate(
        input1,
        min_new_tokens=output_length,
        max_new_tokens=output_length,
        pad_token_id=pad_token_id,
    )[:, -output_length:]
    generate_id2s = model.generate(
        input2,
        min_new_tokens=output_length,
        max_new_tokens=output_length,
        pad_token_id=pad_token_id,
    )[:, -output_length:]

    # Check if too many of the generated ids are the same between the two outputs.
    if torch.sum(torch.eq(generate_id1s, generate_id2s)).item() >= output_length / 2:
        logger.info(
            f"Model with config {model.config} had too much overlap between generated outputs."
        )
        return False

    # Check if internally both responses are too repetitive.
    most_common_counts = []
    for tensor in [generate_id1s, generate_id2s]:
        # Find unique elements and their counts
        _, counts = torch.unique(tensor, return_counts=True)
        # Find the index of the maximum count
        max_count_index = torch.argmax(counts)
        # Extract the count of the most common element
        most_common_counts.append(counts[max_count_index].item())

    if all(count > output_length / 2 for count in most_common_counts):
        logger.info(
            f"Model with config {model.config} had too much repetition in generated outputs."
        )
        return False

    # Passed all the checks, return True.
    return True


def compute_losses(
    model,
    batches: typing.List[np.ndarray],
    device: str,
    pad_token_id: int,
    sample_packing_used: bool,
) -> typing.List[float]:
    """
    Computes the losses for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which losses are to be computed.
        batches (List): A list of batches.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').
        pad_token_id int: Pad token id for the tokenizer used to tokenize the batches.

    Returns:
        list: A list of losses for each batch.
    """
    model.to(device)
    model.eval()

    # First check that model generates reasonable looking outputs.
    # Grab 100 tokens from the first two batches as 'prompts'. (1 x Seq Length tensors.)
    prompt_length = 100
    token_inputs_1 = torch.tensor(batches[0][:, :prompt_length]).to(device)
    token_inputs_2 = torch.tensor(batches[1][:, :prompt_length]).to(device)

    if not check_for_reasonable_output(
        model, token_inputs_1, token_inputs_2, pad_token_id
    ):
        return [math.inf for _ in range(len(batches))]

    # Everything looks good! Continue to computing actual losses.

    # Iterate over each page and corresponding batches
    losses = []
    with torch.no_grad():
        for batch in batches:
            try:
                inputs = torch.tensor(batch).to(device)
                logits = model(inputs).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()

                if not sample_packing_used:
                    # If sample unpacking is used,
                    # create a mask to indicate location of PAD tokens.
                    # Note, PAD tokens are always set to EOS tokens,
                    # For this reason, we want to ignore all but the
                    # first EOS token (the real one)
                    pad_mask = shift_labels == pad_token_id
                    zeros = torch.zeros_like(shift_labels[..., :1])
                    pad_mask = torch.cat((zeros, pad_mask[..., :-1]), dim=-1).bool()
                    # Set all the padded labels to -100, since the
                    # CrossEntropyLoss ignores -100 labels by default.
                    shift_labels[pad_mask] = -100

                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                loss = loss_fct(shift_logits, shift_labels).item()

                losses.append(loss)
            except Exception as e:
                logger.error(f"Exception occurred: {e}")
                losses.append(math.inf)  # Use infinity to indicate failure

    return losses


if __name__ == "__main__":
    run()
