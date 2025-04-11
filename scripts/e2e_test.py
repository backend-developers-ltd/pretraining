import asyncio
import functools
import logging
import os
import random
import sys

import bittensor
from taoverse.model.model_tracker import ModelTracker
from taoverse.model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from taoverse.model.storage.disk.disk_model_store import DiskModelStore
from taoverse.model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from taoverse.utilities import utils

import constants
import pretrain as pt
from pretrain.datasets.factory import DatasetLoaderFactory

netuid = 9

if len(sys.argv) != 3:
    print(
        f"Usage: python {sys.argv[0]} <uid> <hotkey>",
        file=sys.stderr,
    )
    sys.exit(1)

uid = int(sys.argv[1])
hotkey = sys.argv[2]

seed = 0

model_dir = os.path.join(constants.ROOT_DIR, "model-store/")

competition = constants.COMPETITION_SCHEDULE_BY_BLOCK[0][1][0]

eval_tasks = []
data_loaders = []
samples = []

logging.info(f"Seed used for loading data is: {seed}.")

subtensor = bittensor.subtensor(network="finney")

metadata_store = ChainModelMetadataStore(
    subtensor=subtensor,
    subnet_uid=netuid,
)

model_tracker = ModelTracker()

local_store = DiskModelStore(base_dir=model_dir)
remote_store = HuggingFaceModelStore()

# Get the local path based on the local store to download to (top level hotkey path)
path = local_store.get_path(hotkey)

metadata = asyncio.run(metadata_store.retrieve_model_metadata(uid, hotkey))

# Otherwise we need to download the new model based on the metadata.
model = asyncio.run(remote_store.download_model(
    metadata.id, path, competition.constraints
))

# Update the tracker even if the model fails the following checks to avoid redownloading without new metadata.
model_tracker.on_miner_model_updated(hotkey, metadata)

# Get the tokenizer
tokenizer = pt.model.load_tokenizer(
    competition.constraints, cache_dir=model_dir
)

# Load data based on the competition.
for eval_task in competition.eval_tasks:
    try:
        data_loader = DatasetLoaderFactory.get_loader(
            dataset_id=eval_task.dataset_id,
            dataset_kwargs=eval_task.dataset_kwargs,
            seed=seed,
            sequence_length=competition.constraints.sequence_length,
            tokenizer=tokenizer,
        )
    except Exception as e:
        logging.error(f"Error loading data for task {eval_task.name}: {e}")
        logging.error(f"Skipping task {eval_task.name} for competition {competition.id}")
        continue

    batches = list(data_loader)

    # Shuffle before truncating the list
    random.Random(seed).shuffle(batches)

    if batches:
        eval_tasks.append(eval_task)
        data_loaders.append(data_loader)

        logging.debug(
            f"Found {len(batches)} batches of size: {len(batches[0])} for data_loader: {data_loader.name}:{data_loader.config} over pages {data_loader.get_page_names()}. Up to {constants.MAX_BATCHES_PER_DATASET} batches were randomly chosen for evaluation."
        )

        samples.append(batches[: constants.MAX_BATCHES_PER_DATASET])

    else:
        raise ValueError(
            f"Did not find any data for data loader: {data_loader.name}"
        )

kwargs = competition.constraints.kwargs.copy()
kwargs["use_cache"] = True

model_i_metadata = model_tracker.get_model_metadata_for_miner_hotkey(
    hotkey
)

model_i = local_store.retrieve_model(
    hotkey, model_i_metadata.id, kwargs
)

score, score_details = utils.run_in_subprocess(
    functools.partial(
        pt.validation.score_model,
        model_i=model_i,
        eval_tasks=eval_tasks,
        samples=samples,
        device="cuda",
        seed=seed,
    ),
    ttl=550,
    mode="spawn",
)

print(score, score_details)
