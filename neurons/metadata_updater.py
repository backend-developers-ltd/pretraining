from typing import List, Optional, Tuple

import bittensor as bt

from taoverse.model.competition import utils as competition_utils
from taoverse.model.competition.data import Competition
from taoverse.model.data import ModelMetadata
from taoverse.model.model_tracker import ModelTracker
from taoverse.model.storage.model_metadata_store import ModelMetadataStore


class MinerMisconfiguredError(Exception):
    """Error raised when a miner is misconfigured."""

    def __init__(self, hotkey: str, message: str):
        self.hotkey = hotkey
        super().__init__(f"[{hotkey}] {message}")


class MetadataUpdater:
    """Checks if the currently tracked model for a hotkey matches what the miner committed to the chain."""

    def __init__(
        self,
        metadata_store: ModelMetadataStore,
        model_tracker: ModelTracker,
    ):
        self.metadata_store = metadata_store
        self.model_tracker = model_tracker

    async def _get_metadata(self, hotkey: str) -> Optional[ModelMetadata]:
        """Get metadata about a model by hotkey"""
        return await self.metadata_store.retrieve_model_metadata(hotkey)

    async def sync_metadata(
        self,
        hotkey: str,
        curr_block: int,
        schedule_by_block: List[Tuple[int, List[Competition]]],
        force: bool = False,
    ) -> bool:
        """Checks for a hotkey if local metadata is out of sync and returns if it was updated."

        Args:
           hotkey (str): The hotkey of the model to sync.
           curr_block (int): The current block.
           force (bool): Whether to force a sync for this model, even if it's chain metadata hasn't changed.
           schedule_by_block (List[Tuple[int, List[Competition]]]): Which competitions are being run at a given block.
        """
        # Get the metadata for the miner.
        metadata = await self._get_metadata(hotkey)

        if not metadata:
            raise MinerMisconfiguredError(
                hotkey, f"No valid metadata found on the chain"
            )

        # Check that the metadata indicates a competition available at time of upload.
        competition = competition_utils.get_competition_for_block(
            comp_id=metadata.id.competition_id,
            block=metadata.block,
            schedule_by_block=schedule_by_block,
        )
        if not competition:
            raise MinerMisconfiguredError(
                hotkey,
                f"No competition found for {metadata.id.competition_id} at block {metadata.block}",
            )

        # Check that the metadata is old enough to meet the eval_block_delay for the competition.
        # If not we return false and will check again next time we go through the update loop.
        if curr_block - metadata.block < competition.constraints.eval_block_delay:
            bt.logging.debug(
                f"""Sync for hotkey {hotkey} delayed as the current block: {curr_block} is not at least 
                {competition.constraints.eval_block_delay} blocks after the upload block: {metadata.block}. 
                Will automatically retry later."""
            )
            return False

        # Check what model id the model tracker currently has for this hotkey.
        tracker_model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
            hotkey
        )
        # If we are not forcing a sync due to retrying a top model we can short-circuit if no change.
        if not force and metadata == tracker_model_metadata:
            return False

        # Update the tracker even if the model fails the following checks to avoid redownloading without new metadata.
        self.model_tracker.on_miner_model_updated(hotkey, metadata)

        return True
