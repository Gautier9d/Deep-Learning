from typing import Optional
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path


class BaseDataModule(pl.LightningDataModule):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / "data"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Number of examples to operate on per forward step.")
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of subprocesses to use for data loading.")
        return parser

    def prepare_data(self) -> None:
        """Called once before any other methods to download/prepare data."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Split into train, val, test, and set dims."""
        pass

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.data_test,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)