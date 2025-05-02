from typing import Optional
import argparse
from datasets import load_dataset
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


class SilentSignalsDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", 32)
        self.num_workers = self.args.get("num_workers", 4)
        self.max_length = self.args.get("max_length", 128)
        self.val_split = self.args.get("val_split", 0.1)
        self.test_split = self.args.get("test_split", 0.1)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.data_train = None
        self.data_val = None
        self.data_test = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SilentSignalsDataModule")
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Number of examples to operate on per forward step.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of subprocesses to use for data loading.",
        )
        parser.add_argument(
            "--max_length",
            type=int,
            default=128,
            help="Maximum sequence length for tokenization.",
        )
        parser.add_argument(
            "--val_split",
            type=float,
            default=0.1,
            help="Validation set split ratio",
        )
        parser.add_argument(
            "--test_split",
            type=float,
            default=0.1,
            help="Test set split ratio",
        )
        return parent_parser

    def prepare_data(self) -> None:
        """Download data if needed. This method is called only from a single GPU."""
        load_dataset("SALT-NLP/silent_signals")

    def setup(self, stage: Optional[str] = None):
        """Load data and split into train, validation, and test sets."""
        dataset = load_dataset("SALT-NLP/silent_signals")
        train_dataset = dataset["train"].take(1000)
        
        # First split off the test set
        train_val_idx, test_idx = train_test_split(
            range(len(train_dataset)), 
            test_size=self.test_split,
            random_state=42
        )
        
        # Then split the remaining data into train and validation
        val_size = self.val_split / (1 - self.test_split)  # Adjust val_size to account for removed test set
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=42
        )
        
        def select_split(dataset, indices):
            return dataset.select(indices)
        
        if stage == "fit" or stage is None:
            train_data = select_split(train_dataset, train_idx)
            val_data = select_split(train_dataset, val_idx)
            
            self.data_train = SilentSignalsDataset(
                train_data,
                self.tokenizer,
                self.max_length
            )
            
            self.data_val = SilentSignalsDataset(
                val_data,
                self.tokenizer,
                self.max_length
            )

        if stage == "test" or stage is None:
            test_data = select_split(train_dataset, test_idx)
            self.data_test = SilentSignalsDataset(
                test_data,
                self.tokenizer,
                self.max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )


class SilentSignalsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
    
        item = self.dataset[idx]
        text = item["content"]
        dog_whistle = item["dog_whistle"]

        # Tokenize text and create attention mask for dog whistle tokens
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Find dog whistle tokens in the text and create rationale mask
        dog_whistle_tokens = self.tokenizer.encode(
            dog_whistle, add_special_tokens=False
        )
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        
        # Create rationale mask (1 for dog whistle tokens, 0 for others)
        rationale_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids) - len(dog_whistle_tokens)):
            if input_ids[i:i+len(dog_whistle_tokens)].tolist() == dog_whistle_tokens:
                rationale_mask[i:i+len(dog_whistle_tokens)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "rationale_mask": rationale_mask,
            "text": text,
            "dog_whistle": dog_whistle
        }
