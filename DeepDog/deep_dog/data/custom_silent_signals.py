
from typing import Optional
import argparse
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .base_data_module import BaseDataModule


class CustomSilentSignalsDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        content = str(row['content'])
        dog_whistle = str(row['dog_whistle'])
        label = int(row['label'])

        # Tokenize the text
        encoding = self.tokenizer(
            content,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Convert to the expected format
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'rationale': torch.tensor(dog_whistle == '1', dtype=torch.float32),  # Convert dog_whistle to binary tensor
            'label': torch.tensor(label, dtype=torch.long)
        }

        return item


class CustomSilentSignalsDataModule(BaseDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_path = args.data_path
        self.max_length = getattr(args, 'max_length', 512)
        self.val_split = getattr(args, 'val_split', 0.1)
        self.test_split = getattr(args, 'test_split', 0.1)
        self.tokenizer = None  # Will be set later when model type is known

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CustomSilentSignalsDataModule")
        parser.add_argument(
            "--data_path",
            type=str,
            default="DeepDog/deep_dog/data/final_dogwhistle_dataset.csv",
            help="Path to the CSV file containing the custom silent signals dataset"
        )
        parser.add_argument(
            "--max_length",
            type=int,
            default=512,
            help="Maximum length of the input sequence"
        )
        parser.add_argument(
            "--val_split",
            type=float,
            default=0.1,
            help="Validation split ratio"
        )
        parser.add_argument(
            "--test_split",
            type=float,
            default=0.1,
            help="Test split ratio"
        )
        return parent_parser

    def prepare_data(self) -> None:
        """Check if the CSV file exists."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

    def setup(self, stage: Optional[str] = None):
        """Load data and split into train, validation, and test sets."""
        # Read the CSV file
        df = pd.read_csv(self.data_path)
        
        # Verify required columns exist
        required_columns = ['content', 'dog_whistle', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # First split off the test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_split,
            random_state=42,
            stratify=df['label']
        )

        # Then split the remaining data into train and validation
        val_size = self.val_split / (1 - self.test_split)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=42,
            stratify=train_val_df['label']
        )

        if stage == "fit" or stage is None:
            self.data_train = CustomSilentSignalsDataset(
                train_df,
                self.tokenizer,
                self.max_length
            )
            
            self.data_val = CustomSilentSignalsDataset(
                val_df,
                self.tokenizer,
                self.max_length
            )

        if stage == "test" or stage is None:
            self.data_test = CustomSilentSignalsDataset(
                test_df,
                self.tokenizer,
                self.max_length
            )

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer after model initialization."""
        self.tokenizer = tokenizer
