from typing import Optional
import argparse
from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoTokenizer, BertTokenizer
from sklearn.model_selection import train_test_split
import nltk
from .base_data_module import BaseDataModule
from .ss_utils import generate_rationale_mask
from deep_dog.utils import model_name_map

# Download the required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "silent_signals"


class SilentSignalsDataModule(BaseDataModule):

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__(args)
        # Get additional parameters specific to this dataset
        args_dict = vars(args) if args is not None else {}
        self.max_length = args_dict.get("max_length", 128)
        self.val_split = args_dict.get("val_split", 0.1)
        self.test_split = args_dict.get("test_split", 0.1)

        # Initialize tokenizer based on model name
        model_name = args_dict.get("model_name", "distilbert")
        model_id = model_name_map.get(model_name, "distilbert-base-uncased")

        if model_name in ['bert', 'hatexplain', 'bert_mlp', 'distilbert']:
            print('Loading BERT tokenizer...')
            self.tokenizer = BertTokenizer.from_pretrained(model_id)
        elif model_name == 'hatebert':
            print('Loading HateBERT tokenizer...')
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.data_train = None
        self.data_val = None
        self.data_test = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        # First add the parent's arguments (batch_size and num_workers)
        parent_parser = BaseDataModule.add_to_argparse(parent_parser)

        # Add dataset-specific arguments
        parser = parent_parser.add_argument_group("SilentSignalsDataModule")
        parser.add_argument(
            "--max_length",
            type=int,
            default=128,
            help="Maximum sequence length for tokenization.",
        )
        parser.add_argument(
            "--val_split",
            type=float,
            default=0.2,
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
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

        # Only generate data if it doesn't already exist
        if DATA_DIRNAME.exists():
            return

        # Download and pre-process the dataset
        dataset = load_dataset("SALT-NLP/silent_signals")
        processed_dataset = dataset.map(generate_rationale_mask,
                                        fn_kwargs={
                                            'tokenizer': self.tokenizer,
                                            'max_length': self.max_length
                                        },
                                        num_proc=self.num_workers)
        # num_proc=self.num_workers)
        processed_dataset.save_to_disk(DATA_DIRNAME)

    def setup(self, stage: Optional[str] = None):
        """
        Split into train, val, test, and set dims.
        Should set self.data_train, self.data_val, and optionally self.data_test.
        """

        # Load the dataset
        dataset = load_from_disk(DATA_DIRNAME)
        train_dataset = dataset["train"]

        # First split off the test set
        train_val_idx, test_idx = train_test_split(range(len(train_dataset)),
                                                   test_size=self.test_split,
                                                   random_state=42)

        # Then split the remaining data into train and validation
        val_size = self.val_split / (
            1 - self.test_split
        )  # Adjust val_size to account for removed test set
        train_idx, val_idx = train_test_split(train_val_idx,
                                              test_size=val_size,
                                              random_state=42)

        def select_split(dataset, indices):
            return dataset.select(indices)

        if stage == "fit" or stage is None:
            train_data = select_split(train_dataset, train_idx)
            val_data = select_split(train_dataset, val_idx)

            self.data_train = SilentSignalsDataset(train_data, self.tokenizer,
                                                   self.max_length)

            self.data_val = SilentSignalsDataset(val_data, self.tokenizer,
                                                 self.max_length)

        if stage == "test" or "predict" or stage is None:
            test_data = select_split(train_dataset, test_idx)
            self.data_test = SilentSignalsDataset(test_data, self.tokenizer,
                                                  self.max_length)

    # Using parent class dataloaders since they have the same implementation


class SilentSignalsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        obj = {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'rationale_mask': torch.tensor(item['rationale_mask']),
            'all_tokens': item['all_tokens'],
            'content': item['content'],
            'dog_whistle': item['dog_whistle'],
        }
        # assert torch.sum(obj['rationale_mask']) > 0, "No rationale mask found"
        return obj
