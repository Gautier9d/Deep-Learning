import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
import argparse


class DistilBERTRationalePredictor(nn.Module):

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        # Load DistilBERT model and configuration
        self.config = DistilBertConfig.from_pretrained(
            'distilbert-base-uncased', output_attentions=True)
        self.distilbert = DistilBertModel.from_pretrained(
            'distilbert-base-uncased', config=self.config)

        # Attention layer to predict rationales
        self.attention = nn.Sequential(nn.Linear(self.config.hidden_size, 1),
                                       nn.Sigmoid())

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group(
            "DistilBERTRationalePredictor")
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=768,
            help="Hidden size of DistilBERT model",
        )
        return parent_parser

    def forward(self, batch):
        # Get DistilBERT outputs
        # Note: DistilBERT doesn't use token_type_ids
        outputs = self.distilbert(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask'])

        # Get sequence outputs
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Predict rationale scores for each token
        attention_scores = self.attention(sequence_output).squeeze(
            -1)  # [batch_size, seq_len]

        # Apply attention mask to zero out padding tokens
        attention_scores = attention_scores * batch['attention_mask']

        return attention_scores
