import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import argparse


class BERTRationalePredictor(nn.Module):
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        
        # Load BERT model and configuration
        self.config = BertConfig.from_pretrained(
            'bert-base-uncased',
            output_attentions=True
        )
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            config=self.config
        )
        
        # Attention layer to predict rationales
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BERTRationalePredictor")
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=768,
            help="Hidden size of BERT model",
        )
        return parent_parser

    def forward(self, batch):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Get sequence outputs
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Predict rationale scores for each token
        attention_scores = self.attention(sequence_output).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply attention mask to zero out padding tokens
        attention_scores = attention_scores * batch['attention_mask']
        
        return attention_scores
