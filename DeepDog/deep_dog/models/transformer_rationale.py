from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from transformers import BertModel, BertConfig, AutoModel, AutoConfig
import argparse
from deep_dog.utils import model_name_map
from deep_dog.models.lora import replace_linear_with_lora

# From huggingface:
# Note that if you are used to freezing the body of your pretrained model
# (like in computer vision) it may seem a bit strange, as we are directly
# fine-tuning the whole model without taking any precaution.
# It actually works better this way for Transformers model
# (so this is not an oversight on our side).


class TransformerRationale(nn.Module):
    """
    A unified class for transformer-based models that output rationales.
    Can handle BERT, DistilBERT, HateBERT, and other transformer variants.
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        # Initialize tokenizer based on model name
        self.model_name = self.args.get("model_name", "distilbert")

        # Load model configuration and model
        self.config = get_model_config(self.model_name)
        self.transformer = get_transformer(self.model_name, self.config)

        # Attention layer to predict rationales
        self.attention = nn.Sequential(nn.Linear(self.config.hidden_size, 1),
                                       nn.Sigmoid())

        # Apply LoRA if enabled
        if self.args.get("use_lora", False):
            lora_rank = self.args.get("lora_rank", 8)
            lora_alpha = self.args.get("lora_alpha", 1.0)
            print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
            replace_linear_with_lora(self.transformer,
                                     rank=lora_rank,
                                     alpha=lora_alpha)
            replace_linear_with_lora(self.attention,
                                     rank=lora_rank,
                                     alpha=lora_alpha)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TransformerRationale")
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=768,
            help="Hidden size of the transformer model",
        )

        # Read more at: https://lightning.ai/docs/overview/finetune-models/lora-finetuning
        # Paper: https://arxiv.org/abs/2106.09685

        parser.add_argument(
            "--use_lora",
            action="store_true",
            help="Whether to use LoRA for parameter-efficient fine-tuning",
        )
        parser.add_argument(
            "--lora_rank",
            type=int,
            default=8,
            help="Rank of LoRA decomposition",
        )
        parser.add_argument(
            "--lora_alpha",
            type=float,
            default=1.0,
            help="Scaling factor for LoRA",
        )
        return parent_parser

    def forward(self, batch):
        # Get Transformer outputs
        outputs = self.transformer(input_ids=batch['input_ids'],
                                   attention_mask=batch['attention_mask'])

        # Get sequence outputs
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        # Predict rationale scores for each token
        attention_scores = self.attention(sequence_output).squeeze(
            -1)  # [batch_size, seq_len]

        # Apply attention mask to zero out padding tokens
        attention_scores = attention_scores * batch['attention_mask']

        return attention_scores


def get_model_config(model_name: str):
    model_id = model_name_map.get(model_name, "distilbert-base-uncased")

    if model_name == "distilbert":
        return DistilBertConfig.from_pretrained(model_id,
                                                output_attentions=True)
    elif model_name == "bert":
        return BertConfig.from_pretrained(model_id, output_attentions=True)
    elif model_name in ["hatebert", "hatexplain"]:
        return AutoConfig.from_pretrained(model_id, output_attentions=True)


def get_transformer(model_name: str, config):
    model_id = model_name_map.get(model_name, "distilbert-base-uncased")

    if model_name == "distilbert":
        print("Loading DistilBERT model...")
        return DistilBertModel.from_pretrained(model_id, config=config)
    elif model_name == "bert":
        print("Loading BERT model...")
        return BertModel.from_pretrained(model_id, config=config)
    elif model_name in ["hatebert", "hatexplain"]:
        print(f"Loading {model_name} model...")
        return AutoModel.from_pretrained(model_id, config=config)
