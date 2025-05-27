import argparse
import torch
from deep_dog.models.transformer_rationale import TransformerRationale
from deep_dog.lit_models.base import BaseLitModel
from deep_dog.utils import model_name_map
from transformers import BertTokenizer
from pathlib import Path
import pytorch_lightning as pl
import gdown
import os

MODEL_CHECKPOINTS = {
    "BERT": {
        "file_id": "1Pr7_fV_v7i6xhAIpfPbRERwZ6Wl0uTFL",
        "filename": "bert.ckpt"
    },
    "BERT+LoRA": {
        "file_id": "1iejp06i0BTH4e1-YRFHNjQe6n2McpGhl",
        "filename": "bert_lora.ckpt"
    }
}

def load_transformer_rationale(model_type="BERT"):
    """
    Load the specified model type and tokenizer.
    Args:
        model_type (str): Either "BERT" or "BERT+LoRA"
    """
    args = argparse.Namespace()
    
    if model_type == "BERT":
        args.model_name = "bert"
    elif model_type == "BERT+LoRA":
        args.model_name = "bert"
        args.use_lora = True
    
    model = TransformerRationale(args).to('cpu')

    # Get the appropriate checkpoint info
    checkpoint_info = MODEL_CHECKPOINTS[model_type]
    model_path = Path(__file__).resolve().parents[0] / checkpoint_info["filename"]

    # download model if it doesn't exist
    if not model_path.exists():
        print(f"Model checkpoint not found at {model_path}. Downloading...")
        os.system(f"gdown {checkpoint_info['file_id']} -O {str(model_path)}")

    lit_model = BaseLitModel.load_from_checkpoint(
        model_path,
        model=model,
        args=args,
    ).to('cpu')

    lit_model.eval()  # Set the model to evaluation mode
   
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return lit_model, tokenizer
