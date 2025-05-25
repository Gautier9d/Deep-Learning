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

# Load model and tokenizer

def load_transformer_rationale():
    args = argparse.Namespace()
    args.model_name = "bert"
    
    model = TransformerRationale(args).to('cpu')

    model_path = Path(__file__).resolve().parents[0] / "bert.ckpt"

    # download model if it doesn't exist
    if not model_path.exists():
        print(f"Model checkpoint not found at {model_path}. Downloading...")
        
        # run gdown command with os
        os.system("gdown 1Pr7_fV_v7i6xhAIpfPbRERwZ6Wl0uTFL -O " + str(model_path))

    lit_model = BaseLitModel.load_from_checkpoint(
        model_path,
        model=model,
        args=args,
    ).to('cpu')

    lit_model.eval()  # Set the model to evaluation mode
   
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return lit_model, tokenizer
