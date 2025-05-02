import os
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import argparse

from deep_dog.data.silent_signals import SilentSignalsDataModule
from deep_dog.data.custom_silent_signals import CustomSilentSignalsDataModule
from deep_dog.models.bert_rationale import BERTRationalePredictor
from deep_dog.models.distilbert_rationale import DistilBERTRationalePredictor
from deep_dog.lit_models.base import BaseLitModel


def cli_main():
    # Build argument parser
    parser = argparse.ArgumentParser(description="Train a dog whistle detection model")
    
    # Add program level args
    parser.add_argument("--wandb", action="store_true", default=False, help="Use Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        choices=["bert", "distilbert"],
        help="Type of model to use (bert or distilbert)"
    )
    parser.add_argument(
        "--data_module",
        type=str,
        default="huggingface",
        choices=["huggingface", "custom"],
        help="Type of data module to use (huggingface or custom)"
    )
    
    # Add model specific args based on model type
    if "--model_type" in sys.argv and sys.argv[sys.argv.index("--model_type") + 1] == "distilbert":
        parser = DistilBERTRationalePredictor.add_model_specific_args(parser)
    else:
        parser = BERTRationalePredictor.add_model_specific_args(parser)
    
    # Add data specific args based on data_module argument
    if "--data_module" in sys.argv and sys.argv[sys.argv.index("--data_module") + 1] == "custom":
        parser = CustomSilentSignalsDataModule.add_model_specific_args(parser)
    else:
        parser = SilentSignalsDataModule.add_model_specific_args(parser)
    
    # Add trainer specific args
    # Using newer PyTorch Lightning pattern for trainer args
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--min_epochs", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32-true", help="Precision of training")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--fast_dev_run", action="store_true", help="Do a trial run for debugging")
    parser.add_argument("--default_root_dir", type=str, default="training/logs")
    
    # Parse args
    args = parser.parse_args()
    
    # Set seed
    pl.seed_everything(args.seed, workers=True)
    
    # Create data module based on data_module argument
    if args.data_module == "huggingface":
        data_module = SilentSignalsDataModule(args)
    else:  # custom
        data_module = CustomSilentSignalsDataModule(args)
    
    # Create model based on model_type argument
    if args.model_type == "bert":
        model = BERTRationalePredictor(args)
    else:  # distilbert
        model = DistilBERTRationalePredictor(args)
    
    lit_model = BaseLitModel(model=model, args=args)
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_iou",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_iou:.2f}",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_iou",
            mode="max",
            patience=3,
            min_delta=0.001
        ),
    ]
    
    # Create logger
    logger = None
    if args.wandb:
        logger = WandbLogger(project="deep-dog", log_model="all")
        logger.watch(lit_model)
        
    # Create trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        precision=args.precision,
        strategy=args.strategy,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        default_root_dir=args.default_root_dir,
    )
    
    # Train
    trainer.fit(lit_model, datamodule=data_module)
    
    # Test
    if not args.fast_dev_run:  # Skip testing in fast_dev_run mode
        trainer.test(lit_model, datamodule=data_module)


if __name__ == "__main__":
    cli_main()
