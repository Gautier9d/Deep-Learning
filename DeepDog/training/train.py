import os
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse

from deep_dog.data.silent_signals import SilentSignalsDataModule
from deep_dog.models.transformer_rationale import TransformerRationale
from deep_dog.lit_models.base import BaseLitModel
import matplotlib.pyplot as plt
from PIL import Image
from deep_dog.utils import get_car_miles, get_household_fraction

def cli_main():
    # Build argument parser
    parser = argparse.ArgumentParser(
        description="Train a dog whistle detection model")

    # Add program level args
    parser.add_argument("--wandb",
                        action="store_true",
                        default=False,
                        help="Use Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name",
                        type=str,
                        default="distilbert",
                        choices=[
                            "bert",
                            "hatebert",
                            "hatexplain",
                            "distilbert",
                        ],
                        help="Pretrained model to use")

    # Add shared model specific args
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)

    # Add data specific args
    parser = SilentSignalsDataModule.add_model_specific_args(parser)

    # Add model specific args
    parser = TransformerRationale.add_model_specific_args(parser)

    # Add trainer specific args
    # Using newer PyTorch Lightning pattern for trainer args
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--min_epochs", type=int, default=1)
    parser.add_argument("--precision",
                        type=str,
                        default="32-true",
                        help="Precision of training")

    # Read more here: https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision

    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--fast_dev_run",
                        action="store_true",
                        help="Do a trial run for debugging")
    parser.add_argument("--default_root_dir",
                        type=str,
                        default="training/logs")

    # Parse args
    args = parser.parse_args()

    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # Create data module
    data_module = SilentSignalsDataModule(args)

    # Create model from pretrained
    model = TransformerRationale(args)

    lit_model = BaseLitModel(model=model, args=args)

    # count and print the number of parameters
    num_params = sum(p.numel() for p in lit_model.parameters()
                     if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # Create callbacks
    model_checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        filename=
        f"{args.model_name.split('/')[-1]}-{{epoch:02d}}-{{val_iou:.2f}}",
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(monitor="val_iou",
                                            mode="max",
                                            patience=3,
                                            min_delta=0.001)
    model_summary_callback = ModelSummary(max_depth=1)
    callbacks = [
        model_checkpoint_callback, early_stopping_callback,
        model_summary_callback
    ]

    # Create logger
    logger = None
    if args.wandb:
        logger = WandbLogger(project="deep-dog", log_model="all")
        logger.watch(lit_model)
        logger.log_hyperparams(vars(args))

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

    # Upload model to wandb using model checkpoint
    # best_model_path = model_checkpoint_callback.best_model_path
    # if best_model_path:
    #     print(f"Best model path: {best_model_path}")
    #     if args.wandb:
    #         wandb.save(best_model_path)
    #         print("Best model uploaded to wandb.")

    # open and read the emissions file
    emission_dir =lit_model.emission_dir
    with open(emission_dir / "total.txt", "r") as f:
        lines = f.readlines()
        # get the last line
        last_line = lines[-1]
        # get the total emissions
        tot_co2_emission = float(last_line.split(":")[-1].strip().split(" ")[0])

    # create figure
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 4), sharey=False)

    ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"

    ax1.imshow(Image.open(f'{ASSET_DIR}/car_icon.png'));
    ax1.set_title(f"{get_car_miles(tot_co2_emission)} car miles driven", fontsize=16);

    ax2.imshow(Image.open(f'{ASSET_DIR}/house_icon.png'));
    ax2.set_title(f"{get_household_fraction(tot_co2_emission)}% of Weekly American Household Emissions", fontsize=16);

    # save figure
    fig.savefig(emission_dir / f"{model.model_name}.txt.png", dpi=300)
    
    # log figure to W&B
    if args.wandb:
        wandb.run.log({"Exemplary Equivalents for CO2 Emission": fig})

if __name__ == "__main__":
    cli_main()
