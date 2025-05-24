import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
import json
import numpy as np
from deep_dog.data.silent_signals import SilentSignalsDataModule
from deep_dog.models.transformer_rationale import TransformerRationale
from deep_dog.lit_models.base import BaseLitModel
import matplotlib.pyplot as plt
from PIL import Image
from deep_dog.utils import get_car_miles, get_household_fraction

def save_emissions_plot(emission_dir, args, model_name):
    with open(emission_dir / "total.txt", "r") as f:
        lines = f.readlines()
        # get the last line
        last_line = lines[-1]
        # get the total emissions
        tot_co2_emission = float(
            last_line.split(":")[-1].strip().split(" ")[0])

    # create figure
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 4), sharey=False)

    ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"

    ax1.imshow(Image.open(f'{ASSET_DIR}/car_icon.png'))
    ax1.set_title(f"{get_car_miles(tot_co2_emission)} car miles driven",
                  fontsize=16)

    ax2.imshow(Image.open(f'{ASSET_DIR}/house_icon.png'))
    ax2.set_title(
        f"{get_household_fraction(tot_co2_emission)}% of Weekly American Household Emissions",
        fontsize=16)

    # save figure
    fig.savefig(emission_dir / f"{model_name}.txt.png", dpi=300)

    # log figure to W&B
    if args.wandb:
        wandb.run.log({"Exemplary Equivalents for CO2 Emission": fig})


def generate_predictions(lit_model, data_module, trainer, args):
    prediction_dir = lit_model.predict_dirname(
    ) / f"{lit_model.model.model_name}"
    os.makedirs(prediction_dir, exist_ok=True)

    lit_model.eval()  # Set the model to evaluation mode

    # Setup data for predictions
    data_module.setup('predict')
    predictions, ground_truth = trainer.predict(
        lit_model, dataloaders=data_module.predict_dataloader())[0]

    # Create results dictionary
    results = []
    ground_truth['all_tokens'] = np.array(ground_truth['all_tokens']).T
    for pred, content, dw, rm, tokens in zip(predictions,
                                             ground_truth['content'],
                                             ground_truth['dog_whistle'],
                                             ground_truth['rationale_mask'],
                                             ground_truth['all_tokens']):
        res = {
            'content': content,
            'dog_whistle': dw,
            'token_rationale': ' '.join(tokens[rm == 1].tolist()),
            'token_prediction': ' '.join(tokens[pred == 1].tolist()),
            'rationale_mask': ''.join(list(map(str, rm.tolist()))),
            'prediction': ''.join(list(map(str, pred.type_as(rm).tolist()))),
        }
        results.append(res)

    # Save results to JSON file
    with open(prediction_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=4)

    if args.wandb:
        # convert to tables
        table = wandb.Table(columns=[
            "content", "dog_whistle", "token_rationale", "token_prediction",
            "rationale_mask", "prediction"
        ])
        for res in results:
            table.add_data(res['content'], res['dog_whistle'],
                           res['token_rationale'], res['token_prediction'],
                           res['rationale_mask'], res['prediction'])
        wandb.log({"predictions": table})


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
    parser.add_argument("--accelerator",
                        type=str,
                        default="gpu",
                        choices=["cpu", "gpu"])
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--max_epochs", type=int, default=10)
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
        save_weights_only=True,
    )
    early_stopping_callback = EarlyStopping(monitor="val_iou",
                                            mode="max",
                                            patience=3,
                                            min_delta=0.001)
    callbacks = [model_checkpoint_callback, early_stopping_callback]

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
        
    # Save emissions plot
    save_emissions_plot(lit_model.emission_dir, args, model.model_name)
   

    # Generate Predictions
    # Load the best model

    # best_model_path = 'training/logs/lightning_logs/version_0/checkpoints/distilbert-epoch=00-val_iou=0.09.ckpt'
    best_model_path = model_checkpoint_callback.best_model_path

    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        model = TransformerRationale(args)
        lit_model = BaseLitModel.load_from_checkpoint(
            best_model_path,
            model=model,
            args=args,
        )
        generate_predictions(lit_model, data_module, trainer, args)

    if args.wandb:
        # Finish wandb run
        wandb.finish()

    # Upload model to wandb if enabled
    # if args.wandb and best_model_path:
    #     wandb.save(best_model_path)
    #     wandb.finish()


if __name__ == "__main__":
    cli_main()
