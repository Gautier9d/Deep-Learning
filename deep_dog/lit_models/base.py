from typing import Dict
import torch
import pytorch_lightning as pl
from torch import optim
import torchmetrics
from codecarbon import EmissionsTracker
import os
from pathlib import Path


class TokenF1Score(torchmetrics.Metric):

    def __init__(self):
        super().__init__()
        self.add_state("true_positives",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("false_positives",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("false_negatives",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        # Calculate true positives, false positives, false negatives
        self.true_positives += torch.sum(preds & target)
        self.false_positives += torch.sum(preds & ~target)
        self.false_negatives += torch.sum(~preds & target)

    def compute(self):
        precision = self.true_positives / (self.true_positives +
                                           self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives +
                                        self.false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1


class BaseLitModel(pl.LightningModule):

    def __init__(self, model, args=None):
        super().__init__()
        self.model = model

        # Convert args to dict if it's a Namespace, or create empty dict if None
        args_dict = vars(args) if args is not None else {}

        # Set optimization parameters with defaults
        self.learning_rate = args_dict.get("learning_rate", 2e-5)
        self.weight_decay = args_dict.get("weight_decay", 0.01)

        # Save all hyperparameters
        self.save_hyperparameters(args_dict)

        # Metrics as ModuleDict for proper device handling
        self.train_metrics = torch.nn.ModuleDict({
            'iou':
            torchmetrics.JaccardIndex(task='binary'),
            'f1':
            torchmetrics.F1Score(task='binary'),
            'token_f1':
            TokenF1Score(),
            'auprc':
            torchmetrics.AveragePrecision(task='binary')
        })
        self.val_metrics = torch.nn.ModuleDict({
            'iou':
            torchmetrics.JaccardIndex(task='binary'),
            'f1':
            torchmetrics.F1Score(task='binary'),
            'token_f1':
            TokenF1Score(),
            'auprc':
            torchmetrics.AveragePrecision(task='binary')
        })
        self.test_metrics = torch.nn.ModuleDict({
            'iou':
            torchmetrics.JaccardIndex(task='binary'),
            'f1':
            torchmetrics.F1Score(task='binary'),
            'token_f1':
            TokenF1Score(),
            'auprc':
            torchmetrics.AveragePrecision(task='binary')
        })

        self.emission_dir = self.emission_dirname(
        ) / f"{self.model.model_name}"
        os.makedirs(self.emission_dir,
                    exist_ok=True)  # Ensure emissions directory exists

        # Initialize CodeCarbon tracker
        self.tracker = EmissionsTracker(
            project_name=f"{self.model.model_name}",
            output_dir=self.emission_dir,
            measure_power_secs=1  # Tracks CPU/GPU/RAM power usage every second
        )
        self.emissions = {
            'train': 0.0,
            'val': 0.0,
            'test': 0.0
        }  # Track emissions by phase

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseLitModel")
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parent_parser

    @classmethod
    def emission_dirname(cls):
        return Path(__file__).resolve().parents[2] / "emissions"

    @classmethod
    def predict_dirname(cls):
        return Path(__file__).resolve().parents[2] / "predictions"

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        attention_scores = self(batch)
        # Convert attention scores to binary predictions
        predictions = (attention_scores > 0.5).float()
        return predictions, batch

    def training_step(self, batch: Dict[str, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:

        attention_scores = self(batch)
        loss = torch.nn.functional.binary_cross_entropy(
            attention_scores,
            batch["rationale_mask"].type_as(attention_scores))

        # Update and log metrics
        self.train_metrics['iou'](attention_scores > 0.5,
                                  batch["rationale_mask"])
        self.train_metrics['f1'](attention_scores > 0.5,
                                 batch["rationale_mask"])
        self.train_metrics['token_f1'](attention_scores > 0.5,
                                       batch["rationale_mask"])
        self.train_metrics['auprc'](attention_scores, batch["rationale_mask"])

        self.log("train_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)
        self.log_dict(
            {
                f"train_{k}": m
                for k, m in self.train_metrics.items()
            },
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor],
                        batch_idx: int) -> None:

        attention_scores = self(batch)
        val_loss = torch.nn.functional.binary_cross_entropy(
            attention_scores,
            batch["rationale_mask"].type_as(attention_scores))

        # Update and log metrics
        self.val_metrics['iou'](attention_scores > 0.5,
                                batch["rationale_mask"])
        self.val_metrics['f1'](attention_scores > 0.5, batch["rationale_mask"])
        self.val_metrics['token_f1'](attention_scores > 0.5,
                                     batch["rationale_mask"])
        self.val_metrics['auprc'](attention_scores, batch["rationale_mask"])

        self.log("val_loss", val_loss, prog_bar=True)
        self.log_dict({
            f"val_{k}": m
            for k, m in self.val_metrics.items()
        },
                      prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor],
                  batch_idx: int) -> None:

        attention_scores = self(batch)
        test_loss = torch.nn.functional.binary_cross_entropy(
            attention_scores,
            batch["rationale_mask"].type_as(attention_scores))

        # Update and log metrics
        self.test_metrics['iou'](attention_scores > 0.5,
                                 batch["rationale_mask"])
        self.test_metrics['f1'](attention_scores > 0.5,
                                batch["rationale_mask"])
        self.test_metrics['token_f1'](attention_scores > 0.5,
                                      batch["rationale_mask"])
        self.test_metrics['auprc'](attention_scores, batch["rationale_mask"])

        self.log("test_loss", test_loss)
        self.log_dict({f"test_{k}": m for k, m in self.test_metrics.items()})

    def on_train_epoch_start(self):
        self.tracker.start_task("train_epoch")

    def on_train_epoch_end(self):
        epoch_emissions = self.tracker.stop_task(
            "train_epoch")  # Stop train task
        self.emissions['train'] += epoch_emissions.emissions
        self.log("train_emissions_kgCO2e",
                 epoch_emissions.emissions,
                 on_epoch=True)

    def on_validation_epoch_start(self):
        self.tracker.start_task("val_epoch")

    def on_validation_epoch_end(self):
        epoch_emissions = self.tracker.stop_task("val_epoch")  # Stop val task

        self.emissions['val'] += epoch_emissions.emissions
        self.log("val_emissions_kgCO2e",
                 epoch_emissions.emissions,
                 on_epoch=True)

    def on_test_epoch_start(self):
        self.tracker.start_task("test_epoch")

    def on_test_epoch_end(self):
        epoch_emissions = self.tracker.stop_task(
            "test_epoch")  # Stop test task
        self.emissions['test'] += epoch_emissions.emissions
        self.log("test_emissions_kgCO2e",
                 epoch_emissions.emissions,
                 on_epoch=True)

    def on_fit_end(self):
        # Sum and report total emissions after training and validation
        total_emissions = sum(self.emissions.values())
        print(f"Total Emissions (Train + Val): {total_emissions:.10f} kgCO2e")
        with open(self.emission_dir / "train_val.txt", "a") as f:
            f.write(
                f"Train Emissions: {self.emissions['train']:.10f} kgCO2e\n")
            f.write(f"Val Emissions: {self.emissions['val']:.10f} kgCO2e\n")
            f.write(f"Total (Train + Val): {total_emissions:.10f} kgCO2e\n")

    def on_test_end(self):
        # Update total emissions with test phase and report final total
        total_emissions = sum(self.emissions.values())
        print(
            f"Final Total Emissions (Train + Val + Test): {total_emissions:.10f} kgCO2e"
        )
        with open(self.emission_dir / "total.txt", "a") as f:
            f.write(f"Test Emissions: {self.emissions['test']:.10f} kgCO2e\n")
            f.write(
                f"Final Total (Train + Val + Test): {total_emissions:.10f} kgCO2e\n"
            )
