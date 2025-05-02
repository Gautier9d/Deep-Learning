from typing import Any, Dict
import torch
import pytorch_lightning as pl
from torch import optim
import torchmetrics


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
            'iou': torchmetrics.JaccardIndex(task='binary'),
            'f1': torchmetrics.F1Score(task='binary')
        })
        self.val_metrics = torch.nn.ModuleDict({
            'iou': torchmetrics.JaccardIndex(task='binary'),
            'f1': torchmetrics.F1Score(task='binary')
        })
        self.test_metrics = torch.nn.ModuleDict({
            'iou': torchmetrics.JaccardIndex(task='binary'),
            'f1': torchmetrics.F1Score(task='binary')
        })

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseLitModel")
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parent_parser

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        attention_scores = self(batch)
        loss = torch.nn.functional.binary_cross_entropy(
            attention_scores, 
            batch["rationale_mask"].float()
        )
        
        
        # Update and log metrics
        self.train_metrics['iou'](attention_scores > 0.5, batch["rationale_mask"])
        self.train_metrics['f1'](attention_scores > 0.5, batch["rationale_mask"])
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(
            {f"train_{k}": m for k, m in self.train_metrics.items()},
            on_step=False,
            on_epoch=True,
        )
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        attention_scores = self(batch)
        val_loss = torch.nn.functional.binary_cross_entropy(
            attention_scores,
            batch["rationale_mask"].float()
        )
        
        # Update and log metrics
        self.val_metrics['iou'](attention_scores > 0.5, batch["rationale_mask"])
        self.val_metrics['f1'](attention_scores > 0.5, batch["rationale_mask"])
        
        self.log("val_loss", val_loss, prog_bar=True)
        self.log_dict(
            {f"val_{k}": m for k, m in self.val_metrics.items()},
            prog_bar=True
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        attention_scores = self(batch)
        test_loss = torch.nn.functional.binary_cross_entropy(
            attention_scores,
            batch["rationale_mask"].float()
        )
        
        # Update and log metrics
        self.test_metrics['iou'](attention_scores > 0.5, batch["rationale_mask"])
        self.test_metrics['f1'](attention_scores > 0.5, batch["rationale_mask"])
        
        self.log("test_loss", test_loss)
        self.log_dict(
            {f"test_{k}": m for k, m in self.test_metrics.items()}
        )
