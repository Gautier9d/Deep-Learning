import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer that replaces a regular linear layer with a low-rank decomposition.
    
    Args:
        in_features (int): Size of input features
        out_features (int): Size of output features
        rank (int): Rank of the low-rank decomposition
        alpha (float): Scaling factor for the LoRA contribution
        bias (bool): Whether to include a bias term (default: True)
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int = 8,
                 alpha: float = 1.0,
                 bias: bool = True):
        super().__init__()

        # Original linear layer weights frozen and used as-is
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

        # LoRA matrices
        # Note: A is transposed compared to the paper for efficient matmul
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(
            self.lora_B)  # Initialize B to zero so LoRA starts as identity

        self.rank = rank
        self.scaling = alpha / rank

        # Store dimensions for parameter counting
        self.in_features = in_features
        self.out_features = out_features

    def extra_repr(self) -> str:
        """Information to be displayed in print(model)"""
        original_params = self.in_features * self.out_features
        if self.linear.bias is not None:
            original_params += self.out_features

        lora_params = self.in_features * self.rank + self.rank * self.out_features
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'rank={self.rank}, alpha={self.scaling * self.rank}\n'
            f'Original params: {original_params:,} (frozen)\n'
            f'LoRA params: {lora_params:,} (trainable)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular linear transformation
        base_output = self.linear(x)

        # LoRA transformation
        # (B, ..., H) @ (H, R) -> (B, ..., R) @ (R, O) -> (B, ..., O)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling

        return base_output + lora_output

    @staticmethod
    def from_linear(linear: nn.Linear,
                    rank: int = 8,
                    alpha: float = 1.0) -> 'LoRALayer':
        """
        Convert a regular Linear layer to a LoRA layer.
        
        Args:
            linear (nn.Linear): Original linear layer
            rank (int): Rank for LoRA decomposition
            alpha (float): Scaling factor
        """

        lora_layer = LoRALayer(linear.in_features,
                               linear.out_features,
                               rank=rank,
                               alpha=alpha,
                               bias=linear.bias is not None)

        # Copy the weights and bias
        lora_layer.linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora_layer.linear.bias.data.copy_(linear.bias.data)

        return lora_layer


def replace_linear_with_lora(model: nn.Module,
                             rank: int = 8,
                             alpha: float = 1.0):
    """
    Recursively replace all nn.Linear layers in a model with LoRA layers.
    
    Args:
        model (nn.Module): PyTorch model
        rank (int): Rank for LoRA decomposition
        alpha (float): Scaling factor
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name,
                    LoRALayer.from_linear(module, rank=rank, alpha=alpha))
        else:
            replace_linear_with_lora(module, rank=rank, alpha=alpha)
