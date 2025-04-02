import os
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse

from fastvideo.v1.logger import init_logger
from fastvideo.v1.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel,
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    destroy_model_parallel, destroy_distributed_environment,
    cleanup_dist_env_and_memory)
from fastvideo.v1.layers.linear import (ColumnParallelLinear, RowParallelLinear)
from fastvideo.v1.distributed.communication_op import (
    tensor_model_parallel_all_reduce, tensor_model_parallel_all_gather)

logger = init_logger(__name__)


class SimpleTPModel(nn.Module):
    """A simple model that uses tensor parallelism."""

    def __init__(self, hidden_size=1024, intermediate_size=4096):
        super().__init__()
        # Column parallel linear layer (splits output dimension)
        self.fc1 = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=True,
            gather_output=
            False,  # Don't gather output since we're passing to row parallel
            skip_bias_add=False)

        # Row parallel linear layer (splits input dimension)
        self.fc2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=True,
            input_is_parallel=True,  # Input is already split from previous layer
            skip_bias_add=False)

        self.activation = nn.GELU()

    def forward(self, x):
        # Forward through column parallel layer
        hidden_states, _ = self.fc1(x)

        # Apply activation
        hidden_states = self.activation(hidden_states)

        # Forward through row parallel layer
        output, _ = self.fc2(hidden_states)

        return output


def initialize_random_weights(model, seed=42):
    """Initialize the model with random weights using a fixed seed for reproducibility."""
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Initialize weights for each layer
    with torch.no_grad():
        # For ColumnParallelLinear layers
        if hasattr(model, 'fc1'):
            nn.init.normal_(model.fc1.weight, mean=0.0, std=0.02)
            if model.fc1.bias is not None:
                nn.init.zeros_(model.fc1.bias)

        # For RowParallelLinear layers
        if hasattr(model, 'fc2'):
            nn.init.normal_(model.fc2.weight, mean=0.0, std=0.02)
            if model.fc2.bias is not None:
                nn.init.zeros_(model.fc2.bias)

    logger.info("Model initialized with random weights")
    return model


def setup_args():
    parser = argparse.ArgumentParser(
        description='Simple Tensor Parallelism Example')
    parser.add_argument('--tensor-model-parallel-size',
                        type=int,
                        default=8,
                        help='Degree of tensor model parallelism')
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='Batch size for the example')
    parser.add_argument('--hidden-size',
                        type=int,
                        default=1024,
                        help='Hidden size for the model')
    parser.add_argument('--intermediate-size',
                        type=int,
                        default=4096,
                        help='Intermediate size for the model')
    return parser.parse_args()


def main():
    args = setup_args()

    # Initialize distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger.info(
        f"Initializing process: rank={rank}, local_rank={local_rank}, world_size={world_size}"
    )

    init_distributed_environment(world_size=world_size,
                                 rank=rank,
                                 local_rank=local_rank)

    # Initialize tensor model parallel groups
    initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size)

    # Get tensor parallel info
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()

    logger.info(
        f"Process rank {rank} initialized with TP rank {tp_rank} in TP world size {tp_world_size}"
    )

    # Create a simple model
    model = SimpleTPModel(hidden_size=args.hidden_size,
                          intermediate_size=args.intermediate_size)

    # Initialize with random weights
    model = initialize_random_weights(model)

    # Create a random input tensor
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    x = torch.randn(batch_size, hidden_size, dtype=torch.float)

    # Move to GPU if available
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)

    # Forward pass
    logger.info(f"Running forward pass on TP rank {tp_rank}")
    with torch.no_grad():
        output = model(x)

    # Print output shape and statistics
    logger.info(f"Output shape: {output.shape}")
    logger.info(
        f"Output mean: {output.mean().item()}, std: {output.std().item()}")

    # Clean up
    logger.info("Cleaning up distributed environment")
    destroy_model_parallel()
    destroy_distributed_environment()
    cleanup_dist_env_and_memory()

    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
