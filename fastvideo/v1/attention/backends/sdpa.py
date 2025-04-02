from typing import List, Optional, Type

import torch

from fastvideo.v1.attention.backends.abstract import (
    AttentionBackend)  # FlashAttentionMetadata,
from fastvideo.v1.attention.backends.abstract import (AttentionImpl,
                                                      AttentionMetadata)
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class SDPABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SDPA"

    @staticmethod
    def get_impl_cls() -> Type["SDPAImpl"]:
        return SDPAImpl

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return FlashAttentionMetadata


class SDPAImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        dropout_rate: float,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: Optional[int] = None,
    ) -> None:
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # transpose to bs, heads, seq_len, head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout_rate,
            is_causal=self.causal,
            scale=self.softmax_scale)
        output = output.transpose(1, 2)
        return output
