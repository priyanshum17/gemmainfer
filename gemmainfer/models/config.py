from dataclasses import dataclass
from typing import Optional, Literal, List


@dataclass
class GemmaConfig:
    vocab_size: int = 262144
    hidden_size: int = 2816
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    num_global_key_value_heads: int = 2
    
    full_attention_layers: List[int] = None
    sliding_window: int = 1024
    
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = 30.0
    query_pre_attn_scalar: Optional[float] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    global_rope_theta: float = 1000000.0
    max_position_embeddings: int = 262144
    add_unit_offset: bool = False
    
    is_moe: bool = True
    num_routed_experts: int = 128
    num_experts_per_tok: int = 8
    has_shared_mlp: bool = True
    k_eq_v: bool = True
    
    dtype: str = "bfloat16"
    quant_method: Optional[Literal["w4a16", "fp8", "none"]] = None
    quant_group_size: int = 64

    def __post_init__(self):
        if self.full_attention_layers is None:
            self.full_attention_layers = [5, 11, 17, 23, 29]
        if self.query_pre_attn_scalar is None:
            self.query_pre_attn_scalar = self.head_dim ** -0.5

    def is_full_attention(self, layer_idx: int) -> bool:
        return layer_idx in self.full_attention_layers

    @classmethod
    def gemma_4_26b_a4b(cls) -> "GemmaConfig":
        return cls()

    @classmethod
    def gemma_26b_moe(cls) -> "GemmaConfig":
        return cls()

    @classmethod
    def gemma_test_small(cls) -> "GemmaConfig":
        return cls(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            global_head_dim=128,
            num_global_key_value_heads=1,
            full_attention_layers=[1, 3],
            sliding_window=64,
            num_routed_experts=8,
            num_experts_per_tok=2,
            max_position_embeddings=512,
            add_unit_offset=False,
        )
