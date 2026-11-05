import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM, GemmaBatchedMoE
from gemmainfer.kernels.norm import GemmaRMSNorm
from gemmainfer.kernels.activations import apply_logit_softcapping
from gemmainfer.core.block_manager import BlockManager
from gemmainfer.speculative.dspark import DSparkSemiAutoregressiveDrafter, DSparkVerifier
from gemmainfer.core.engine import GemmaInferEngine


def test_standard_rmsnorm():
    norm = GemmaRMSNorm(dim=64, add_unit_offset=False)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_logit_softcapping():
    x = torch.tensor([-100.0, 0.0, 100.0])
    cap = 30.0
    out = apply_logit_softcapping(x, cap)
    assert (out.abs() <= cap).all()
    assert out[1] == 0.0


def test_block_manager_swa_bounded():
    bm = BlockManager(num_blocks=100, block_size=16, sliding_window=64)
    assert bm.max_swa_blocks == 4

    g_blocks, s_blocks = bm.allocate_sequence("req_1", num_tokens=32)
    assert len(g_blocks) == 2
    assert len(s_blocks) == 2

    bm.free_sequence("req_1")
    assert bm.get_num_free_blocks() == 100


def test_dspark_neural_drafter():
    drafter = DSparkSemiAutoregressiveDrafter(hidden_size=64, vocab_size=1000, max_draft_len=4)
    h_T = torch.randn(1, 1, 64)
    draft_tokens, confidences = drafter(h_T, confidence_threshold=0.5)
    assert len(draft_tokens) > 0
    assert len(draft_tokens) <= 4
    assert len(confidences) == len(draft_tokens)


def test_gemma_4_forward_small():
    config = GemmaConfig.gemma_test_small()
    model = GemmaForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    logits, kv = model(input_ids)
    assert logits.shape == (1, 1, config.vocab_size)
    assert len(kv) == config.num_hidden_layers


def test_gemma_4_moe_shared_mlp():
    config = GemmaConfig.gemma_test_small()
    layer = model = GemmaForCausalLM(config).layers[0]
    assert layer.shared_mlp is not None
    assert layer.moe is not None
    x = torch.randn(1, 4, config.hidden_size)
    pos = torch.arange(4).unsqueeze(0)
    out, kv = layer(x, pos)
    assert out.shape == x.shape


def test_marlin_quantized_batch_m():
    config = GemmaConfig.gemma_test_small()
    config.quant_method = "w4a16"
    model = GemmaForCausalLM(config)
    
    # Test batch M > 1 (prefill / multi-token verification)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits, kv = model(input_ids)
    assert logits.shape == (2, 1, config.vocab_size)
    assert not torch.isnan(logits).any()


def test_engine_generate():
    config = GemmaConfig.gemma_test_small()
    engine = GemmaInferEngine(config=config, enable_dspark=True)
    prompt = [1, 2, 3, 4]
    out = engine.generate(prompt, max_tokens=10)
    assert len(out) == 10


if __name__ == "__main__":
    test_standard_rmsnorm()
    test_logit_softcapping()
    test_block_manager_swa_bounded()
    test_dspark_neural_drafter()
    test_gemma_4_forward_small()
    test_gemma_4_moe_shared_mlp()
    test_marlin_quantized_batch_m()
    test_engine_generate()
    print("All tests passed.")
