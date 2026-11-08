import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.kernels.norm import GemmaRMSNorm
from gemmainfer.kernels.activations import apply_logit_softcapping
from gemmainfer.core.block_manager import BlockManager
from gemmainfer.speculative.dspark import DSparkSemiAutoregressiveDrafter, DSparkVerifier, Gemma4AssistantDrafter
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
    layer = GemmaForCausalLM(config).layers[0]
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


def test_fused_batched_moe_decode():
    config = GemmaConfig.gemma_test_small()
    config.quant_method = "w4a16"
    model = GemmaForCausalLM(config)
    
    # Single token decode (M=1) to exercise fused_batched_moe
    input_ids = torch.randint(0, config.vocab_size, (1, 1))
    logits, kv = model(input_ids)
    assert logits.shape == (1, 1, config.vocab_size)
    assert not torch.isnan(logits).any()



def test_engine_generate():
    config = GemmaConfig.gemma_test_small()
    engine = GemmaInferEngine(config=config, enable_dspark=True)
    prompt = [1, 2, 3, 4]
    out = engine.generate(prompt, max_tokens=10)
    assert len(out) == 10


class MockAssistantOutput:
    def __init__(self, logits: torch.Tensor, last_hidden_state: torch.Tensor):
        self.logits = logits
        self.last_hidden_state = last_hidden_state


class MockAssistantModel(torch.nn.Module):
    """Mock companion assistant model with custom output tokens and KV shape logging."""
    def __init__(self, vocab_size: int, hidden_size: int, token_sequence: list = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.token_sequence = token_sequence or [42, 43, 44]
        self.call_count = 0
        self.received_position_ids = []
        self.received_kv_shapes = []

    def forward(self, inputs_embeds, position_ids, shared_kv_states=None, use_cache=False):
        self.received_position_ids.append(position_ids.item())
        if shared_kv_states:
            first_key = next(iter(shared_kv_states))
            self.received_kv_shapes.append(shared_kv_states[first_key][0].shape)

        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)

        tok = self.token_sequence[min(self.call_count, len(self.token_sequence) - 1)]
        self.call_count += 1
        logits[:, -1, tok] = 10.0

        hidden = torch.randn(batch_size, seq_len, self.hidden_size)
        return MockAssistantOutput(logits=logits, last_hidden_state=hidden)


def test_gemma4_assistant_drafter():
    vocab_size = 500
    embed_dim = 64
    hidden_dim = 64
    embed_tokens = torch.nn.Embedding(vocab_size, embed_dim)

    # 1. Full length speculative draft sequence
    mock_model = MockAssistantModel(vocab_size, hidden_dim, token_sequence=[101, 102, 103])
    drafter = Gemma4AssistantDrafter(
        assistant_model=mock_model,
        embed_tokens=embed_tokens,
        max_draft_len=3,
    )

    last_hidden_3d = torch.randn(1, 1, hidden_dim)
    shared_kv = {
        "layer_28": (torch.randn(1, 2, 10, 32), torch.randn(1, 2, 10, 32)),
        "layer_29": (torch.randn(1, 2, 10, 32), torch.randn(1, 2, 10, 32)),
    }
    cur_pos = 5

    draft = drafter.draft(
        last_token_id=50,
        last_hidden_state=last_hidden_3d,
        shared_kv=shared_kv,
        cur_pos=cur_pos,
    )

    assert draft == [101, 102, 103]
    assert mock_model.call_count == 3
    assert mock_model.received_position_ids == [4, 5, 6]  # cur_pos - 1 + step
    # Verify shared KV cache was cropped to cur_pos=5
    assert mock_model.received_kv_shapes[0] == torch.Size([1, 2, 5, 32])

    # 2. Early termination on EOS token (token 1)
    mock_model_eos = MockAssistantModel(vocab_size, hidden_dim, token_sequence=[201, 1, 203])
    drafter_eos = Gemma4AssistantDrafter(
        assistant_model=mock_model_eos,
        embed_tokens=embed_tokens,
        max_draft_len=3,
    )
    draft_eos = drafter_eos.draft(
        last_token_id=50,
        last_hidden_state=last_hidden_3d,
        shared_kv=shared_kv,
        cur_pos=cur_pos,
    )
    assert draft_eos == [201, 1]
    assert mock_model_eos.call_count == 2

    # 3. 2D hidden state input tensor handling
    last_hidden_2d = torch.randn(1, hidden_dim)
    mock_model_2d = MockAssistantModel(vocab_size, hidden_dim, token_sequence=[301, 302])
    drafter_2d = Gemma4AssistantDrafter(
        assistant_model=mock_model_2d,
        embed_tokens=embed_tokens,
        max_draft_len=2,
    )
    draft_2d = drafter_2d.draft(
        last_token_id=50,
        last_hidden_state=last_hidden_2d,
        shared_kv=shared_kv,
        cur_pos=cur_pos,
    )
    assert draft_2d == [301, 302]

    # 4. Integration via DSparkVerifier propose_draft pipeline
    mock_model_verifier = MockAssistantModel(vocab_size, hidden_dim, token_sequence=[401, 402, 403])
    verifier = DSparkVerifier(
        assistant_model=mock_model_verifier,
        embed_tokens=embed_tokens,
        hidden_size=hidden_dim,
        vocab_size=vocab_size,
        max_draft_len=3,
    )
    assert verifier.assistant_drafter is not None
    proposed = verifier.propose_draft(
        tokens=[10, 20, 50],
        last_hidden_state=last_hidden_3d,
        shared_kv=shared_kv,
        cur_pos=cur_pos,
    )
    assert proposed == [401, 402, 403]


if __name__ == "__main__":
    test_standard_rmsnorm()
    test_logit_softcapping()
    test_block_manager_swa_bounded()
    test_dspark_neural_drafter()
    test_gemma_4_forward_small()
    test_gemma_4_moe_shared_mlp()
    test_marlin_quantized_batch_m()
    test_fused_batched_moe_decode()
    test_engine_generate()
    test_gemma4_assistant_drafter()
    print("All tests passed.")
