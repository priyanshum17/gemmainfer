import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.kernels.norm import GemmaRMSNorm
from gemmainfer.kernels.activations import apply_logit_softcapping
from gemmainfer.kernels.quant import MarlinLinear
from gemmainfer.speculative.dspark import PromptLookupCache, DSparkVerifier, Gemma4AssistantDrafter
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


def test_bounded_swa_kv_cache_and_rollback():
    config = GemmaConfig.gemma_test_small()
    config.sliding_window = 16
    model = GemmaForCausalLM(config)

    past_kv = model.init_kv_cache(batch_size=1, max_seq_len=512)
    # Layer 0 is SWA (bounded to sliding_window + 64 = 80, NOT 512)
    assert past_kv[0][0].shape[2] == 80
    # Layer 1 is Full Attention (allocated to full max_seq_len = 512)
    assert past_kv[1][0].shape[2] == 512

    # Run 90 tokens through SWA buffer (exceeding 80-token capacity) to trigger compaction
    ids_1 = torch.randint(0, config.vocab_size, (1, 50))
    _, past_kv = model(ids_1, past_key_values=past_kv)
    assert past_kv[0][2][0] == 50  # logical_pos
    assert past_kv[0][2][1] == 50  # buf_pos

    ids_2 = torch.randint(0, config.vocab_size, (1, 40))
    _, past_kv = model(ids_2, past_key_values=past_kv)
    assert past_kv[0][2][0] == 90  # logical_pos = 90
    assert past_kv[0][2][1] == 15 + 40  # compacted to (sliding_window - 1) + 40 = 55

    # Simulate speculative verification of 5 draft tokens where 3 are rejected (2 accepted)
    spec_ids = torch.randint(0, config.vocab_size, (1, 5))
    _, past_kv = model(spec_ids, past_key_values=past_kv)
    dropped = 3
    for layer_kv in past_kv:
        layer_kv[2][0] -= dropped
        layer_kv[2][1] -= dropped

    assert past_kv[0][2][0] == 92
    assert past_kv[0][2][1] == 57


def test_prompt_lookup_cache():
    cache = PromptLookupCache(ngram_size=2, max_draft_len=3)
    tokens = [10, 20, 30, 40, 50, 20, 30]
    draft = cache.propose(tokens)
    assert draft == [40, 50, 20]


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
    assert isinstance(model.lm_head, MarlinLinear)

    # Test batch M > 1 (prefill / multi-token verification)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))
    logits, kv = model(input_ids)
    assert logits.shape == (2, 1, config.vocab_size)
    assert not torch.isnan(logits).any()


def test_token_union_batched_moe():
    config = GemmaConfig.gemma_test_small()
    config.quant_method = "w4a16"
    model = GemmaForCausalLM(config)
    moe = model.layers[0].moe

    # Test single-token (M=1) and multi-token speculative batch (M=5)
    for num_tokens in [1, 5]:
        x = torch.randn(1, num_tokens, config.hidden_size)
        out = moe(x)
        assert out.shape == (1, num_tokens, config.hidden_size)
        assert not torch.isnan(out).any()


def test_engine_generate():
    config = GemmaConfig.gemma_test_small()
    engine = GemmaInferEngine(config=config, enable_dspark=True)
    prompt = [10, 20, 30, 40, 20, 30]
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
        logits[:, -1, tok] = 20.0

        hidden = torch.randn(batch_size, seq_len, self.hidden_size)
        return MockAssistantOutput(logits=logits, last_hidden_state=hidden)


def test_gemma4_assistant_drafter():
    vocab_size = 500
    embed_dim = 64
    hidden_dim = 64
    embed_tokens = torch.nn.Embedding(vocab_size, embed_dim)

    # 1. Standalone draft sequence
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
    assert mock_model.received_position_ids == [5, 5, 5]
    assert mock_model.received_kv_shapes[0] == torch.Size([1, 2, 5, 32])

    # 2. Bonus token drafting (single-pass speculative mode)
    mock_model_tf = MockAssistantModel(vocab_size, hidden_dim, token_sequence=[201, 202, 203])
    drafter_tf = Gemma4AssistantDrafter(
        assistant_model=mock_model_tf,
        embed_tokens=embed_tokens,
        max_draft_len=3,
    )
    draft_tf = drafter_tf.draft(
        last_token_id=50,
        last_hidden_state=last_hidden_3d,
        shared_kv=shared_kv,
        cur_pos=cur_pos,
        forced_first_token=77,
    )
    assert draft_tf == [201, 202, 203]
    assert mock_model_tf.call_count == 3
    assert mock_model_tf.received_position_ids == [5, 5, 5]

    # 3. Early termination on EOS token (token 1)
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

    # 4. Integration via DSparkVerifier
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
    test_bounded_swa_kv_cache_and_rollback()
    test_prompt_lookup_cache()
    test_gemma_4_forward_small()
    test_gemma_4_moe_shared_mlp()
    test_marlin_quantized_batch_m()
    test_token_union_batched_moe()
    test_engine_generate()
    test_gemma4_assistant_drafter()
    print("All tests passed.")
