### Model description

RWKV-7 ("Goose") is an attention-free recurrent language model from the RWKV project, released across 0.1B–7.2B. It is not compatible with the `rwkv` implementation already in Transformers, which targets RWKV-4: the token-mixing and state-update rules are different, so it needs its own model family rather than a flag on the existing one.

Two things make it awkward to use from Transformers today, and both are about packaging rather than the architecture:

- Every RWKV-7 checkpoint published under `RWKV/` requires `trust_remote_code`. The `*-PTH` repos are raw `.pth` with no `safetensors`; the `*-HF` repos do ship `model.safetensors`, but each also ships `modeling_rwkv7.py` and an `auto_map`, so none of them would load through a library implementation if one existed.
- That was the reason a previous request (#46984) was declined, and it was a fair reason at the time.

Since then a set converted directly from the canonical `BlinkDL/*.pth` releases has been published in the standard layout — `safetensors` only, no pickle, no remote code, ordinary `config.json` with `architectures` / `model_type` — spanning 0.1B to 7.2B. The smallest, `Hakureirm/rwkv7-168m-pile-hf`, has all 399 of its tensors verified bit-identical against the source `.pth` rather than spot-checked. It is a Pile model, so its tokenizer is the ordinary GPT-NeoX-20B fast tokenizer rather than the World vocabulary — the same reason the existing `rwkv` model documents `RWKV/rwkv-4-169m-pile`.

Opening this so there is a tracking issue for the architecture. An implementation is already up as #47780 (portable PyTorch, no third-party runtime dependency; chunk-parallel prefill, sequential decode, `Rwkv7Cache` on `LinearAttentionLayer`), including an integration test matched token-for-token against BlinkDL's own runtime and an independent numpy re-derivation of the forward pass that shares no code with the model.

If maintainers would rather RWKV-7 stay Hub-first, that is a reasonable answer too — I would just like to know before more time goes into it.

### Open source status

- [x] The model implementation is available
- [x] The model weights are available

### Provide useful links for the implementation

**Upstream project and weights**
- Reference implementation: https://github.com/BlinkDL/RWKV-LM (author @BlinkDL)
- Canonical `.pth` releases: https://huggingface.co/BlinkDL
- Official HF-format repos (all require `trust_remote_code`): https://huggingface.co/RWKV

**Standard-layout conversions**
- https://huggingface.co/Hakureirm/rwkv7-168m-pile-hf — no remote code, 399/399 tensors bit-identical to source
- 0.1B / 0.4B / 1.5B / 2.9B / 7.2B under the same account

**In Transformers**
- Implementation PR: #47780
- Earlier request and the conventions discussion: #46984 (cc @Cyrilvallez, who raised the packaging objection there)

