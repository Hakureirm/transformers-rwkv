<!-- ci-dashboard-badge:start -->
[![CI](https://transformers-ci.lor-e.huggingface.cool/badge/pr?pr=47780)](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47780)
<!-- ci-dashboard-badge:end -->

Adds RWKV-7 ("Goose") as `Rwkv7Model` / `Rwkv7ForCausalLM`.

Follows up on #46984, where the integration was declined because the published checkpoints did not follow Transformers conventions. That objection was correct at the time; it has since been resolved upstream — see below.

## Checkpoints (updated 2026-08-07)

On 2026-08-05 the RWKV org published official checkpoints in the standard layout: [`RWKV/RWKV7-1.5B-20260805`](https://huggingface.co/RWKV/RWKV7-1.5B-20260805), [`2.9B`](https://huggingface.co/RWKV/RWKV7-2.9B-20260805), [`7.2B`](https://huggingface.co/RWKV/RWKV7-7.2B-20260805) and [`13.3B`](https://huggingface.co/RWKV/RWKV7-13.3B-20260805) — `safetensors` only, no `auto_map`, no remote code, a standard `config.json`, and the World vocabulary shipped as a plain `tokenizer.json` that `AutoTokenizer` loads as an ordinary fast tokenizer.

This implementation loads them unchanged:

- every key in the official `config.json` is a declared `Rwkv7Config` parameter, except `model_type`, which the base class handles;
- all 798 tensors match by name and by shape;
- the release itself is byte-identical to the canonical `BlinkDL/rwkv7-g1` `.pth` it was packed from — 798/798 tensors matched by content hash, a bijection (measured in #47787).

The docs and the `configuration_rwkv7.py` doc checkpoint point at the official repos. (Historical note, since it was the reason #46984 was closed: the older `RWKV/*-PTH` releases ship no `safetensors`, and the older `RWKV/*-HF` ones require `trust_remote_code`. The 20260805 line is what resolved this.)

## Implementation

Portable PyTorch, no third-party runtime dependency:

- Prefill uses a chunk-parallel form of the recurrence; decode runs the sequential single-token path.
- `Rwkv7Cache` is built on `LinearAttentionLayer`. The WKV state dtype is configurable independently of the model dtype, since the recurrent state is more precision-sensitive than the weights.
- Parameter names follow the upstream RWKV reference implementation, so converting a native `.pth` is a prefix rename. The converter also reads the flash-linear-attention layout.

## Correctness

Three independent checks, in addition to the common mixins:

- `tests/models/rwkv7/test_modeling_rwkv7.py` — an integration test matched token-for-token against BlinkDL's own runtime (the `rwkv` package at cpu fp32), through this PR's converter, on a checkpoint whose head count (12) and head width (64) differ, so a quantity indexed by the wrong one cannot pass by coincidence.
- `tests/models/rwkv7/test_modeling_rwkv7_numpy_reference.py` — a numpy re-derivation of the forward pass that shares no code with the model, so an error in the PyTorch path cannot be masked by the same error in the reference.
- Cross-runtime numerics on the official 1.5B (protocol and criteria in #47787): against the `rwkv` package's PyTorch path, 24 lambada prompts, same forced tokens, per-position logits and per-layer recurrent states, prefill (`T>1`) and decode (`T=1`) legs both. fp32: worst relative deviation **8.2e-6** across everything compared. bf16 vs bf16, same protocol: median deviation 6e-3 (bf16's own precision), with signed bias 20–180× smaller than the deviation itself — symmetric rounding, with no directional offset at the state level, which is where the products the dtype question concerns are formed.

## Code Agent Policy

The Transformers repo is currently being overwhelmed by a large number of PRs and issue comments written by
code agents. These often are low-quality, or fix extremely minor issues that occur rarely or never in practice.
As a result, we're instituting a rule that **first-time contributors should not use code agents to submit PRs or issues**.
We'd also ask autonomous "OpenClaw"-like agents not to open any PRs or issues.

Issues/PRs from first-time contributors that violate this rule will probably just be closed without review, and we
might block you, especially if you open more than one or appear to be deliberately ignoring this. We especially do not
want new contributors to jump in on random issues to contribute an agent-written fix. This creates lots of noise
for reviewers and other users and will almost certainly get you blocked.

For more information, please read [`CONTRIBUTING.md`](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).

- [ ] (First-time contributors only): I confirm that this PR description and code is not written by an LLM or code agent

## Before submitting

- [x] Did you read the [contributor guideline](https://huggingface.co/docs/transformers/contributing) and the [Pull Request](https://huggingface.co/docs/transformers/pr_checks) checks?
- [x] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? — #46984
- [x] Did you make sure to update the documentation with your changes according to the [guidelines](https://github.com/huggingface/transformers/tree/main/docs)?
- [x] Did you write any new necessary [tests](https://huggingface.co/docs/transformers/testing)?

## Who can review?

@Cyrilvallez — you raised the conventions objection on #46984; the checkpoint section above is meant to answer it directly.
@ArthurZucker @vasqu

