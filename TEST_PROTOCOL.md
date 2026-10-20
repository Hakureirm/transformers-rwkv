# RWKV-7 rework — test protocol

Written and committed **before** any GPU measurement. Changes after the first run go to the
change log at the end, with the reason; nothing below is edited in place once runs start.

## 1. Questions

- **Q1 (correctness)**: does the reworked implementation (branch `rwkv7-arthur-rework`) reproduce the
  official RWKV-7 runtime on the official checkpoints, with no `trust_remote_code`?
- **Q2 (review claim)**: the review states that greedy `generate` of the reworked version "matches the
  PR output token for token". Is that true, and in which dtype?
- **Q3 (kernel paths)**: do the `fla` `chunk_rwkv7` path and the per-token PyTorch fallback give the
  same results? Each is reported separately; one never stands in for the other.

## 2. Arms (implementations under comparison)

| id | what | code |
|---|---|---|
| `PR` | the PR as it is on GitHub, not rebased | `317ae0169f20b7f1333481d78a10b4ccd120f4b8` |
| `RW-fla` | the rework, `flash-linear-attention` installed | branch head at run time (sha logged) |
| `RW-torch` | the rework, `flash-linear-attention` **not installed** | same sha as `RW-fla` |
| `OFF` | the official runtime shipped in each checkpoint repo (`inference/`), `backend="torch"`, `state_dtype="float32"` | repo revision logged |

- `RW-fla` and `RW-torch` run in **two different container images** (with / without fla). The kernel is
  chosen at import time by `use_kernel_func_from_hub_with_fallback`, so a single process cannot be
  trusted to switch. Each run logs which implementation `torch_rwkv7` resolved to.
- `PR` runs in its own image built from that commit (it is a different `transformers` version).
- `OFF` is the numerical anchor (fp32). Its torch backend runs the prefill as an exact per-token scan
  (`sequence_mode="auto"` → `layer-exact`) and decode token by token.
- Intermediate rebased-but-not-reworked state `2c594b6e6e` is used only to check that the rebase alone
  broke nothing (CPU unit tests: 131 passed / 0 failed, `logs/`); it is not a GPU arm.

## 3. Checkpoints

| repo | role |
|---|---|
| `RWKV/RWKV7-G1k-1.5B-20260930` | main test: full suite, all arms, fp32 + bf16 |
| `RWKV/RWKV7-G1k-2.9B-20260930`, `-7.2B-`, `-13.3B-` | `RW-fla` and `OFF` (plus `RW-torch` if budget allows); bf16, and fp32 where it fits |
| `RWKV/RWKV7-G1j-1.5B-20260831`, `RWKV/RWKV7-1.5B-20260805` | regression: `RW-fla` vs `OFF`, fp32 |

Before any measurement, every weight shard is checked against the sha256 recorded by the Hub
(LFS oid) and, where the repo has one, against `release-manifest.json`; the tensor count is checked
against the index (798 / 1062 / 1062 / 2019). A mismatch stops the run. Revisions are pinned to the
commit sha read at download time and logged.

Name/shape alignment of all six checkpoints with the reworked model (no weights) was done
beforehand from the safetensors headers: every tensor maps 1:1 with identical shape after the four
`conversion_mapping` rules.

## 4. Inputs

- **P-lambada**: the first 24 examples of `EleutherAI/lambada_openai` (config `en`, split `test`),
  final word removed, dataset revision logged. *The PR's original 24 lambada prompts are not listed in
  the public record, so this is a reconstruction, not guaranteed to be the same 24.*
- **P-readme**: the example prompt of the G1k model card.
- **P-long**: one prompt of ≥ 2048 tokens built by concatenating lambada test passages 25… until the
  length is reached (spans many fla chunks of 16 or 64 and the state well past the decay horizon).
- **Tokenization**: every prompt is tokenized **once** with the official tokenizer of the checkpoint
  repo (`trust_remote_code=True` for the tokenizer only), and the ids are stored; every arm consumes
  the same ids. Model loading never uses `trust_remote_code` except inside `OFF`.

## 5. Procedure (per checkpoint × dtype × arm)

- Greedy only; `torch.manual_seed(0)`; `torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.backends.cudnn.allow_tf32 = False`. GPU: Modal A100-40GB.
- **Prefill leg**: one forward over the prompt (`T > 1`); logits at **every** prompt position.
- **Decode, free**: 64 new tokens by greedy decoding through the cache (one token per forward).
  The `RW` arms also run `model.generate(max_new_tokens=64, do_sample=False)` and must agree with
  their own manual loop.
- **Decode, forced**: the continuation produced by `OFF` fp32 is fed one token at a time through the
  cache; logits at every step. This makes per-step logits comparable even after a free-decode
  divergence.
- **Batch invariance** (`RW-fla`, `RW-torch`, 1.5B, fp32): the 24 P-lambada prompts in batches of 4
  with left padding + `attention_mask`, against each prompt alone (prefill last-position logits and
  free 64-token continuation).
- **fla variant** (1.5B only, reported separately): `RW-fla` with `chunk_rwkv7(..., safe_gate=True,
  chunk_size=64)` (what fla's own RWKV-7 layer passes), injected by wrapping the kernel; not part of
  the reworked code.
- `OFF` writes its per-position logits (fp32) to a Modal volume; the other arms compare against those
  files inside their container. Raw per-run logs are written to `logs/` and quoted, never rewritten.

## 6. Pre-declared criteria

**fp32 (vs `OFF` fp32)**
- Pass: free-decode continuations identical on **all** prompts (64/64 tokens each).
- Recorded, not a pass condition: max |Δlogit| and max relative deviation over every prefill position
  and every forced-decode step. **Expected non-zero**: the official runtime uses the truncated decay
  constant `-0.606531`, the rework and fla use `-0.6065306597126334` (relative difference ≈ 5.6e-7),
  plus summation-order differences. Not a failure by itself.
- Any token difference in fp32 is reported with its position, the `OFF` top1–top2 margin there, and
  the forced-leg Δ of those two logits — and treated as a finding to explain, not rounded away.

**bf16**
- bf16 is never reported as "identical" or "wrong" on its own. For every bf16 continuation that
  differs from `OFF` fp32: first divergence position *k*, the `OFF` fp32 top1–top2 margin at *k*,
  and the bf16-vs-fp32 deviations Δ₁, Δ₂ of those two logits at *k* from the forced leg.
- A divergence is classed **explained by precision** iff `margin < |Δ₁| + |Δ₂|`; otherwise
  **unexplained**, and listed for investigation. All margins are reported, explained or not.
- bf16 is also compared same-dtype (`RW-*` bf16 vs `OFF` bf16) with the same analysis.

**Q2 (`RW` vs `PR`)**: free-decode continuations compared directly, per dtype; fp32 and bf16 reported
separately, with the same near-tie analysis for any difference.

**Q3 (`RW-fla` vs `RW-torch`)**: same criteria as fp32 above, both directions; plus the same
comparison in bf16 with the precision analysis.

**Batch / padding invariance**: fp32 continuations identical to the unbatched ones; max |Δlogit|
recorded (row-batched matmuls are not bitwise identical to single-row ones).

## 7. Reporting rules

- Failures, divergences and skipped items all go into REPORT.md with their raw log path. A failed
  first run is kept; a re-run is allowed only with a logged reason, and both appear.
- If fla cannot run on a configuration, the reason is recorded and the torch result is not presented
  as an fla result.
- Every run logs: git sha of each arm, `transformers` / `torch` / `fla-core` /
  `flash-linear-attention` versions, CUDA / driver, GPU model, checkpoint revision, shard hashes.
- Cost is read from `modal billing` before and after each GPU batch; if the projected total exceeds
  **$10**, stop and report before continuing.

## 8. Known limits declared in advance

- 13.3B in fp32 needs ~53 GB of weights and does not fit an A100-40GB: its fp32 arm is not run.
- `PR` and `RW` are different `transformers` versions; any difference between them may come from the
  library around the model, not only from the model code. Version strings are logged.
- Generation with a caller-supplied warm cache currently raises in the rework
  (`DynamicCache.get_seq_length()` on a linear-attention-only cache); free decode here starts from an
  empty cache, so this protocol does not exercise that path.

## Change log

- 2026-10-20 (before any GPU run): GPU choice per the user's instruction of 2026-10-20 — **A100-40GB
  for all sizes, including 1.5B**. This replaces the earlier plan (T4 for 1.5B/2.9B, L4 for 7.2B,
  A100-40GB for 13.3B).
- 2026-10-20 (after the first RW-fla run; that run is kept and marked invalid): the `RW-*` arms load
  `transformers.Rwkv7ForCausalLM` explicitly instead of `AutoModelForCausalLM`. Reason: with fla installed,
  importing the reworked modeling module imports `fla`, whose package init registers its own `rwkv7`
  config/model with the Auto classes (`exist_ok=True`), so `AutoModelForCausalLM` returned fla's model with
  every weight randomly initialized (all logits NaN). This is reported as a finding; loading the class
  explicitly keeps the arm what it is meant to be: the reworked model with the fla kernel. The `PR` arm
  (no fla installed) keeps `AutoModelForCausalLM`.
- 2026-10-20 (recorded after the runs, during the final review; the protocol text above was not changed):
  - **Hardware**: A100-40GB was requested for every run; Modal sometimes served an A100-SXM4-80GB (same sm80
    architecture), including the G1k-1.5B fp32 anchor. The GPU model is logged per run.
  - **Official runtime bf16 vs fp32 on 2.9B / 7.2B**: the OFF bf16 and OFF fp32 anchors ran concurrently, so the bf16
    run found no fp32 anchor to compare against. The comparison was done afterwards offline from the two saved
    anchors (prefill deviation, token identity, margin; no forced leg). The same offline code reproduces the
    GPU-computed 1.5B numbers exactly.
  - **Batch check detail**: the first RW-fla batch run recorded only token identity. To apply the section 6 analysis
    to its one divergence, the batch check was re-implemented as a manual batched decode loop recording logits,
    and RW-fla fp32 on 1.5B was re-run (code sha 4f6a611); the first result is kept.
  - **Added checks not in the original protocol**: a packed-sequence check (RW-fla, 1.5B, fp32) after the packed-boundary
    fix, with a no-boundary control; the slow tests (integration vs BlinkDL's runtime, torch.export, ONNX, OpenVINO)
    on CPU. The fla `safe_gate` variant was run in both fp32 and bf16.
  - **Re-run after a code change**: the packed-boundary change (e85329e) came after the first 1.5B runs; RW-fla fp32
    1.5B was re-run on the final code and matched the earlier run exactly (tokens and deviations).
