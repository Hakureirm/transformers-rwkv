# Findings log (raw notes for REPORT.md)

All times UTC. Evidence paths are relative to ~/workspace/rwkv7-work unless noted.

## Environment / state at takeover (2026-10-20)
- Modal ~/.modal.toml [hakureirm] token sha256 prefixes 27cd3cccfe57d8f0 / d1c51efe57ab17a4 (match handoff).
- Direct network down; everything via http proxy 127.0.0.1:7897.
- gh on the Mac IS logged in (Hakureirm, keyring, scopes repo/workflow) -- handoff said invalid; verified with read-only API calls.
- PR #47780 head still 317ae0169; fork has no rwkv7-arthur-rework branch.
- Billing before any run: logs/billing-00-before-any-run.txt (billed $0.00; volumes $1.90 covered by free storage).

## Rebase
- upstream/main = 823a6ea6d8 (2026-10-20, "Make longrope compileable (#49555)"); 746 commits ahead of merge-base c1ff11866.
- merge-tree: 0 conflicts. rebase: 13/13 commits replayed cleanly.
- Intermediate sha (rebased, pre-refactor) = 2c594b6e6e.
- Baseline on Modal CPU: tests/models/rwkv7 -> 131 passed, 198 skipped, 0 failed
  (logs/baseline-rebased-2c594b6-20261020T043410Z.log).

## Checkpoint audit (ckpt_audit/, safetensors headers by HTTP range, no weights)
- 6 repos: G1k-{1.5B,2.9B,7.2B,13.3B}-20260930, G1j-1.5B-20260831, 1.5B-20260805.
- After 4 rename rules every tensor maps 1:1 with identical shape (798/798/798/1062/1062/2019). All BF16.
- x_* are (1,1,C), r_k is (H,N). ckpt_audit/compare_reviewer.out.
- Configs: extra keys norm_bias=True, wkv_state_dtype=float32, wkv_implementation (eager for 20260805, chunked otherwise).
  They parse with the strict trimmed config and are kept as plain attributes (handoff 5b closed).

## Commits so far (branch rwkv7-arthur-rework)
- d253af8e25 config trim (item 1)
- 65e8362785 modular rewrite + regenerated modeling (items 2-11)
- 028156eb84 conversion_mapping rwkv7 entry (item 10 gap)
- d3749a0025 conversion script fixed for standard names (not in review)

## Regressions vs the original PR found in the reviewer version (need user decision: fixing deviates from Arthur)
1. generate() with a warm cache crashes. `DynamicCache.get_seq_length()` raises ValueError for a cache made only of
   LinearAttention layers, and GenerationMixin's prefill calls it whenever `past_key_values` is passed
   (generation/utils.py ~L4276). Tried input_ids=FULL, NEW only, NEW + full mask: all raise.
   Fresh generate and a manual cached greedy loop are fine and agree token for token.
   Evidence: scratch/warm_generate.py. Original PR supported generate-with-state (commit d1ba899cb fixed it).
2. Packed sequences are not wired. The decoder layer calls `self.att(residual, v_first, past_key_values)` without
   kwargs, and `torch_rwkv7`/`chunk_rwkv7` are called without `cu_seqlens`. qwen3_next passes
   `cu_seqlens=kwargs.pop("cu_seq_lens_q")`. With fla installed, packed training would silently mix sequences
   (WKV state and token shift). Original PR had test_packed_batch_matches_each_sequence_run_alone.

## Minor
- Rwkv7ForCausalLM.forward docstring example (generated from LlamaForCausalLM) names a non-existent repo
  "meta-rwkv7/Rwkv7-2-7b-hf". Common pattern for `class X(LlamaForCausalLM): pass`; left as Arthur wrote it.
- `use_kernel_func_from_hub_with_fallback` picks fla vs torch at import time -> fla path and torch path must be
  measured in separate images (with / without fla) to be cleanly separated.

## Modal workspace disabled (2026-10-20 ~04:50Z)
- Second run (refactor tests) failed at AttemptStart: `ConflictError: workspace ac-CK5zMJu3RFCcFbTFNgwQHF is disabled`.
  Image build + upload worked; no container ran. logs/modal-run-refactor-v1.out.
- Spend at that point: today $0.0077 (baseline run only), billed $0.00 (credits). Both apps stopped, 0 tasks.
  logs/billing-01-after-workspace-disabled.txt. Cause not visible from the CLI -> needs the web dashboard.
- Did NOT retry and did NOT fall back to kunin-tech (red line).
- Cause (from user): workspace spend limit had been set to $0.01; user raised it to $29.90. Resumed after that.
- Two more attempts were refused with "exceeded its spend limit"; user had changed the wrong setting; fixed the usage limit afterwards.

## More commits
- 2ffd5fbd93 tests ported (Modal CPU: 120 passed / 188 skipped / 2 failed = regression #1, left visible)
  logs/refactor-tests-v2-20261020T062705Z.log
- 7c52f973ef model card updated
- 6d0f5c56eb rwkv7 __init__ regenerated (main's auto-generated init format; rebase-induced)
- 2bb70578e0 model-card date refreshed by utils/add_dates.py

## Repo consistency (make check-repo equivalent)
- Modal (tree without .git): logs/check-repo-v1-*.log, logs/check-repo-v2-*.log (with torchvision).
  OK: modeling_structure, ruff_check, ruff_format, init_isort, sort_auto_mappings, noisy_comments, auto_mappings,
      imports, import_complexity, copies, inits (after fix), doc_toc, modeling_rules_doc, dummies, repo,
      pipeline_typing, config_docstrings, config_attributes, doctest_list, update_metadata, deps_table.
  FAILED: types -> single error in src/transformers/cli/serving/response.py:316 (`name=` unknown argument);
      file untouched by this branch; not rwkv7-related (likely openai package version in the image).
- Local (needs git): check_modular_conversion (rwkv7) OK; check_docstrings (diff) OK; add_dates OK after fix.
  check_reviewers skipped: needs transformers-ci from git (reviewer assignment only).

## 3. (HIGH) With fla installed, Auto* resolves "rwkv7" to fla's own model -> random weights, silently
- Repro (Modal CPU, fla-core/flash-linear-attention 0.5.2): logs/gpu/auto-hijack-repro-*.json
  before importing transformers.models.rwkv7.modeling_rwkv7: CONFIG_MAPPING["rwkv7"] = transformers Rwkv7Config,
  AutoModelForCausalLM -> transformers Rwkv7ForCausalLM.
  after the import: both -> fla.models.rwkv7.* (fla/models/rwkv7/__init__.py:13-15 registers with exist_ok=True,
  model_type 'rwkv7').
- The import of `fla` is triggered by use_kernel_func_from_hub_with_fallback("chunk_rwkv7", "fla", ...) (review item 2).
- Effect seen: RW-fla fp32 run of G1k-1.5B loaded fla.models.rwkv7.modeling_rwkv7.RWKV7ForCausalLM, every
  parameter MISSING (randomly initialized), all logits NaN. That run is INVALID as an RW-fla measurement; kept:
  logs/gpu/measure-RWKV7-G1k-1.5B-20260930-RW-fla-float32-*.json, logs/modal-g1k15-RWfla-fp32.out.
- Also printed by fla: "ChunkDeltaRuleFunction does not support float32 on some platforms. Please use bfloat16/float16."
- Harness change: RW arms now load transformers.Rwkv7ForCausalLM explicitly (protocol change log 2026-10-20 #2).

## GPU numbers so far (G1k-1.5B, A100; OFF fp32 ran on an A100-SXM4-80GB although A100-40GB was requested)
- RW-torch fp32 vs OFF fp32: 26/26 free continuations identical; prefill max_abs <= 8.9e-4, max_rel 9.7e-6;
  forced max_rel 1.6e-6; batch-4 left-padded: 24/24 tokens identical, last-logit |d| 1.4e-4; generate==manual 4/4.
- PR fp32 vs OFF fp32: 26/26 identical; prefill max_rel 1.3e-5, forced 5.3e-6.
- OFF bf16 vs OFF fp32: 13/26 differ; 12 explained by precision, lambada-21 NOT (margin 0.076).
- RW-torch bf16, PR bf16: divergences all explained vs OFF fp32; PR bf16 vs OFF bf16 lambada-12 NOT explained.

## G1k-1.5B results after the harness fix (valid runs)
fp32 vs OFF fp32 (26 prompts; free decode 64 tokens each):
  PR       26/26 identical; prefill abs 1.15e-3 rel 1.26e-5; forced abs 3.43e-4 rel 5.34e-6
  RW-torch 26/26 identical; prefill abs 8.93e-4 rel 9.73e-6; forced abs 1.05e-4 rel 1.58e-6
  RW-fla   26/26 identical; prefill abs 1.18e-1 rel 1.49e-3; forced abs 1.08e-1 rel 1.40e-3   <- ~100x larger:
           fla's chunk kernel is not full fp32 (it warns "ChunkDeltaRuleFunction does not support float32 on some
           platforms"); reviewer code routes decode (T=1) through chunk_rwkv7 as well.
  batch-4 left padding: RW-torch 24/24; RW-fla 23/24 (lambada-16 differs, last-logit |d| 4.1e-3; divergence
           position/margin not recorded by that run -> needs a follow-up run).
bf16 (identical / diverged / unexplained-under-criterion):
  OFF bf16 vs OFF fp32  14/12/1 ; PR vs OFF fp32 13/13/0 ; PR vs OFF bf16 13/13/1
  RW-torch vs OFF fp32  14/12/0 ; RW-torch vs OFF bf16 12/14/0
  RW-fla vs OFF fp32    16/10/1 ; RW-fla vs OFF bf16 11/15/0 ; RW-fla vs RW-torch bf16 12/14/1
  Every "unexplained" case is an exact tie: margin == |d1|+|d2| (arm's top1/top2 equal after rounding):
    OFF lambada-21 (0.0759), RW-fla lambada-04 (0.0557), RW-fla-vs-RWtorch lambada-00 (0.0469).
  PR-vs-OFF-bf16 lambada-12: PR chose 39934, which in the OFF bf16 anchor is TIED for 2nd (2.171875, same as the
  "top2" 22590 picked by topk), 0.03125 below top1 -> three-way near-tie; unexplained only through tie ordering.
  The criterion (strict <) is kept as declared; the tie inspection is reported alongside.
Cost so far: credits used $3.05 (logs/billing-03-after-1.5B-batch.txt); compute today ~$2.81.

## Decision 1 (fla Auto hijack): fixed at the source, library level -> commit "Keep native model types ..."
- Pre-existing on main: importing qwen3_next modeling with fla installed re-maps native mamba (logs/gpu/qwen3next-mamba-hijack-*.json).
- `import fla` re-maps native: bitnet, mamba, mamba2, rwkv7 (logs/gpu/fla-collisions-*.json); 8 main models + rwkv7
  import fla via the helper.
- Fix in hub_kernels._import_kernel_package; test passes with fix, fails without (logs/kernelfix-with-fix-v2-*,
  logs/kernelfix-NEGATIVE-control-v2-*). Real-fla repros after fix resolve to native classes.

## Large models + regressions (all GPU runs done; credits used $11.09, billed $0; logs/billing-05-after-all-gpu.txt)
- All 5 extra checkpoints verified (hub LFS + release-manifest, tensor counts 1062/1062/2019/798/798).
- RW-torch fp32 26/26 identical on G1k-1.5B, 2.9B, 7.2B, G1j-1.5B, 1.5B-20260805; prefill max rel <= 1.31e-5.
- RW-fla fp32: 26/26 on G1k-1.5B and 20260805; 25/26 on 2.9B and G1j; 24/26 on 7.2B -> fails the pre-declared fp32
  pass criterion there. Every divergence at an anchor margin < 0.001 logit, below |d1|+|d2| (~0.002-0.005):
  explained by precision (fla chunk kernel is not full fp32: prefill rel up to 4.1e-3).
- 7.2B bf16 prefill deviation (RW-torch 16.6 / rel 0.156, RW-fla 18.4) matches the official runtime's own
  bf16-vs-fp32 deviation at 7.2B (17.2 / rel 0.165): inherent bf16 effect, not an implementation one.
  (offline anchor comparison; cross-checked on 1.5B against the GPU-computed numbers: identical.)
- 13.3B: bf16 only (fp32 does not fit 40GB, declared); RW-torch 14/26 and RW-fla 17/26 identical vs OFF bf16, no
  unexplained divergence vs OFF; RW-fla vs RW-torch lambada-02 unexplained = exact tie.
- generate()==manual cached loop: 4/4 on every RW run.

## Final checks on the final code (4f6a611)
- Modal CPU: tests/models/rwkv7 + tests/kernels/test_kernels.py + tests/models/qwen3_next -> 281 passed, 396 skipped, 0 failed.
- RUN_SLOW: Rwkv7IntegrationTests 4/4 passed (BlinkDL 0.1B .pth via converter vs rwkv package values);
  torch.export 6/6, OpenVINO 2 passed (6 OpenVINO variants skipped by the mixin), ONNX export 6/6 (onnxscript+onnxruntime added);
  executorch 6 not run (package pins its own torch; would replace the tested torch).
- check-repo: only `types` fails, at src/transformers/cli/serving/response.py:316 (untouched by the branch).
  Local: modular conversion, add_dates, docstrings OK.
- Long-prompt prefill (2048 tokens): torch fallback 7.7 s (1.5B fp32) / 13.6 s (7.2B) / 23.4 s (13.3B bf16) vs fla 0.1-1.7 s
  vs PR's chunked torch solver 0.65 s (1.5B).
