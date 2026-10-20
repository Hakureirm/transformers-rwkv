# rwkv7_eval — evidence for REPORT.md (work branch only, not part of the upstream PR)

- `harness/` — Modal runners: `modal_ci.py` (CPU tests / repo checks on the working tree), `modal_gpu.py`
  (checkpoint download + hash verification, prompt building, GPU arms, anchor comparisons, repros),
  `summarize.py` (builds `logs/gpu/SUMMARY.md` from the raw JSON logs). All runs used `MODAL_PROFILE=hakureirm`.
- `logs/` — raw logs, quoted by REPORT.md and never edited: CPU test runs (`*.log`, `modal-*.out`),
  `logs/gpu/measure-*.json` (one per checkpoint × arm × dtype), `prepare-*.json` (shard hashes), `prompts-*.json`
  (the exact token ids every arm consumed), repro JSONs, `billing-*.txt` (cost snapshots).
- `checkpoint_audit/` — safetensors headers of the six official checkpoints (HTTP range reads) and the
  name/shape comparison against the reworked model.
- `pr_meta/` — PR #47780 body/comments and issue #47787 as fetched (read-only).
- `local_checks/` — the small local scripts referred to in commit messages (rename round trip, converter,
  warm-cache generate).
- `FINDINGS_raw_notes.md` — running notes taken during the work, kept as written.
