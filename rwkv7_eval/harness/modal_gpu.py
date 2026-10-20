"""GPU measurements for the RWKV-7 rework, following TEST_PROTOCOL.md (profile hakureirm only).

Steps (each a separate `modal run` so cost can be read in between):
    modal run modal_gpu.py::prepare --repo RWKV7-G1k-1.5B-20260930
    modal run modal_gpu.py::measure --repo RWKV7-G1k-1.5B-20260930 --arm OFF --dtype float32 --save
    ...

Arms: OFF (official runtime in the checkpoint repo, backend=torch, fp32 state), RW-torch (rework,
no fla), RW-fla (rework with fla-core + flash-linear-attention), PR (317ae01, its own torch solver).
Anchors (OFF and RW-torch when --save) are written to the volume; other arms compare in-container.
"""

import datetime
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RW_TREE = os.path.expanduser("~/workspace/rwkv7-tf-rework")
PR_TREE = os.path.expanduser("~/workspace/rwkv7-pr-317ae01")
PR_SHA = "317ae0169f20b7f1333481d78a10b4ccd120f4b8"
IGNORE = ["**/__pycache__", "**/*.pyc"]
GPU = "A100-40GB"
N_NEW = 64
LONG_LEN = 2048
LONG_KEEP_TAIL = 128  # stored positions for the long prompt: every 64th + the last 128
VOL = modal.Volume.from_name("rwkv7-eval", create_if_missing=True)


def _base(reqs_file):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install("torch==2.14.1")
        .pip_install_from_requirements(os.path.join(HERE, reqs_file))
        .pip_install("datasets", "jinja2")
        .env({"PYTHONPATH": "/repo/src", "HF_HOME": "/tmp/hf", "PYTHONDONTWRITEBYTECODE": "1",
              "TOKENIZERS_PARALLELISM": "false"})
    )


img_rw_torch = _base("gpu_reqs_rw.txt").add_local_dir(os.path.join(RW_TREE, "src"), "/repo/src", ignore=IGNORE)
img_rw_fla = (
    _base("gpu_reqs_rw.txt")
    .pip_install("fla-core==0.5.2", "flash-linear-attention==0.5.2")
    .add_local_dir(os.path.join(RW_TREE, "src"), "/repo/src", ignore=IGNORE)
)
img_pr = _base("gpu_reqs_pr.txt").add_local_dir(os.path.join(PR_TREE, "src"), "/repo/src", ignore=IGNORE)

app = modal.App("rwkv7-gpu-eval")


# ---------------------------------------------------------------------------------------------
# helpers (run inside containers)
# ---------------------------------------------------------------------------------------------


def _ckpt_dir(repo):
    meta = json.load(open(f"/vol/ckpt/{repo}/CURRENT.json"))
    return meta["path"], meta


def _env_info():
    import platform
    import subprocess

    import torch

    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import transformers

        info["transformers"] = transformers.__version__
        info["transformers_path"] = transformers.__file__
    except Exception as e:  # noqa: BLE001
        info["transformers"] = f"import failed: {e}"
    for dist in ("fla-core", "flash-linear-attention", "triton", "tilelang"):
        try:
            from importlib.metadata import version

            info[dist] = version(dist)
        except Exception:  # noqa: BLE001
            info[dist] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        try:
            info["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            info["driver"] = None
    info["allow_tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    info["allow_tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    return info


def _resolved_wkv():
    """Which implementation the rework's `torch_rwkv7` resolved to at import time."""
    import inspect

    import transformers.models.rwkv7.modeling_rwkv7 as M

    fn = M.torch_rwkv7
    seen, frontier = set(), [fn]
    while frontier:
        f = frontier.pop()
        if id(f) in seen or not callable(f):
            continue
        seen.add(id(f))
        try:
            nl = inspect.getclosurevars(f).nonlocals
        except (TypeError, ValueError):
            continue
        if "implementation" in nl:
            impl = nl["implementation"]
            return f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__qualname__', '?')}"
        frontier.extend(v for v in nl.values() if callable(v))
    return "unresolved"


class Adapter:
    """Uniform prefill / step over the three code bases."""

    def __init__(self, arm, path, dtype):
        import torch

        self.arm, self.dtype = arm, dtype
        if arm == "OFF":
            import sys

            sys.path.insert(0, path)
            from inference.model_loader import load_model_and_tokenizer

            self.model, _ = load_model_and_tokenizer(
                path, device="cuda", dtype=dtype, backend="torch", state_dtype="float32"
            )
        elif arm == "PR":
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to("cuda").eval()
            self.loading_class = type(self.model).__module__ + "." + type(self.model).__name__
        else:
            # explicit class: with fla installed, the Auto classes resolve "rwkv7" to fla's own model
            from transformers import Rwkv7ForCausalLM

            self.model, info = Rwkv7ForCausalLM.from_pretrained(path, dtype=dtype, output_loading_info=True)
            self.model = self.model.to("cuda").eval()
            self.loading_class = type(self.model).__module__ + "." + type(self.model).__name__
            self.loading_info = {k: sorted(v) if isinstance(v, (set, list)) else str(v) for k, v in info.items()}
        if arm != "OFF" and not self.loading_class.startswith("transformers.models.rwkv7."):
            raise RuntimeError(f"{arm} loaded {self.loading_class}, not the transformers RWKV-7 model")
        if getattr(self, "loading_info", None) and (
            self.loading_info.get("missing_keys") or self.loading_info.get("unexpected_keys")
        ):
            raise RuntimeError(f"{arm} load has missing/unexpected keys: {self.loading_info}")
        self.model.eval()
        torch.set_grad_enabled(False)

    def _cache(self, out):
        return out.state if self.arm == "PR" else out.past_key_values

    def prefill(self, ids, attention_mask=None):
        import torch

        x = torch.tensor([ids] if isinstance(ids[0], int) else ids, device="cuda")
        kw = {"use_cache": True}
        if attention_mask is not None:
            kw["attention_mask"] = torch.tensor(attention_mask, device="cuda")
        out = self.model(input_ids=x, **kw)
        return out.logits.float(), self._cache(out)

    def step(self, tokens, cache):
        import torch

        x = torch.tensor(tokens, device="cuda").view(-1, 1)
        if self.arm == "PR":
            out = self.model(input_ids=x, state=cache, use_cache=True)
        else:
            out = self.model(input_ids=x, past_key_values=cache, use_cache=True)
        return out.logits[:, -1].float(), self._cache(out)


def _stored_positions(T):
    if T <= 512:
        return list(range(T))
    keep = set(range(0, T, 64)) | set(range(T - LONG_KEEP_TAIL, T))
    return sorted(keep)


def _dev(a, b):
    """max abs and max relative (per row: |a-b|_inf / |b|_inf) deviation of two [N, V] float tensors."""
    d = (a - b).abs()
    max_abs = float(d.max())
    rel = (d.max(dim=-1).values / b.abs().max(dim=-1).values.clamp_min(1e-30)).max()
    return max_abs, float(rel)


def _margin(row):
    import torch

    top = torch.topk(row, 2)
    return int(top.indices[0]), int(top.indices[1]), float(top.values[0] - top.values[1])


# ---------------------------------------------------------------------------------------------
# prepare: download + verify checkpoint, build and tokenize prompts (CPU)
# ---------------------------------------------------------------------------------------------


@app.function(image=img_rw_torch, volumes={"/vol": VOL}, cpu=4.0, memory=16384, timeout=3600)
def prepare_remote(repo: str) -> dict:
    import hashlib

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(f"RWKV/{repo}", files_metadata=True)
    sha = info.sha
    path = f"/vol/ckpt/{repo}/{sha}"
    snapshot_download(f"RWKV/{repo}", revision=sha, local_dir=path)

    report = {"repo": repo, "sha": sha, "path": path, "files": {}, "ok": True, "problems": []}
    lfs = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs is not None}
    manifest = None
    if os.path.exists(f"{path}/release-manifest.json"):
        manifest = json.load(open(f"{path}/release-manifest.json"))["files"]
    for name in sorted(lfs):
        h = hashlib.sha256()
        with open(f"{path}/{name}", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        digest = h.hexdigest()
        entry = {"sha256": digest, "hub_lfs": lfs[name], "hub_ok": digest == lfs[name]}
        if manifest is not None and name in manifest:
            entry["manifest"] = manifest[name]["sha256"]
            entry["manifest_ok"] = digest == manifest[name]["sha256"]
        report["files"][name] = entry
        if not entry["hub_ok"] or entry.get("manifest_ok") is False:
            report["ok"] = False
            report["problems"].append(f"hash mismatch: {name}")
    if manifest is not None:  # small files listed by the manifest too
        for name, m in manifest.items():
            if name in lfs or not os.path.exists(f"{path}/{name}"):
                continue
            digest = hashlib.sha256(open(f"{path}/{name}", "rb").read()).hexdigest()
            if digest != m["sha256"]:
                report["problems"].append(f"manifest mismatch (small file): {name}")
                report.setdefault("manifest_small_mismatch", []).append(name)
    idx_path = f"{path}/model.safetensors.index.json"
    if os.path.exists(idx_path):
        report["tensor_count"] = len(json.load(open(idx_path))["weight_map"])
    else:
        from safetensors import safe_open

        with safe_open(f"{path}/model.safetensors", "pt") as f:
            report["tensor_count"] = len(list(f.keys()))

    # prompts ------------------------------------------------------------------------------
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    ds_rev = getattr(ds, "_fingerprint", None)
    try:
        ds_rev = api.dataset_info("EleutherAI/lambada_openai").sha
    except Exception:  # noqa: BLE001
        pass
    prompts = []
    for i in range(24):
        text = ds[i]["text"]
        prompt = text.rsplit(" ", 1)[0]
        prompts.append({"id": f"lambada-{i:02d}", "text": prompt, "ids": tok.encode(prompt)})
    readme_ids = None
    try:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": "Explain why RWKV uses constant state."}],
            tokenize=True, add_generation_prompt=True, thinking=False, return_dict=True,
        )
        readme_ids = list(enc["input_ids"])
    except Exception as e:  # noqa: BLE001
        report["problems"].append(f"readme chat template: {type(e).__name__}: {e}")
    if readme_ids is not None:
        prompts.append({"id": "readme", "text": "<chat template: Explain why RWKV uses constant state.>",
                        "ids": readme_ids})
    long_ids, j = [], 24
    while len(long_ids) < LONG_LEN:
        long_ids += tok.encode(ds[j]["text"] + "\n\n")
        j += 1
    prompts.append({"id": "long-2048", "text": f"<lambada test {24}..{j - 1} concatenated, cut at {LONG_LEN}>",
                    "ids": long_ids[:LONG_LEN]})
    bundle = {"repo": repo, "sha": sha, "dataset": "EleutherAI/lambada_openai/en/test", "dataset_revision": ds_rev,
              "tokenizer": f"{repo}@{sha} (trust_remote_code tokenizer)", "prompts": prompts}
    json.dump(bundle, open(f"{path}/../prompts.json", "w"))
    json.dump({"path": path, "sha": sha}, open(f"/vol/ckpt/{repo}/CURRENT.json", "w"))
    json.dump(report, open(f"/vol/ckpt/{repo}/verify.json", "w"), indent=1)
    VOL.commit()
    report["prompts"] = [{"id": p["id"], "n_tokens": len(p["ids"])} for p in prompts]
    report["prompt_bundle"] = bundle
    return report


@app.local_entrypoint()
def prepare(repo: str):
    r = prepare_remote.remote(repo)
    os.makedirs(os.path.join(HERE, "logs", "gpu"), exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = r.pop("prompt_bundle")
    json.dump(r, open(os.path.join(HERE, "logs", "gpu", f"prepare-{repo}-{stamp}.json"), "w"), indent=1)
    json.dump(bundle, open(os.path.join(HERE, "logs", "gpu", f"prompts-{repo}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "files"}, indent=1))
    bad = [n for n, e in r["files"].items() if not e["hub_ok"] or e.get("manifest_ok") is False]
    print("weight files:", len(r["files"]), "hash problems:", bad)


# ---------------------------------------------------------------------------------------------
# measure: one arm, one checkpoint, one dtype (GPU)
# ---------------------------------------------------------------------------------------------


def _measure(repo, arm, dtype_name, save, variant, batch_check, packed_check=False):
    import time

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    path, meta = _ckpt_dir(repo)
    verify = json.load(open(f"/vol/ckpt/{repo}/verify.json"))
    if not verify["ok"]:
        raise RuntimeError(f"checkpoint verification failed: {verify['problems']}")
    bundle = json.load(open(f"{path}/../prompts.json"))
    sha = meta["sha"]
    result = {"repo": repo, "ckpt_sha": sha, "arm": arm, "dtype": dtype_name, "variant": variant,
              "env": _env_info(), "started": time.time(), "prompts": {}}
    if arm.startswith("RW"):
        result["wkv_impl"] = _resolved_wkv()
        if variant == "safe_gate64":
            import functools

            import transformers.models.rwkv7.modeling_rwkv7 as M
            from fla.ops.rwkv7 import chunk_rwkv7

            M.torch_rwkv7 = functools.partial(chunk_rwkv7, safe_gate=True, chunk_size=64)
            result["wkv_impl"] = "fla.ops.rwkv7.chunk_rwkv7(safe_gate=True, chunk_size=64) [injected]"
        if arm == "RW-fla" and "fla" not in result["wkv_impl"]:
            raise RuntimeError(f"RW-fla resolved to {result['wkv_impl']}; refusing to report it as fla")
        if arm == "RW-torch" and "fla" in result["wkv_impl"]:
            raise RuntimeError(f"RW-torch resolved to {result['wkv_impl']}")
    t0 = time.time()
    ad = Adapter(arm, path, dtype)
    result["load_secs"] = round(time.time() - t0, 1)
    result["model_class"] = getattr(ad, "loading_class", "official runtime RWKV7ForCausalLM")
    result["loading_info"] = getattr(ad, "loading_info", None)

    anchor_root = f"/vol/anchor/{repo}/{sha}"
    fp32_anchor = f"{anchor_root}/OFF-float32"
    same_anchor = f"{anchor_root}/OFF-{dtype_name}"
    rwt_anchor = f"{anchor_root}/RW-torch-{dtype_name}"
    save_dir = f"{anchor_root}/{arm}-{dtype_name}" if save else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def load_anchor(root, pid):
        p = f"{root}/{pid}.pt"
        return torch.load(p) if os.path.exists(p) else None

    for p in bundle["prompts"]:
        pid, ids = p["id"], p["ids"]
        T = len(ids)
        rec = {"n_tokens": T}
        t1 = time.time()
        logits, cache = ad.prefill(ids)
        logits = logits[0]
        pos = _stored_positions(T)
        pre = logits[pos].cpu()
        rec["prefill_secs"] = round(time.time() - t1, 2)
        # free decode
        free, free_logits = [], []
        nxt = int(logits[-1].argmax())
        for _ in range(N_NEW):
            free.append(nxt)
            row, cache = ad.step([nxt], cache)
            free_logits.append(row[0].cpu())
            nxt = int(row[0].argmax())
        rec["free_tokens"] = free
        last_rows = torch.cat([logits[-1:].cpu(), torch.stack(free_logits[:-1])])  # logits that chose each free token
        rec["free_margins"] = [round(_margin(r)[2], 5) for r in last_rows]
        if arm.startswith("RW") and pid.startswith("lambada") and int(pid[-2:]) < 4:
            x = torch.tensor([ids], device="cuda")
            g = ad.model.generate(x, max_new_tokens=N_NEW, do_sample=False, pad_token_id=0)[0, T:].tolist()
            rec["generate_equals_manual"] = g == free[: len(g)]
            if not rec["generate_equals_manual"]:
                rec["generate_tokens"] = g

        def forced(along):
            logits0, c = ad.prefill(ids)
            rows = [logits0[0, -1].cpu()]
            for t in along[:-1]:
                r, c = ad.step([t], c)
                rows.append(r[0].cpu())
            return torch.stack(rows)  # row k = logits predicting token k of `along`

        comps = {}
        for name, root in (("vs_OFF_fp32", fp32_anchor), ("vs_OFF_same", same_anchor), ("vs_RWtorch_same", rwt_anchor)):
            if root == save_dir or (name == "vs_OFF_same" and dtype_name == "float32"):
                continue
            a = load_anchor(root, pid)
            if a is None:
                continue
            c = {}
            c["prefill_max_abs"], c["prefill_max_rel"] = _dev(pre, a["prefill"])
            along = a["free_tokens"]
            frows = forced(along)
            c["forced_max_abs"], c["forced_max_rel"] = _dev(frows, a["free_logits"])
            c["free_identical"] = free == along
            k = next((i for i, (x, y) in enumerate(zip(free, along)) if x != y), None)
            c["first_divergence"] = k
            if k is not None:
                arow = a["free_logits"][k]
                i1, i2, m = _margin(arow)
                d1 = float(frows[k, i1] - arow[i1])
                d2 = float(frows[k, i2] - arow[i2])
                c["divergence"] = {"anchor_top1": i1, "anchor_top2": i2, "anchor_margin": m, "delta_top1": d1,
                                   "delta_top2": d2, "arm_token": free[k], "anchor_token": along[k],
                                   "explained_by_precision": m < abs(d1) + abs(d2)}
            comps[name] = c
        rec["comparisons"] = comps
        if save_dir:
            # free_logits row k = the logits that predicted free token k (row 0 = last prefill position)
            torch.save({"positions": pos, "prefill": pre, "free_tokens": free, "free_logits": last_rows},
                       f"{save_dir}/{pid}.pt")
        rec["secs"] = round(time.time() - t1, 2)
        result["prompts"][pid] = rec

    if batch_check and arm.startswith("RW"):
        lam = [p for p in bundle["prompts"] if p["id"].startswith("lambada")]
        bres = []
        for b0 in range(0, len(lam), 4):
            group = lam[b0 : b0 + 4]
            L = max(len(p["ids"]) for p in group)
            ids = [[0] * (L - len(p["ids"])) + p["ids"] for p in group]
            mask = [[0] * (L - len(p["ids"])) + [1] * len(p["ids"]) for p in group]
            blogits, bcache = ad.prefill(ids, attention_mask=mask)
            rows = [blogits[:, -1].cpu()]  # rows[k][r] = batched logits predicting token k of row r
            toks = [rows[0].argmax(-1).tolist()]
            for _ in range(N_NEW - 1):
                lg, bcache = ad.step(toks[-1], bcache)
                rows.append(lg.cpu())
                toks.append(rows[-1].argmax(-1).tolist())
            x = torch.tensor(ids, device="cuda")
            m = torch.tensor(mask, device="cuda")
            gen = ad.model.generate(x, attention_mask=m, max_new_tokens=N_NEW, do_sample=False, pad_token_id=0)
            for r, p in enumerate(group):
                single = result["prompts"][p["id"]]["free_tokens"]
                bt = [toks[k][r] for k in range(N_NEW)]
                entry = {"id": p["id"], "tokens_identical": bt == single,
                         "generate_equals_manual_batched": gen[r, L:].tolist() == bt}
                # single-sequence logits along its own continuation, for the margins
                slog, sc = ad.prefill(p["ids"])
                srows = [slog[0, -1].cpu()]
                for t_ in single[:-1]:
                    lg, sc = ad.step([t_], sc)
                    srows.append(lg[0].cpu())
                entry["last_prefill_logit_max_abs"] = float((rows[0][r] - srows[0]).abs().max())
                k = next((i for i, (a_, b_) in enumerate(zip(bt, single)) if a_ != b_), None)
                if k is not None:
                    i1, i2, mg = _margin(srows[k])
                    d1 = float(rows[k][r][i1] - srows[k][i1])
                    d2 = float(rows[k][r][i2] - srows[k][i2])
                    entry["divergence"] = {"k": k, "single_top1": i1, "single_top2": i2, "single_margin": mg,
                                           "delta_top1": d1, "delta_top2": d2, "batched_token": bt[k],
                                           "explained_by_precision": mg < abs(d1) + abs(d2)}
                bres.append(entry)
        result["batch_left_padding"] = bres

    if packed_check and arm.startswith("RW"):
        group = [p for p in bundle["prompts"] if p["id"].startswith("lambada")][:4]
        flat = [t_ for p in group for t_ in p["ids"]]
        bounds = [0]
        for p in group:
            bounds.append(bounds[-1] + len(p["ids"]))
        cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
        mx = max(len(p["ids"]) for p in group)
        pres = {"ids": [p["id"] for p in group], "cu_seq_lens": bounds}
        try:
            out = ad.model(input_ids=torch.tensor([flat], device="cuda"), cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                           max_length_q=mx, max_length_k=mx, use_cache=False).logits[0].float().cpu()
            segs = []
            for i, p in enumerate(group):
                seg = out[bounds[i] : bounds[i + 1]]
                alone = ad.model(input_ids=torch.tensor([p["ids"]], device="cuda"), use_cache=False).logits[0].float().cpu()
                segs.append({"id": p["id"], "max_abs": float((seg - alone).abs().max()),
                             "argmax_equal_all_positions": bool((seg.argmax(-1) == alone.argmax(-1)).all())})
            pres["segments"] = segs
            # control: the same packed row WITHOUT boundaries must differ from the sequences alone
            nob = ad.model(input_ids=torch.tensor([flat], device="cuda"), use_cache=False).logits[0].float().cpu()
            pres["control_no_boundaries_max_abs_seg2"] = float(
                (nob[bounds[1] : bounds[2]] - ad.model(input_ids=torch.tensor([group[1]["ids"]], device="cuda"),
                                                        use_cache=False).logits[0].float().cpu()).abs().max())
        except Exception as e:  # noqa: BLE001
            pres["error"] = f"{type(e).__name__}: {e}"
        result["packed"] = pres

    result["finished"] = time.time()
    if save_dir:
        VOL.commit()
    return result


@app.function(image=img_rw_torch, gpu=GPU, volumes={"/vol": VOL}, timeout=3600, memory=65536)
def measure_rw_torch(repo, arm, dtype, save, variant, batch_check, packed_check=False):
    return _measure(repo, arm, dtype, save, variant, batch_check, packed_check)


@app.function(image=img_rw_fla, gpu=GPU, volumes={"/vol": VOL}, timeout=3600, memory=65536)
def measure_rw_fla(repo, arm, dtype, save, variant, batch_check, packed_check=False):
    return _measure(repo, arm, dtype, save, variant, batch_check, packed_check)


@app.function(image=img_pr, gpu=GPU, volumes={"/vol": VOL}, timeout=3600, memory=65536)
def measure_pr(repo, arm, dtype, save, variant, batch_check, packed_check=False):
    return _measure(repo, arm, dtype, save, variant, batch_check, packed_check)


@app.local_entrypoint()
def measure(repo: str, arm: str, dtype: str = "float32", save: bool = False, variant: str = "",
            batch_check: bool = False, packed_check: bool = False):
    fn = {"OFF": measure_rw_torch, "RW-torch": measure_rw_torch, "RW-fla": measure_rw_fla, "PR": measure_pr}[arm]
    shas = {
        "RW": __import__("subprocess").run(["git", "-C", RW_TREE, "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip(),
        "PR": PR_SHA,
    }
    r = fn.remote(repo, arm, dtype, save, variant, batch_check, packed_check)
    r["code_sha"] = shas["PR"] if arm == "PR" else shas["RW"]
    r["rw_tree_dirty"] = bool(__import__("subprocess").run(["git", "-C", RW_TREE, "status", "--porcelain"],
                                                           capture_output=True, text=True).stdout.strip())
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(os.path.join(HERE, "logs", "gpu"), exist_ok=True)
    tag = f"{repo}-{arm}-{dtype}{('-' + variant) if variant else ''}"
    out = os.path.join(HERE, "logs", "gpu", f"measure-{tag}-{stamp}.json")
    json.dump(r, open(out, "w"), indent=1)
    # short summary
    print(f"{tag}: env={r['env'].get('gpu')} transformers={r['env'].get('transformers')} "
          f"wkv={r.get('wkv_impl', '-')} load={r['load_secs']}s total={round(r['finished'] - r['started'])}s")
    for pid, rec in r["prompts"].items():
        cs = " | ".join(
            f"{n}: free_eq={c['free_identical']} pre={c['prefill_max_abs']:.2e} forced={c['forced_max_abs']:.2e}"
            + (f" div@{c['first_divergence']} m={c['divergence']['anchor_margin']:.3f} "
               f"expl={c['divergence']['explained_by_precision']}" if c.get("divergence") else "")
            for n, c in rec["comparisons"].items())
        print(f"  {pid:10s} T={rec['n_tokens']:5d} {rec['secs']:6.1f}s {cs}")
    if r.get("batch_left_padding"):
        b = r["batch_left_padding"]
        print("  batch/left-pad: tokens identical", sum(x["tokens_identical"] for x in b), "/", len(b),
              "max last-prefill-logit |d|", max(x["last_prefill_logit_max_abs"] for x in b),
              "generate==manual batched", sum(x["generate_equals_manual_batched"] for x in b), "/", len(b))
        for x in b:
            if "divergence" in x:
                print("    batch divergence:", x["id"], x["divergence"])
    if r.get("packed"):
        print("  packed:", json.dumps(r["packed"]))
    print("log:", out)


# ---------------------------------------------------------------------------------------------
# minimal reproduction: does importing the rework's modeling module change Auto* resolution?
# ---------------------------------------------------------------------------------------------


@app.function(image=img_rw_fla, cpu=2.0, memory=8192, timeout=900)
def auto_hijack_repro() -> dict:
    import importlib.util
    import os as _os

    out = {}
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    out["transformers"] = transformers.__version__
    out["1_config_class_before"] = CONFIG_MAPPING["rwkv7"].__module__ + "." + CONFIG_MAPPING["rwkv7"].__name__
    cfg = AutoConfig.for_model("rwkv7")
    out["2_automodel_before"] = AutoModelForCausalLM._model_mapping[type(cfg)].__module__
    import transformers.models.rwkv7.modeling_rwkv7 as M  # noqa: F401  (this is what from_pretrained imports)

    out["3_config_class_after_import"] = CONFIG_MAPPING["rwkv7"].__module__ + "." + CONFIG_MAPPING["rwkv7"].__name__
    cfg2 = AutoConfig.for_model("rwkv7")
    out["4_automodel_after_import"] = AutoModelForCausalLM._model_mapping[type(cfg2)].__module__
    spec = importlib.util.find_spec("fla")
    root = _os.path.dirname(spec.origin)
    hits = []
    for dirpath, _, files in _os.walk(_os.path.join(root, "models", "rwkv7")):
        for fn in files:
            if fn.endswith(".py"):
                for i, line in enumerate(open(_os.path.join(dirpath, fn)), 1):
                    if "register(" in line or "model_type" in line:
                        hits.append(f"{_os.path.relpath(_os.path.join(dirpath, fn), root)}:{i}: {line.strip()}")
    out["5_fla_registration_lines"] = hits
    return out


@app.local_entrypoint()
def hijack():
    r = auto_hijack_repro.remote()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json.dump(r, open(os.path.join(HERE, "logs", "gpu", f"auto-hijack-repro-{stamp}.json"), "w"), indent=1)
    print(json.dumps(r, indent=1))


@app.function(image=img_rw_fla, cpu=2.0, memory=8192, timeout=900)
def fla_collisions() -> dict:
    import importlib

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING, CONFIG_MAPPING_NAMES

    native = set(CONFIG_MAPPING_NAMES)
    before = {t: CONFIG_MAPPING[t].__module__ for t in native}
    importlib.import_module("fla")  # what use_kernel_func_from_hub_with_fallback does for any fla kernel
    after = {t: CONFIG_MAPPING[t].__module__ for t in native}
    hijacked = sorted(t for t in native if before[t] != after[t])
    fla_types = sorted(k for k in CONFIG_MAPPING._extra_content)
    # which native models import fla through the kernel helper?
    import pathlib
    import re

    root = pathlib.Path("/repo/src/transformers/models")
    users = sorted({p.parent.name for p in root.rglob("modeling_*.py")
                    if re.search(r'use_kernel_func_from_hub_with_fallback\([^)]*"fla"', p.read_text())})
    return {"fla_registered_types": fla_types, "native_types_hijacked_by_import_fla": hijacked,
            "native_models_that_import_fla_via_kernel_helper": users}


@app.local_entrypoint()
def collisions():
    r = fla_collisions.remote()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json.dump(r, open(os.path.join(HERE, "logs", "gpu", f"fla-collisions-{stamp}.json"), "w"), indent=1)
    print(json.dumps(r, indent=1))


@app.function(image=img_rw_fla, cpu=2.0, memory=8192, timeout=900)
def qwen3_next_hijack() -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    out = {"mamba_before": CONFIG_MAPPING["mamba"].__module__}
    import transformers.models.qwen3_next.modeling_qwen3_next  # noqa: F401  (unrelated model, on main already)

    out["mamba_after_importing_qwen3_next"] = CONFIG_MAPPING["mamba"].__module__
    cfg = AutoConfig.for_model("mamba", vocab_size=64, hidden_size=16, num_hidden_layers=1)
    out["AutoModelForCausalLM(mamba config)"] = type(AutoModelForCausalLM.from_config(cfg)).__module__
    return out


@app.local_entrypoint()
def qwen3next():
    r = qwen3_next_hijack.remote()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json.dump(r, open(os.path.join(HERE, "logs", "gpu", f"qwen3next-mamba-hijack-{stamp}.json"), "w"), indent=1)
    print(json.dumps(r, indent=1))


@app.function(image=img_rw_torch, volumes={"/vol": VOL}, cpu=4.0, memory=32768, timeout=3600)
def compare_anchors_remote(repo: str, a: str, b: str) -> dict:
    """Offline comparison of two saved anchors (no forced leg): prefill deviation, token identity, margins."""
    import torch

    path, meta = _ckpt_dir(repo)
    root = f"/vol/anchor/{repo}/{meta['sha']}"
    bundle = json.load(open(f"{path}/../prompts.json"))
    out = {"repo": repo, "a": a, "b": b, "sha": meta["sha"], "prompts": {}}
    for p in bundle["prompts"]:
        A = torch.load(f"{root}/{a}/{p['id']}.pt")
        B = torch.load(f"{root}/{b}/{p['id']}.pt")
        mx, rel = _dev(A["prefill"], B["prefill"])
        k = next((i for i, (x, y) in enumerate(zip(A["free_tokens"], B["free_tokens"])) if x != y), None)
        rec = {"prefill_max_abs": mx, "prefill_max_rel": rel, "free_identical": A["free_tokens"] == B["free_tokens"],
               "first_divergence": k}
        if k is not None:
            i1, i2, m = _margin(B["free_logits"][k])
            rec["b_margin_at_k"] = m
        out["prompts"][p["id"]] = rec
    return out


@app.local_entrypoint()
def compare_anchors(repo: str, a: str = "OFF-bfloat16", b: str = "OFF-float32"):
    r = compare_anchors_remote.remote(repo, a, b)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    f = os.path.join(HERE, "logs", "gpu", f"anchors-{repo}-{a}-vs-{b}-{stamp}.json")
    json.dump(r, open(f, "w"), indent=1)
    ps = r["prompts"]
    print(f"{repo} {a} vs {b}: identical {sum(x['free_identical'] for x in ps.values())}/{len(ps)} "
          f"prefill max abs {max(x['prefill_max_abs'] for x in ps.values()):.2e} rel {max(x['prefill_max_rel'] for x in ps.values()):.2e}")
    top = sorted(((x["prefill_max_abs"], p) for p, x in ps.items()), reverse=True)[:5]
    print("  top prefill abs:", [(p, round(v, 2)) for v, p in top])
    print("  log:", f)
