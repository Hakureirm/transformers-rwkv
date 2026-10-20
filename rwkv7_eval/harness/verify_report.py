"""Re-derive the numbers quoted in REPORT.md from the raw logs and flag mismatches."""

import glob
import json
import os
import re
import subprocess

E = os.path.expanduser("~/workspace/rwkv7-tf-rework/rwkv7_eval")
REPO = os.path.expanduser("~/workspace/rwkv7-tf-rework")
G = os.path.join(E, "logs", "gpu")
results = []


def check(name, ok, detail=""):
    results.append((ok, name, detail))


def runs():
    out = []
    for f in sorted(glob.glob(os.path.join(G, "measure-*.json"))):
        r = json.load(open(f))
        r["_invalid"] = r["arm"].startswith("RW") and not (r.get("model_class") or "").startswith("transformers.")
        out.append(r)
    return out


R = runs()


def latest(repo, arm, dtype, variant=""):
    c = [r for r in R if r["repo"] == repo and r["arm"] == arm and r["dtype"] == dtype
         and (r.get("variant") or "") == variant and not r["_invalid"]]
    return c[-1] if c else None


def stats(r, comp="vs_OFF_fp32"):
    cs = [rec["comparisons"][comp] for rec in r["prompts"].values() if comp in rec["comparisons"]]
    return (sum(c["free_identical"] for c in cs), len(cs), max(c["prefill_max_rel"] for c in cs),
            max(c["forced_max_rel"] for c in cs))


# §4.2 fp32 table
table = [("RWKV7-G1k-1.5B-20260930", "PR", 26, 1.26e-5, 5.34e-6), ("RWKV7-G1k-1.5B-20260930", "RW-torch", 26, 9.73e-6, 1.58e-6),
         ("RWKV7-G1k-1.5B-20260930", "RW-fla", 26, 1.49e-3, 1.40e-3), ("RWKV7-G1k-2.9B-20260930", "RW-torch", 26, 1.08e-5, 1.49e-6),
         ("RWKV7-G1k-2.9B-20260930", "RW-fla", 25, 2.60e-3, 3.58e-3), ("RWKV7-G1k-7.2B-20260930", "RW-torch", 26, 1.27e-5, 7.76e-6),
         ("RWKV7-G1k-7.2B-20260930", "RW-fla", 24, 4.13e-3, 1.51e-3), ("RWKV7-G1j-1.5B-20260831", "RW-torch", 26, 1.31e-5, 1.96e-6),
         ("RWKV7-G1j-1.5B-20260831", "RW-fla", 25, 1.66e-3, 6.12e-4), ("RWKV7-1.5B-20260805", "RW-torch", 26, 1.03e-5, 1.57e-6),
         ("RWKV7-1.5B-20260805", "RW-fla", 26, 2.15e-3, 7.41e-4)]
for repo, arm, ident, pre, fd in table:
    r = latest(repo, arm, "float32")
    i, n, p, f = stats(r)
    ok = i == ident and n == 26 and abs(p - pre) / pre < 0.01 and abs(f - fd) / fd < 0.01
    check(f"fp32 table {repo} {arm}", ok, f"log: {i}/{n} pre {p:.3e} forced {f:.3e}")

# headline: RW-torch fp32 max rel <= 1.3e-5 on all five
mx = max(stats(latest(rp, "RW-torch", "float32"))[2] for rp in
         ["RWKV7-G1k-1.5B-20260930", "RWKV7-G1k-2.9B-20260930", "RWKV7-G1k-7.2B-20260930", "RWKV7-G1j-1.5B-20260831", "RWKV7-1.5B-20260805"])
check("RW-torch fp32 max rel <= 1.3e-5 (headline says <= 1.3e-5)", mx <= 1.32e-5, f"{mx:.3e}")

# fla divergence details: margins 0.00035-0.00082, sums 0.0019-0.0050, k 29-40
ms, ss, ks = [], [], []
for repo in ["RWKV7-G1k-2.9B-20260930", "RWKV7-G1k-7.2B-20260930", "RWKV7-G1j-1.5B-20260831"]:
    for rec in latest(repo, "RW-fla", "float32")["prompts"].values():
        c = rec["comparisons"]["vs_OFF_fp32"]
        if not c["free_identical"]:
            d = c["divergence"]
            ms.append(d["anchor_margin"]); ss.append(abs(d["delta_top1"]) + abs(d["delta_top2"])); ks.append(c["first_divergence"])
            if not d["explained_by_precision"]:
                check("fla fp32 divergence explained", False, str(d))
check("fla fp32 divergences count == 4", len(ms) == 4, str(len(ms)))
check("fla margins range 0.00035-0.00082", round(min(ms), 5) == 0.00035 and round(max(ms), 5) == 0.00082, f"{min(ms):.5f}-{max(ms):.5f}")
check("fla sums range 0.0019-0.0050", round(min(ss), 4) == 0.0019 and round(max(ss), 3) == 0.005, f"{min(ss):.5f}-{max(ss):.5f}")
check("fla k range 29-40", min(ks) == 29 and max(ks) == 40, f"{min(ks)}-{max(ks)}")

# Q2
pr32, rw32 = latest("RWKV7-G1k-1.5B-20260930", "PR", "float32"), latest("RWKV7-G1k-1.5B-20260930", "RW-torch", "float32")
same32 = sum(pr32["prompts"][p]["free_tokens"] == rw32["prompts"][p]["free_tokens"] for p in pr32["prompts"])
pr16, rw16 = latest("RWKV7-G1k-1.5B-20260930", "PR", "bfloat16"), latest("RWKV7-G1k-1.5B-20260930", "RW-torch", "bfloat16")
same16 = sum(pr16["prompts"][p]["free_tokens"] == rw16["prompts"][p]["free_tokens"] for p in pr16["prompts"])
check("Q2 fp32 PR==RW 26/26", same32 == 26, str(same32))
check("Q2 bf16 PR==RW-torch 13/26", same16 == 13, str(same16))

# bf16 unexplained list (6) and tie classification
unex = []
for r in R:
    if r["_invalid"] or r["dtype"] != "bfloat16" or r.get("variant"):
        continue
    for comp in ("vs_OFF_fp32", "vs_OFF_same", "vs_RWtorch_same"):
        for p, rec in r["prompts"].items():
            c = rec["comparisons"].get(comp)
            if c and not c["free_identical"] and not c["divergence"]["explained_by_precision"]:
                d = c["divergence"]
                tie = abs(d["anchor_margin"] - abs(d["delta_top1"]) - abs(d["delta_top2"])) < 1e-3
                unex.append((r["repo"].split("-")[2], r["arm"], comp, p, tie))
uniq = sorted(set(unex))
check("bf16 unexplained cases == 6 (unique repo/arm/comp/prompt)", len(uniq) == 6, str(uniq))

# 7.2B bf16 vs official own bf16
a = [json.load(open(f)) for f in glob.glob(os.path.join(G, "anchors-RWKV7-G1k-7.2B-*.json"))][-1]
mx = max(x["prefill_max_abs"] for x in a["prompts"].values()); rl = max(x["prefill_max_rel"] for x in a["prompts"].values())
check("7.2B OFF bf16 vs fp32 17.2 / 0.165", round(mx, 1) == 17.2 and round(rl, 3) == 0.165, f"{mx:.2f} {rl:.3f}")
rt = latest("RWKV7-G1k-7.2B-20260930", "RW-torch", "bfloat16")
cs = [rec["comparisons"]["vs_OFF_fp32"] for rec in rt["prompts"].values()]
check("7.2B RW-torch bf16 16.6 / 0.156", round(max(c["prefill_max_abs"] for c in cs), 1) == 16.6, f"{max(c['prefill_max_abs'] for c in cs):.2f}")

# perf numbers
def pre(repo, arm, dtype):
    return latest(repo, arm, dtype)["prompts"]["long-2048"]["prefill_secs"]
check("perf torch 1.5B fp32 7.7s", round(pre("RWKV7-G1k-1.5B-20260930", "RW-torch", "float32"), 1) == 7.7)
check("perf torch 7.2B fp32 13.6s", round(pre("RWKV7-G1k-7.2B-20260930", "RW-torch", "float32"), 1) == 13.6)
check("perf torch 13.3B bf16 23.4s", round(pre("RWKV7-G1k-13.3B-20260930", "RW-torch", "bfloat16"), 1) == 23.4)
check("perf PR 1.5B fp32 0.65s (rounded 0.66 in log?)", abs(pre("RWKV7-G1k-1.5B-20260930", "PR", "float32") - 0.65) < 0.02,
      str(pre("RWKV7-G1k-1.5B-20260930", "PR", "float32")))

# checkpoint hashes / tensor counts
for f in sorted(glob.glob(os.path.join(G, "prepare-*.json"))):
    p = json.load(open(f))
    allok = p["ok"] and all(e["hub_ok"] and e.get("manifest_ok") is not False for e in p["files"].values())
    check(f"checkpoint verified {p['repo']} ({p['tensor_count']})", allok)

# batch / packed
fla = [r for r in R if r["repo"] == "RWKV7-G1k-1.5B-20260930" and r["arm"] == "RW-fla" and r["dtype"] == "float32"
       and not r["_invalid"] and r.get("packed")][-1]
b = fla["batch_left_padding"]
d = [x for x in b if "divergence" in x]
check("batch fla 23/24, lambada-16 k=20 explained", sum(x["tokens_identical"] for x in b) == 23 and d[0]["id"] == "lambada-16"
      and d[0]["divergence"]["k"] == 20 and d[0]["divergence"]["explained_by_precision"])
check("packed max_abs <= 0.036 and control 13.99", max(s["max_abs"] for s in fla["packed"]["segments"]) <= 0.036
      and round(fla["packed"]["control_no_boundaries_max_abs_seg2"], 2) == 13.99)

# commits referenced exist
txt = open(os.path.join(REPO, "REPORT.md"), encoding="utf-8").read()
for sha in sorted(set(re.findall(r"`([0-9a-f]{10})`", txt))):
    ok = subprocess.run(["git", "-C", REPO, "cat-file", "-e", sha], capture_output=True).returncode == 0
    check(f"commit {sha} exists", ok)
# log paths referenced exist (glob patterns)
for pat in sorted(set(re.findall(r"`((?:rwkv7_eval/)?logs/[^`]+)`", txt))):
    full = pat if pat.startswith("rwkv7_eval/") else "rwkv7_eval/" + pat
    hits = glob.glob(os.path.join(REPO, full))
    check(f"log path {pat}", bool(hits), "" if hits else "NOT FOUND")
# billing
b6 = open(os.path.join(E, "logs", "billing-06-final.txt")).read()
check("billing final credits 10.50 / billed 0.00", "-10.50" in b6 and "$0.00" in b6)

bad = [x for x in results if not x[0]]
for ok, name, detail in results:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
print(f"\n{len(results) - len(bad)}/{len(results)} claims verified; {len(bad)} failing")
