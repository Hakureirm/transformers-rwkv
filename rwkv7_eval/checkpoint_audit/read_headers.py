"""Read config.json + every safetensors header of official RWKV-7 checkpoints via HTTP Range.
No weights are downloaded: only the 8-byte length prefix and the JSON header of each shard."""
import json, struct, sys, urllib.request, os
REPOS = ["RWKV7-G1k-1.5B-20260930", "RWKV7-G1k-2.9B-20260930", "RWKV7-G1k-7.2B-20260930",
         "RWKV7-G1k-13.3B-20260930", "RWKV7-G1j-1.5B-20260831", "RWKV7-1.5B-20260805"]
OUT = os.path.dirname(os.path.abspath(__file__))

def get(url, rng=None, tries=6):
    import time
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            print("  retry", i + 1, type(e).__name__, file=sys.stderr, flush=True)
            time.sleep(2 * (i + 1))

for repo in REPOS:
    if os.path.exists(os.path.join(OUT, repo + ".json")):
        print(repo, "cached"); continue
    api = json.loads(get(f"https://huggingface.co/api/models/RWKV/{repo}"))
    sha = api["sha"]
    base = f"https://huggingface.co/RWKV/{repo}/resolve/{sha}/"
    cfg = json.loads(get(base + "config.json"))
    shards = sorted(s["rfilename"] for s in api["siblings"] if s["rfilename"].endswith(".safetensors"))
    tensors = {}
    for shard in shards:
        n = struct.unpack("<Q", get(base + shard, "bytes=0-7"))[0]
        hdr = json.loads(get(base + shard, f"bytes=8-{8 + n - 1}"))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            tensors[k] = {"dtype": v["dtype"], "shape": v["shape"], "shard": shard}
    json.dump({"repo": repo, "sha": sha, "config": cfg, "shards": shards, "tensors": tensors},
              open(os.path.join(OUT, repo + ".json"), "w"), indent=1, sort_keys=True)
    print(repo, sha[:12], "shards=%d" % len(shards), "tensors=%d" % len(tensors), flush=True)
