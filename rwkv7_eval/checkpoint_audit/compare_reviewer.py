"""Compare official checkpoint tensors (names/shapes) against the parameters the reviewer
Rwkv7ForCausalLM would create, after the candidate rename mapping. No torch needed."""
import json, re, sys, glob, os

RENAMES = [  # candidate conversion mapping (checkpoint key -> model key)
    (r"^rwkv7\.emb\.", "model.embed_tokens."),
    (r"^rwkv7\.blocks\.(\d+)\.", r"model.layers.\1."),
    (r"^rwkv7\.ln_out\.", "model.norm."),
    (r"^head\.", "lm_head."),
]
REVIEWER_CFG_FIELDS = {"vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_heads",
    "head_dim", "decay_low_rank_dim", "a_low_rank_dim", "v_low_rank_dim", "gate_low_rank_dim", "norm_eps",
    "tie_word_embeddings", "use_cache", "bos_token_id", "eos_token_id", "pad_token_id"}
GENERIC = {"architectures", "model_type", "transformers_version", "torch_dtype", "dtype", "auto_map"}

def expected(c):
    C, H, N, I = c["hidden_size"], c["num_heads"], c["head_dim"], c["intermediate_size"]
    e = {"model.embed_tokens.weight": [c["vocab_size"], C], "model.norm.weight": [C], "model.norm.bias": [C],
         "lm_head.weight": [c["vocab_size"], C]}
    for L in range(c["num_hidden_layers"]):
        p = f"model.layers.{L}."
        if L == 0:
            e[p + "ln0.weight"] = [C]; e[p + "ln0.bias"] = [C]
        for n in ("ln1", "ln2"):
            e[p + n + ".weight"] = [C]; e[p + n + ".bias"] = [C]
        a = p + "att."
        for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "v0", "k_k", "k_a"):
            e[a + n] = [1, 1, C]
        e[a + "w1"] = [C, c["decay_low_rank_dim"]]; e[a + "w2"] = [c["decay_low_rank_dim"], C]
        e[a + "a1"] = [C, c["a_low_rank_dim"]]; e[a + "a2"] = [c["a_low_rank_dim"], C]
        e[a + "v1"] = [C, c["v_low_rank_dim"]]; e[a + "v2"] = [c["v_low_rank_dim"], C]
        e[a + "g1"] = [C, c["gate_low_rank_dim"]]; e[a + "g2"] = [c["gate_low_rank_dim"], C]
        e[a + "r_k"] = [H, N]
        for n in ("receptance", "key", "value", "output"):
            e[a + n + ".weight"] = [C, C]
        e[a + "ln_x.weight"] = [C]; e[a + "ln_x.bias"] = [C]
        f = p + "ffn."
        e[f + "x_k"] = [1, 1, C]; e[f + "key.weight"] = [I, C]; e[f + "value.weight"] = [C, I]
    return e

def rename(k):
    for pat, rep in RENAMES:
        k2 = re.sub(pat, rep, k)
        if k2 != k:
            return k2
    return k

ok_all = True
for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RWKV7-*.json"))):
    d = json.load(open(path)); c = d["config"]
    extra = sorted(set(c) - REVIEWER_CFG_FIELDS - GENERIC)
    missing_cfg = sorted(REVIEWER_CFG_FIELDS - set(c) - {"use_cache", "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id"})
    ck = {rename(k): v["shape"] for k, v in d["tensors"].items()}
    ex = expected(c)
    miss = sorted(set(ex) - set(ck)); unexp = sorted(set(ck) - set(ex))
    bad = sorted(k for k in set(ex) & set(ck) if ex[k] != ck[k])
    dtypes = sorted({v["dtype"] for v in d["tensors"].values()})
    status = "OK" if not (miss or unexp or bad) else "MISMATCH"
    ok_all &= status == "OK"
    print(f"== {d['repo']} ({d['sha'][:12]}) tensors={len(ck)} expected={len(ex)} dtypes={dtypes} -> {status}")
    print(f"   cfg: C={c['hidden_size']} L={c['num_hidden_layers']} H={c['num_heads']} N={c['head_dim']} I={c['intermediate_size']} "
          f"ranks(w,a,v,g)=({c['decay_low_rank_dim']},{c['a_low_rank_dim']},{c['v_low_rank_dim']},{c['gate_low_rank_dim']}) eps={c['norm_eps']} tie={c.get('tie_word_embeddings')}")
    print(f"   extra config keys vs reviewer Rwkv7Config: { {k: c[k] for k in extra} }")
    if missing_cfg: print("   reviewer fields absent from config.json:", missing_cfg)
    for name, lst in (("missing", miss), ("unexpected", unexp), ("shape-mismatch", bad)):
        if lst:
            print(f"   {name} ({len(lst)}):", [(k, ex.get(k), ck.get(k)) for k in lst[:8]])
print("ALL_OK" if ok_all else "SOME_MISMATCH")
