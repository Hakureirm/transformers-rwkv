"""Tiny random RWKV-7: official-name checkpoint -> from_pretrained -> save_pretrained -> reload."""
import json, os, re, tempfile, torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, Rwkv7Config, Rwkv7ForCausalLM, Rwkv7Model

torch.manual_seed(0)
cfg = Rwkv7Config(vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=2, num_heads=2, head_dim=32,
                  decay_low_rank_dim=8, a_low_rank_dim=8, v_low_rank_dim=8, gate_low_rank_dim=16)
ref = Rwkv7ForCausalLM(cfg)
with torch.no_grad():
    for p in ref.parameters():
        p.normal_(0, 0.1)
TO_OFFICIAL = [(r"^model\.embed_tokens\.", "rwkv7.emb."), (r"^model\.layers\.", "rwkv7.blocks."),
               (r"^model\.norm\.", "rwkv7.ln_out."), (r"^lm_head\.", "head.")]
def official(k):
    for a, b in TO_OFFICIAL:
        if re.match(a, k):
            return re.sub(a, b, k)
    return k
sd = {official(k): v.contiguous() for k, v in ref.state_dict().items()}
d = tempfile.mkdtemp()
save_file(sd, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
c = cfg.to_dict(); c.update(architectures=["Rwkv7ForCausalLM"], norm_bias=True, wkv_state_dtype="float32", wkv_implementation="chunked")
json.dump(c, open(os.path.join(d, "config.json"), "w"))
print("official keys sample:", sorted(sd)[:3], "n=", len(sd))

def same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb), (set(sa) ^ set(sb))
    return all(torch.equal(sa[k], sb[k]) for k in sa)

m, info = Rwkv7ForCausalLM.from_pretrained(d, output_loading_info=True)
print("load official ->", {k: len(v) for k, v in info.items() if hasattr(v, '__len__')}, "| identical:", same(ref, m))
m2 = AutoModelForCausalLM.from_pretrained(d)
print("AutoModelForCausalLM class:", type(m2).__name__, "identical:", same(ref, m2))
base, binfo = Rwkv7Model.from_pretrained(d, output_loading_info=True)
print("base Rwkv7Model ->", {k: len(v) for k, v in binfo.items() if hasattr(v, '__len__')},
      "| identical:", all(torch.equal(base.state_dict()[k], ref.model.state_dict()[k]) for k in base.state_dict()))
d2 = tempfile.mkdtemp(); m.save_pretrained(d2)
saved = load_file(os.path.join(d2, "model.safetensors"))
print("save_pretrained keys == official keys:", set(saved) == set(sd), "sample:", sorted(saved)[:3])
m3, info3 = Rwkv7ForCausalLM.from_pretrained(d2, output_loading_info=True)
print("reload saved ->", {k: len(v) for k, v in info3.items() if hasattr(v, '__len__')}, "| identical:", same(ref, m3))
x = torch.randint(0, 128, (2, 9))
with torch.no_grad():
    print("logits equal ref vs reloaded:", torch.equal(ref(x).logits, m3(x).logits))
