import json, os, re, subprocess, sys, tempfile, torch
from transformers import Rwkv7Config, Rwkv7ForCausalLM
torch.manual_seed(1)
cfg = Rwkv7Config(vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=2, num_heads=2, head_dim=32,
                  decay_low_rank_dim=8, a_low_rank_dim=8, v_low_rank_dim=8, gate_low_rank_dim=16)
ref = Rwkv7ForCausalLM(cfg)
with torch.no_grad():
    for p in ref.parameters(): p.normal_(0, 0.1)
NATIVE = [(r"^model\.embed_tokens\.", "emb."), (r"^model\.layers\.", "blocks."), (r"^model\.norm\.", "ln_out."), (r"^lm_head\.", "head.")]
def native(k):
    for a, b in NATIVE:
        if re.match(a, k): return re.sub(a, b, k)
    return k
d = tempfile.mkdtemp(); pth = os.path.join(d, "native.pth")
torch.save({native(k): v for k, v in ref.state_dict().items()}, pth)
cfgp = os.path.join(d, "cfg.json"); json.dump({**cfg.to_dict(), "norm_bias": True}, open(cfgp, "w", encoding="utf-8"))
script = "src/transformers/models/rwkv7/convert_rwkv7_checkpoint_to_hf.py"
x = torch.randint(0, 128, (2, 7))
for label, extra in (("inferred config", []), ("explicit config", ["--config", cfgp])):
    out = os.path.join(d, label.replace(" ", "_"))
    r = subprocess.run([sys.executable, script, "--checkpoint", pth, "--output_dir", out, *extra], capture_output=True, text=True)
    print(label, "-> exit", r.returncode, r.stdout.strip()[-80:], r.stderr.strip()[-300:] if r.returncode else "")
    m, info = Rwkv7ForCausalLM.from_pretrained(out, output_loading_info=True)
    with torch.no_grad():
        print("   load:", {k: len(v) for k, v in info.items() if hasattr(v, '__len__')}, "logits equal:", torch.equal(ref(x).logits, m(x).logits))
# negative control: wrong config width must be refused
bad = os.path.join(d, "bad.json"); json.dump({**cfg.to_dict(), "intermediate_size": 128}, open(bad, "w", encoding="utf-8"))
r = subprocess.run([sys.executable, script, "--checkpoint", pth, "--output_dir", os.path.join(d, "bad"), "--config", bad], capture_output=True, text=True)
print("wrong-width config -> exit", r.returncode, "| refused:", "do not match the shapes" in r.stderr)
