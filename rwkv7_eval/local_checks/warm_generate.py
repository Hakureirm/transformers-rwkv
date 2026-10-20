import torch, traceback
from transformers import Rwkv7Config, Rwkv7ForCausalLM, DynamicCache
torch.manual_seed(0)
cfg = Rwkv7Config(vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=2, num_heads=2, head_dim=32,
                  decay_low_rank_dim=8, a_low_rank_dim=8, v_low_rank_dim=8, gate_low_rank_dim=16)
m = Rwkv7ForCausalLM(cfg).eval()
with torch.no_grad():
    for p in m.parameters(): p.normal_(0, 0.5)
P = torch.tensor([[5, 17, 42, 8, 55, 9, 31]])
gen = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)
with torch.no_grad():
    fresh = m.generate(P, **gen)[0, P.shape[1]:].tolist()
    print("fresh            :", fresh)
    # reference: manual greedy with the cache
    out = m(P, use_cache=True); cache = out.past_key_values; ids = [int(out.logits[0, -1].argmax())]
    for _ in range(5):
        out = m(torch.tensor([[ids[-1]]]), past_key_values=cache, use_cache=True); ids.append(int(out.logits[0, -1].argmax()))
    print("manual cache loop:", ids, "== fresh:", ids == fresh)
    print("cache type:", type(out.past_key_values).__name__)
    for label, kwargs in [
        ("warm cache + FULL ids          ", lambda c: dict(input_ids=P)),
        ("warm cache + NEW ids only      ", lambda c: dict(input_ids=P[:, 4:])),
        ("warm cache + NEW ids + full mask", lambda c: dict(input_ids=P[:, 4:], attention_mask=torch.ones_like(P))),
    ]:
        warm = m(P[:, :4], use_cache=True).past_key_values
        try:
            g = m.generate(past_key_values=warm, **kwargs(warm), **gen)
            print(label, "->", g[0, -6:].tolist(), "== fresh:", g[0, -6:].tolist() == fresh)
        except Exception as e:
            print(label, "-> EXC", type(e).__name__, str(e)[:160])
