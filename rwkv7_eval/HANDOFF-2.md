# RWKV-7 PR #47780 重构 + 实测 —— 第二次交接记录（2026-10-20，Mac 上的 Claude）

本文给下一个会话用：看完就能接着干，不需要翻聊天记录。权威结论在 `REPORT.md`，这里补充"环境、操作方法、未完成事项"。

## 1. 当前状态（一句话）
重构和实测**已全部完成并推送**。工作分支：`Hakureirm/transformers-rwkv:rwkv7-arthur-rework` @ `ecf4817549`。
PR 分支 `add-rwkv7-upstream` 和 PR #47780 的 head 仍是 `317ae0169`，**没有改动**。没有在 PR 下评论，也没有开新 PR。剩下的只有"需要用户拍板"的事（见 §6）。

## 2. 位置
| 路径 | 内容 |
|---|---|
| `~/workspace/rwkv7-tf-rework` | 完整克隆，当前分支 `rwkv7-arthur-rework`。remote：`origin`=fork，`upstream`=huggingface/transformers |
| `~/workspace/rwkv7-tf-rework/REPORT.md` | **最终报告**（中文，§0 是结论速览，§8 是待决事项） |
| `~/workspace/rwkv7-tf-rework/TEST_PROTOCOL.md` | 测试协议，在首次 GPU 运行之前提交，变更记录在文末 |
| `~/workspace/rwkv7-tf-rework/rwkv7_eval/` | 证据：`logs/`（原始日志）、`harness/`（Modal 脚本、`verify_report.py`）、`checkpoint_audit/`、`pr_meta/`、`local_checks/`、`FINDINGS_raw_notes.md` |
| `~/workspace/rwkv7-work/` | 工作目录：`.venv`（py3.12：torch CPU、modal、ruff、libcst）、`modal_ci.py`、`modal_gpu.py`、`summarize.py`、`verify_report.py`、`logs/`（与 `rwkv7_eval/logs` 同步）、`FINDINGS.md`、`scratch/`（含一个 33MB 的锚点 `.pt`，未入库） |
| `~/workspace/rwkv7-handoff/` | 第一次交接（VM212 写的 HANDOFF.txt + 评审原文 + hf_meta） |
| Modal volume `rwkv7-eval`（profile hakureirm） | `/ckpt/<repo>/<sha>/`：6 个 checkpoint（约 56GB）+ prompts.json/verify.json；`/anchor/<repo>/<sha>/<ARM>-<dtype>/<prompt>.pt`：18 组锚点 logits（约 20GB）。**尚未删除，删不删待用户决定** |

## 3. 环境要点（踩过的坑）
- **网络**：直连不通，所有外网访问都要走代理：
  `export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897`
  （不要用 socks5 的 `all_proxy`，pip 和 sdkmanager 不支持。）
- **Modal**：每条命令都要带 `MODAL_PROFILE=hakureirm`，只能用这个 profile，**禁止用 kunin-tech**。先 `source ~/workspace/rwkv7-work/.venv/bin/activate`。
  - 用户把 usage limit 设成了 $29.9。限额太低时任务会报 `workspace ... disabled` / `exceeded its spend limit`。
  - 请求 A100-40GB 时，Modal 有时会给 A100-SXM4-80GB，每条运行都记录了实际型号。
  - 记账：`modal billing summary`、`modal billing report --for today`。
- **gh**：Mac 上是登录状态（Hakureirm，keyring），可以直接推送。之前交接说 token 失效，是不对的。
- **本机资源紧张**（磁盘约 83GB 可用）：重活一律放 Modal；本地只做 git、编辑、`ruff`、modular 转换这类轻量命令。
- 运行仓库工具时要设 `PYTHONPATH=src`，例如：
  `python utils/modular_model_converter.py --num_workers 1 src/transformers/models/rwkv7/modular_rwkv7.py`
- `modal_ci.py` 会**按空格切分参数**，所以 `-k "a or b"` 不能用；改成不含空格的 `-k xxx`，或直接写测试节点 id。慢测试加 `--slow`。

## 4. 怎么复现 / 继续
```bash
cd ~/workspace/rwkv7-work && source .venv/bin/activate && export <代理> MODAL_PROFILE=hakureirm
# CPU 测试（挂载整个工作树，不含 .git）
modal run modal_ci.py --label X --pytest-args "tests/models/rwkv7 -q -p no:randomly -rfE"
modal run modal_ci.py --label X --slow --pytest-args "tests/models/rwkv7/test_modeling_rwkv7.py::Rwkv7IntegrationTests -q"
# 仓库检查（add_dates、modular、docstrings 需要 git，在本地跑）
modal run modal_ci.py --label X --python-args "utils/checkers.py <逗号分隔、不含空格的列表> --keep-going --no-cache"
# GPU（见 modal_gpu.py 顶部说明）
modal run modal_gpu.py::prepare --repo RWKV7-G1k-1.5B-20260930
modal run modal_gpu.py::measure --repo <repo> --arm {OFF|RW-torch|RW-fla|PR} --dtype {float32|bfloat16} [--save] [--batch-check] [--packed-check] [--variant safe_gate64]
modal run modal_gpu.py::compare_anchors --repo <repo>      # 离线比较两组已保存的锚点
python3 summarize.py      # 生成 logs/gpu/SUMMARY.md
python3 verify_report.py  # 用原始日志核对 REPORT.md 里的数字（当前 64/64）
```
- 注意：PR 组（arm `PR`）需要 `~/workspace/rwkv7-pr-317ae01` 这个 worktree，**已经删掉了**。要重跑的话：
  `git -C ~/workspace/rwkv7-tf-rework worktree add --detach ~/workspace/rwkv7-pr-317ae01 317ae0169f20b7f1333481d78a10b4ccd120f4b8`
- RW 组必须**显式**加载 `transformers.Rwkv7ForCausalLM`（脚本已经这样做，并做了校验）；否则在 fla 镜像里如果没有 §5.1 的修复，会加载到 fla 的随机初始化模型。

## 5. 做了什么（提交顺序，`upstream/main`=`823a6ea6d8` 之后，前 13 个是 PR 原有提交）
| sha | 内容 |
|---|---|
| d253af8e25 | config 精简（第 1 条） |
| 65e8362785 | 照 suggestion 原样重写 modular（第 2–11 条），modeling 由工具生成 |
| 028156eb84 | `conversion_mapping` 中的 rwkv7 改名规则（review body 为空，这里补写；6 个 checkpoint 全部对齐） |
| d3749a0025 | 修复转换脚本（标准命名后它会失效） |
| 2ffd5fbd93 | 重写测试 |
| 7c52f973ef | 改写模型卡 |
| 6d0f5c56eb / 2bb70578e0 | rebase 引入的 `__init__` 格式变化和日期刷新 |
| 481d29366e / 3c1e45cfa3 / d95edf1541 | 测试协议及其变更记录（仅工作分支） |
| **9dd223c4f4** | **库级修复**：`hub_kernels._import_kernel_package`，防止 fla 导入时劫持原生的 `rwkv7/mamba/mamba2/bitnet`（main 上本来就有的 bug，qwen3_next 等 8 个模型受影响）。有测试，负对照通过 |
| e85329ed7f | packed 序列：`cu_seq_lens_q` 传给 kernel，token shift 在序列边界清零（参照 qwen3_next） |
| 4f6a611b5c | 跳过 2 个"从 cache 继续 generate"的公共测试并在文档中说明（库限制：纯线性层的 `DynamicCache.get_seq_length()` 会报错） |
| c19d1315b1 | 模型卡补充 packed 说明 |
| ecf4817549 | REPORT.md + rwkv7_eval/ 证据（仅工作分支） |

## 6. 待用户决定（REPORT §8）
1. §5 中的三处改动（Auto 劫持修复、packed、cache-generate 限制）要不要带进 PR、怎么向 Arthur 说明。建议 Auto 劫持修复单独开一个 PR。
2. 只有 torch 回退时长 prompt 慢 10 倍以上（2048 token：1.5B 7.7 s，原 PR 的 chunked torch 是 0.66 s），要不要请 Arthur 保留一个 chunked 的 torch 回退。
3. fla 在 fp32 下不是全精度，且单 token decode 也走 chunk kernel（fla 自己的层在 seq_len<64 时用 fused_recurrent），可以向 Arthur 提一下。
4. 要不要删除 Modal volume `rwkv7-eval`（约 75GB）。
5. 更新上游 PR 时要去掉 `REPORT.md`、`TEST_PROTOCOL.md`、`rwkv7_eval/`。
6. `Rwkv7ForCausalLM.forward` 的 docstring 示例指向一个不存在的仓库（Llama 模板生成的），先照 Arthur 原样保留。
**红线不变**：不动 `add-rwkv7-upstream`，不在 PR 下评论，不开新 PR，GitHub 写操作只用 gh/git。不要在 PR 公开文字里提 RWKV 作者与 FLA 的立场问题。

## 7. 关键数字（备查，权威来源是 REPORT 和 SUMMARY）
- fp32 与官方运行时对照：RW-torch 在 5 个 checkpoint 上都是 26/26，相对偏差 ≤1.3e-5；PR 26/26；RW-fla 有 4 条近似平局导致的分歧（2.9B 1、7.2B 2、G1j 1），相对偏差 ≤4e-3。
- bf16：约一半续写出现分歧，都可以用精度解释（6 例判为"不可解释"的都是平局，卡在判据的严格不等号上）。7.2B 的大偏差在官方运行时自己的 bf16 里同样存在（17.2 / 0.165）。
- CPU 最终测试：281 通过 / 0 失败；慢测试集成 4/4，torch.export 6/6，ONNX 6/6；check-repo 只剩 `cli/serving/response.py:316` 一处 types 报错（与本分支无关）。
- 费用：$10.51（GPU 10.42 + CPU 0.09），全部由赠送额度抵扣，实际计费 $0。

## 8. 与本任务无关、但交接里提到的事
- 第一次交接里"删除 `~/workspace/rwkv7-tf-rework`"和"推送 clockpay 的 `1621fc1`"指的是 **VM212**，不是这台 Mac。Mac 上的 `rwkv7-tf-rework` 是现在的工作仓库，**不要删**。
- clockpay 安卓（`/Users/hakureirm/codespace/clockpay-flutter`，暖色 UI，已修好 shared_preferences 导致的崩溃，一加真机运行正常）：飞书凭证还没写进一加手机；之前停在"等用户决定手填，还是由我从 keychain 解密后写入"。
