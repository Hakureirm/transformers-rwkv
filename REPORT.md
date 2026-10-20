# RWKV-7 PR #47780：按维护者评审重构 + 实测 —— 报告

- 分支：`Hakureirm/transformers-rwkv:rwkv7-arthur-rework`（工作分支）。PR 分支 `add-rwkv7-upstream` 未改动，仍是 `317ae0169`。没有在 PR 下发评论，也没有开新 PR。
- 基线：`upstream/main` = `823a6ea6d8`（2026-10-20）。rebase 时 0 冲突。rebase 后、重构前的中间状态是 `2c594b6e6e`。
- 测试协议：[`TEST_PROTOCOL.md`](TEST_PROTOCOL.md)，在第一次 GPU 测量之前提交（`481d29366e`），之后的改动都记在其变更记录里。
- 证据：`rwkv7_eval/`（原始日志、脚本、checkpoint 审计）。本文只引用这些日志，不改写。
- `REPORT.md`、`TEST_PROTOCOL.md`、`rwkv7_eval/` 只放在工作分支上，**不要带进上游 PR**。

## 0. 结论速览

1. Arthur 的 12 条意见全部落实，采用的是他写好的 suggestion 版本（modular 从 1196 行减到 272 行；modeling 由工具重新生成，从 1174 行减到 290 行）。评审正文（review body）是空的，其中提到的改名映射是这次补写的，已在 6 个官方 checkpoint 上验证可以无损加载。
2. **数值正确（fp32）**：在 G1k-1.5B / 2.9B / 7.2B、G1j-1.5B、1.5B-20260805 上，以官方运行时为锚点，每个 checkpoint 26 条 prompt，每条贪心续写 64 个 token：
   - 重构版（torch 路径）**全部 5 个 checkpoint 都是 26/26 完全一致**，logits 最大相对偏差 ≤ 1.3e-5。
   - PR 原版 26/26 一致。
   - 重构版装 fla 时**有 4 条分歧**（2.9B 1 条、7.2B 2 条、G1j 1 条），没有达到预声明的 fp32 通过判据。每一条都分歧在 margin 小于 0.001 logit 的近似平局处，可以用 fla kernel 在 fp32 下的精度损失解释（相对偏差可达 4e-3）。
3. **Arthur 的"贪心生成与 PR 逐 token 一致"**：fp32 下成立（26/26）。bf16 下不成立（13/26），所有分歧都是 bf16 精度造成的近似平局，详见 §4.4。
4. **发现并处理的问题**（详见 §2）：
   - 🔴 装了 fla 时，`AutoModelForCausalLM` 会静默加载一个随机权重的模型。这是 main 上早已存在的库级 bug，`mamba` / `mamba2` / `bitnet` 也受影响。已在源头修复，补了测试并做了负对照。
   - packed 序列的边界没有接到 kernel 和 token shift 上，按 qwen3_next 的做法补上了。
   - 传入已有 cache 后无法继续 `generate`。这是库层面的限制，没有修，只是跳过相应测试并在文档里说明。相对原 PR 是一项功能退化。
5. **费用**：Modal profile `hakureirm`，算力共 $10.51（GPU $10.42 + CPU $0.09），全部由赠送额度抵扣，实际计费 $0.00。

## 1. Arthur 12 条的落实情况

| # | 评审意见 | 落实 | 提交 |
|---|---|---|---|
| 1 | config 只保留实际用到的字段；`layer_types` + `number_of_conv_states` | 照原样采用。6 个官方 config 都能解析，多出来的 `norm_bias / wkv_state_dtype / wkv_implementation` 作为普通属性保留，不报错 | `d253af8e25` |
| 2 | WKV 改用 `fla.ops.rwkv7.chunk_rwkv7` + 逐 token 的 torch 回退 | 照原样采用。额外接上了 `cu_seqlens`（§2.2） | `65e8362785`, `e85329ed7f` |
| 3 | token shift 放进 conv 槽 | 照原样采用 | `65e8362785` |
| 4 | 去掉 layer_id 分支和 lora_gates；用 `has_previous_state`/`update_recurrent_state` | 照原样采用 | `65e8362785` |
| 5 | 删除 deep_embed 和 `_project`；用 `ACT2FN["relu2"]` | 照原样采用 | `65e8362785` |
| 6 | 用 `DynamicCache(config)` | 照原样采用 | `65e8362785` |
| 7 | 标准 DecoderLayer、`ln0` 用 `nn.Identity`、`apply_mask_to_padding_states` | 照原样采用 | `65e8362785` |
| 8 | 标准输出类型 + `past_key_values` | 照原样采用。由此引出 §2.3 的限制 | `65e8362785` |
| 9 | 用 `_can_record_outputs` | 照原样采用 | `65e8362785` |
| 10 | 骨干用标准名 + `conversion_mapping.py` 改名（"见 review body"）| review body 是空的，改名规则是补写的，共 4 条，保存时会反向改回 | `028156eb84` |
| 11 | `Rwkv7ForCausalLM` 直接继承 `LlamaForCausalLM` | 照原样采用。由此生成的 forward docstring 示例里写的是一个不存在的仓库 `meta-rwkv7/Rwkv7-2-7b-hf`，这是 `X(LlamaForCausalLM): pass` 写法的通病，没有改 | `65e8362785` |
| 12 | 复用 main 的现成组件；rebase 到最新 main | rebase 0 冲突；rebase 后的旧代码基线测试 131 通过 / 0 失败 | — |

suggestion 之外，在 Arthur 修改范围内我只改了一处：模块顶部 docstring 的 NAMING 段原来写着"不需要改名表"，与标准命名矛盾，已改写。

**评审之外的配套修改**（不涉及偏离）：
- 转换脚本 `convert_rwkv7_checkpoint_to_hf.py`：它拿模型的键名做校验，标准命名后会全部报不匹配，相当于已经失效。现在先把键名映射回官方格式再校验，输出格式不变（`d3749a0025`）。
- 因为 rebase，按 main 的新格式重新生成了 `__init__.py`（`6d0f5c56eb`），刷新了模型卡日期（`2bb70578e0`）。
- 测试按新 API 重写（`2ffd5fbd93`），模型卡按新架构改写（`7c52f973ef`）。

## 2. 偏离 / 追加修改，以及原因

### 2.1 🔴 装了 fla 时 Auto 类被劫持（库级修复，`9dd223c4f4`）
- **现象**：在装了 fla 的镜像里，`AutoModelForCausalLM.from_pretrained("RWKV/RWKV7-G1k-1.5B-…")` 返回的是 `fla.models.rwkv7.RWKV7ForCausalLM`，所有权重都是 MISSING（随机初始化），logits 全是 NaN。第一次 RW-fla 实测就是这样，那次结果已作废，日志保留：`rwkv7_eval/logs/modal-g1k15-RWfla-fp32.out`。
- **根因**：kernel 加载器 `use_kernel_func_from_hub_with_fallback` 会 `import fla`，而 fla 的包初始化会用 `exist_ok=True` 把自己的类注册到 Auto 映射里，把原生的 **`bitnet`、`mamba`、`mamba2`、`rwkv7`** 都改到 fla 的类上（`rwkv7_eval/logs/gpu/fla-collisions-*.json`）。
- **这是 main 上已有的 bug**：在 main 原有代码里只导入 qwen3_next，`mamba` 就会被劫持（`qwen3next-mamba-hijack-*.json`）。main 上有 8 个模型通过这个加载器导入 fla。
- **修复**：`hub_kernels._import_kernel_package` 导入可选 kernel 包后，只撤销那些与库内原生类型重名的注册；fla 自有的类型保持不动。修复后两个复现都解析到原生类。新测试在有修复时通过、撤掉修复时失败（负对照：`logs/kernelfix-*-v2-*.log`）。
- **建议**：这个修复可以单独拆成一个 PR。

### 2.2 packed 序列（`e85329ed7f`，按 Arthur 本意补全）
Arthur 说 packed 序列由 fla 处理，但他的代码里解码层丢掉了 kwargs，也没有把 `cu_seqlens` 传给 kernel。结果是装了 fla 时，packed batch 会被当成一条长序列，状态和 token shift 都跨过了序列边界，而且不报错。现在参照 qwen3_next 补上：kwargs 往下传，`cu_seq_lens_q` 交给 kernel，token shift 在每条序列的起点清零。没装 fla 时，加载器会明确报错，而不是悄悄把序列混在一起。GPU 验证：4 条序列拼在一起，每一段的逐位置 argmax 都与单独运行一致（最大偏差 ≤ 0.036，在 fla fp32 的精度范围内）；对照组不给边界时，偏差达到 13.99。

### 2.3 传入已有 cache 无法 `generate`（只记录限制，`4f6a611b5c`）
`generate` 在 prefill 时会调用 `DynamicCache.get_seq_length()`，而只包含线性注意力层的 cache 会对此报错。mamba/mamba2 的输出字段叫 `cache_params`，公共测试因此跳过了它们，所以 rwkv7 是第一个碰到这个问题的。要真正修好需要改 cache 核心，让线性层记录已经处理过的 token 数，这超出了本次范围。目前的处理：两个公共测试标注跳过并写明原因（与 nemotron_h / zamba2 的做法相同）；用 forward 逐步续写的路径由自定义测试覆盖；模型卡里写明了用法。**相对原 PR 是功能退化**：原 PR 的 d1ba899 支持过带状态的 generate。

### 2.4 交接中其他待决问题的结论
- (b) `@strict` 遇到多余的键不会报错，见第 1 条。
- (c) **只有 torch 回退时，长 prompt 会很慢**：2048 个 token 的 prefill，1.5B 要 7.7 s，7.2B 要 13.6 s，13.3B（bf16）要 23.4 s；装 fla 只要 0.1–1.7 s；PR 原版自带的 chunked torch 求解器在 1.5B 上是 0.66 s。Arthur 删掉了 PR 的 chunked torch 回退，要不要就此向他说明，由你决定。
- (d) fla 的 `safe_gate=True, chunk_size=64`：单独测过，精度与默认参数相当（fp32 25/26，其中 1 条分歧可由精度解释），没有必要采用。
- (f) 从 Llama 继承来的 `_tied_weights_keys` 只在 `tie_word_embeddings=True` 时生效（官方 checkpoint 都是 False），`_tp_plan` 只切分 `lm_head`，都不影响正确性。
- (g) config 里没有 `initializer_range` 时，`_init_weights` 会回落到 0.02。官方权重的加载测试和"重新加载后保留全部权重"的测试都通过了。
- (i) 导出：`torch.export` 6/6、ONNX 6/6、OpenVINO 2/2 通过。executorch 没有跑，因为它会强制替换成它自己绑定的 torch 版本，测出来的就不是被测环境了。

## 3. CPU 测试与仓库检查

| 项目 | 结果 | 日志 |
|---|---|---|
| rebase 后、重构前的旧代码 | 131 通过 / 198 跳过 / 0 失败 | `logs/baseline-rebased-2c594b6-*.log` |
| 最终代码：rwkv7 + kernels + qwen3_next | **281 通过 / 396 跳过 / 0 失败** | `logs/final-tests-*.log` |
| 慢测试：集成测试（BlinkDL 0.1B `.pth`，与 `rwkv` 包的值对照） | 4/4 | `logs/slow-integration-*.log` |
| 慢测试：torch.export / ONNX / OpenVINO | 6/6、6/6、2/2 | `logs/slow-export-*.log`, `logs/slow-onnx-export-*.log` |
| numpy 参考实现（独立重新推导：prefill、130 token、逐 token decode） | 通过 | 包含在上面的 rwkv7 测试中 |
| check-repo（等价于 `make check-repo`） | 只有 `types` 一项报错，位置在 `cli/serving/response.py:316`，本分支没有改动过这个文件；需要 git 的检查（modular、add_dates、docstrings）在本地通过；`reviewers` 检查需要从 git 安装额外包，跳过 | `logs/final-check-repo-*.log` |

## 4. GPU 实测（A100，Modal）

### 4.1 环境与留痕
- 每条运行都记录了 GPU 型号、驱动、torch/transformers/fla 版本、代码 sha、实际解析到的 WKV 实现，以及实际加载的模型类（`logs/gpu/measure-*.json`）。汇总表：`logs/gpu/SUMMARY.md`。
- 请求的都是 A100-40GB，但 Modal 有时分配的是 **A100-SXM4-80GB**，其中包括 1.5B 的 fp32 锚点。两者同为 sm80 架构。实际型号按运行逐条记录。
- checkpoint：6 个仓库都锁定了 revision，所有分片的 sha256 都与 Hub LFS 记录和 `release-manifest.json` 一致，张量数分别为 798/1062/1062/2019/798/798（`logs/gpu/prepare-*.json`）。
- prompt：原 PR 用过的 24 条 lambada prompt 在公开记录里找不到，所以使用 `EleutherAI/lambada_openai` 测试集的前 24 条（去掉最后一个词，数据集 revision `900124bf`），这是**重建的版本**，不保证与原来的 24 条相同；另外加上 G1k 模型卡的示例，以及一条 2048 token 的长序列。所有 prompt 只用官方 tokenizer 分词一次，各组输入完全相同的 token id（`logs/gpu/prompts-*.json`）。

### 4.2 fp32，以官方运行时为锚（26 条 prompt × 64 token）
| checkpoint | 对照组 | 续写完全一致 | prefill 最大相对偏差 | 逐 token decode 最大相对偏差 |
|---|---|---|---|---|
| G1k-1.5B | PR 原版 | 26/26 | 1.26e-5 | 5.34e-6 |
| G1k-1.5B | 重构版 torch | 26/26 | 9.73e-6 | 1.58e-6 |
| G1k-1.5B | 重构版 fla | 26/26 | 1.49e-3 | 1.40e-3 |
| G1k-2.9B | 重构版 torch / fla | 26/26 / **25/26** | 1.08e-5 / 2.60e-3 | 1.49e-6 / 3.58e-3 |
| G1k-7.2B | 重构版 torch / fla | 26/26 / **24/26** | 1.27e-5 / 4.13e-3 | 7.76e-6 / 1.51e-3 |
| G1j-1.5B | 重构版 torch / fla | 26/26 / **25/26** | 1.31e-5 / 1.66e-3 | 1.96e-6 / 6.12e-4 |
| 1.5B-20260805 | 重构版 torch / fla | 26/26 / 26/26 | 1.03e-5 / 2.15e-3 | 1.57e-6 / 7.41e-4 |

- 非零偏差在预期之内：官方运行时用的衰减常数是截断值 `-0.606531`，再加上求和顺序的差异。
- fla 的 4 条分歧：分歧位置 k = 29–40，锚点在该位置的 top1−top2 差距是 0.00035–0.00082，都小于被测组那两个 logit 的偏差之和（0.0019–0.0050），属于精度可以解释的情况。fla 自己也会提示 "ChunkDeltaRuleFunction does not support float32 on some platforms"。Arthur 的代码让单 token decode 也走 chunk kernel，所以 decode 同样受影响。

### 4.3 批处理、packed 与 generate
- 4 条一批、左填充（1.5B fp32）：重构版 torch 24/24；重构版 fla 23/24。不一致的是 lambada-16，在第 20 步分歧，单独运行时那一步的 margin 只有 0.0008，小于偏差之和 0.0021，可由精度解释。批量 `generate` 与手写的批量循环 24/24 一致。
- packed：见 §2.2。
- 在所有 RW 运行中，`generate` 与手写的 cache 贪心循环都一致（4/4）。

### 4.4 bf16
- 各组都有约一半的续写在某一步出现分歧。按预声明的判据，大多数分歧可由精度解释（判据：锚点在分歧处的 margin 小于两个 logit 的偏差之和）。
- 按判据被列为"不可解释"的共 6 例，逐一检查过：其中 5 例是被测组在 bf16 舍入后 top1 和 top2 **完全相等**，判据用的是严格小于号，平局正好卡在边界上（OFF@1.5B lambada-21、RW-fla@1.5B lambada-04、RW-fla 对 RW-torch@1.5B lambada-00、RW-torch@2.9B lambada-21、RW-fla 对 RW-torch@13.3B lambada-02）；另外 1 例是锚点自身在第二名上就是两个 token 并列（PR 对 OFF bf16@1.5B lambada-12）。判据没有因此修改。
- 7.2B 上 bf16 的 prefill 偏差较大（重构版 torch 为 16.6 / 相对 0.156），但**官方运行时自身的 bf16 与 fp32 之间偏差也有 17.2 / 相对 0.165**（离线比对锚点得到，在 1.5B 上与 GPU 计算的结果交叉验证过，完全一致），所以这是 bf16 本身的效应，不是实现问题。
- 13.3B 的 fp32 权重需要约 53 GB，放不进 40GB 显存，按协议只测了 bf16。

### 4.5 作废与补测的运行（都保留了日志）
- 第一次 RW-fla fp32 运行：测到的是 fla 的随机初始化模型，作废（见 §2.1）。
- 改为显式加载模型类之后重跑；packed 修改之后又重跑了一次，两次运行的续写和偏差**完全相同**，说明 packed 修改对普通输入没有影响。
- 2.9B/7.2B 的官方 bf16 锚点和 fp32 锚点是同时跑的，bf16 那次没能对照 fp32，之后改用离线比对补上。

## 5. 费用
| 时点 | 计量 | 赠送额度抵扣 | 实际计费 | 日志 |
|---|---|---|---|---|
| 开始前 | 1.90（仅存储，由免费额度抵扣） | 0 | $0.00 | `billing-00` |
| 所有 GPU 运行之后 | 13.01 | 11.09 | $0.00 | `billing-05` |
| 最终（按 app 汇总：GPU $10.42，CPU $0.09） | 12.42 | 10.50 | $0.00 | `billing-06` |

两次快照的计量数不一样（13.01 和 12.42），应该是 Modal 账单汇总口径或时点不同，两份原始快照都保留了。全部容器在运行结束后自动停止，没有空转。Volume `rwkv7-eval` 里存有 checkpoint（约 56 GB）和 18 组锚点 logits（每组约 1.1 GB），合计约 75 GB（估算），确认不再需要后可以删除。

## 6. git 提交与推送
`upstream/main` 之后的前 13 个提交是 PR 原有的提交（rebase 后），之后是本次的提交：
```
d253af8e25 Trim Rwkv7Config to the fields the official checkpoints use
65e8362785 Rewrite modular_rwkv7 along the review and regenerate modeling
028156eb84 Rename the RWKV-7 reference backbone names at load time
d3749a0025 Keep the conversion script working with the standard backbone names
2ffd5fbd93 Port the RWKV-7 tests to the reviewed architecture
7c52f973ef Update the RWKV-7 model card for the reviewed architecture
6d0f5c56eb Regenerate the rwkv7 __init__ in main's auto-generated format
2bb70578e0 Refresh the model-card contribution date
481d29366e Add the GPU test protocol, written before any measurement      (work branch only)
3c1e45cfa3 Protocol change log: load the RW arms' class explicitly        (work branch only)
9dd223c4f4 Keep native model types when a kernel package registers its own on import   (library; could be its own PR)
e85329ed7f Pass packed-sequence boundaries to the WKV kernel and the token shift
4f6a611b5c Skip the generate-from-cache common tests and document the limit
c19d1315b1 Document packed-sequence support in the RWKV-7 model card
d95edf1541 Protocol change log: record the remaining deviations found in review   (work branch only)
+ 报告与证据提交（work branch only）
```
推送状态：见本次的最后一个提交。只推送到 fork 的 `rwkv7-arthur-rework`。

## 7. 独立 review 记录（交付前）
- **数字核对**：`rwkv7_eval/harness/verify_report.py` 直接从原始日志重新计算本报告引用的数字，包括 fp32 表、分歧计数与 margin、bf16 不可解释例、7.2B 基线、性能数字、checkpoint 校验、批处理/packed 结果、提交 sha、日志路径和账单，结果见 `rwkv7_eval/logs/verify_report.txt`。
- **review 中改正的问题**：PR 原版 prefill 时间按日志改为 0.66 s；volume 占用从"约 60 GB"改为约 75 GB（估算）；修正一处乱码笔误；模型卡补充 packed 序列说明（`c19d1315b1`）；协议变更记录补登了此前遗漏的偏差，包括 80GB 卡替代、离线 bf16 基线、批处理检查重写与重跑、新增检查（`d95edf1541`）。
- **代码复核**：逐行重读 modular、hub_kernels 和 conversion_mapping 的完整 diff，没有发现正确性问题。复核时确认的边界情况：梯度检查点下 cache 以关键字传入，能被正确解除；packed 输入在 `use_cache` 默认开启时会额外建一个不用的 cache，但不影响输出；带已有 cache 再传 packed 输入不受支持，fla 会因状态的批大小不匹配而报错；如果用户在导入 transformers 模型之前自己先 `import fla`，fla 的注册会保留，这属于用户的显式选择，修复不干预。
- review 之后只改了文档和协议，modeling 代码没有再变动，所以 §3 的最终测试仍然有效（测试对应的代码 sha 为 `4f6a611b5c`）。

## 8. 遗留问题与需要你决定的事
1. **是否把 §2 的三处改动带进 PR**，以及怎么向 Arthur 说明：Auto 类劫持的修复（建议单独开 PR）、packed 序列补全、带 cache 的 generate 退化。
2. **只有 torch 回退时长 prompt 慢 10 倍以上**（§2.4 c），要不要请 Arthur 保留一个 chunked 的 torch 回退（qwen3_next 也保留了）。
3. **fla 路径在 fp32 下不是全精度**，而且单 token decode 也会走 chunk kernel。fla 自己的 RWKV-7 层在 seq_len 小于 64 且不在训练时，会改用 `fused_mul_recurrent`。这个点可以告诉 Arthur，供他判断。
4. 上游更新 PR 时，要去掉 `REPORT.md`、`TEST_PROTOCOL.md`、`rwkv7_eval/`。
5. `Rwkv7ForCausalLM.forward` 的示例 docstring 指向一个不存在的仓库（由 Llama 模板生成）。
6. 原 PR 的 24 条 lambada prompt 无法恢复，本次使用的是重建版本。
