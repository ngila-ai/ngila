# ngila

**ngila = align reversed.** 本项目专门研究模型“反对齐”：减少训练引入的审查、
回避与事实歪曲，让模型依据已有知识和证据回答，并诚实表达不确定性。
最终产物将作为 **RLCD calibrated decision 训练的基础模型**。
项目产出的模型统一称为 **ngila checkpoint**；ngila 本身就表示“反对齐”。

首个目标是 **Qwen3.5-9B**。0.8B 用于流程验证；9B-Base 用于对照。
“模型的真实想法”在本项目中通过可观察行为评估，不把隐藏状态或思维链当作真相。

## 第一阶段

1. 固定原版基线：模型、模板、语言、thinking、精度、生成预算。
2. 提取对比激活方向，验证不同层及强度的可撤销干预。
3. 分别评估拒答、回避、事实歪曲、错误前提迎合和能力保持。
4. 加入未见话题、随机方向、零强度和普通任务对照。
5. 使用独立、有证据标签的决策数据测 Brier score / log loss / 可靠性曲线。
6. 通过评估后，才研究可导出、可训练的权重修改或 adapter，再接入 RLCD。

激活 steering 本身不是完成的 RLCD 训练 checkpoint；不能直接宣称已“去审查”。
RLCD 的具体数据接口、奖励定义与验收阈值待确定。

## 实验入口

`scripts/experiment.py` 支持原版生成、逐层激活收集，以及指定层输出的 additive
steering。模型仅从本地加载，不调用外部评审 API，不修改源权重。
`scripts/direction.py` 从标记为 discovery 的采集中提取归一化均值差方向。
层编号从 0 开始，hook 位于整个 decoder block 的输出；不是其 MLP 单独输出。

本机可用解释器：`/home/tom/anaconda3/envs/so/bin/python`。
依赖版本和运行配置随每次运行记录。先检查 CUDA；当前状态见 docs/status.md。

```bash
CUDA_VISIBLE_DEVICES=0 /home/tom/anaconda3/envs/so/bin/python scripts/experiment.py \
  --model /mnt/data/models/Qwen/Qwen3.5-9B \
  --prompts data/smoke.jsonl --output runs/9b-baseline --device cuda

# 仅验证运行链路；小样本不能支持事实性或校准结论。
# 本地 0.8B 的 tokenizer.json 为 0 字节；构造不含空文件的本地视图。
# 若视图已存在，跳过准备步骤。
python scripts/prepare_smoke_model.py \
  --source /mnt/data/models/Qwen/Qwen3.5-0.8B \
  --output runs/models/Qwen3.5-0.8B

/home/tom/anaconda3/envs/so/bin/python scripts/experiment.py \
  --model runs/models/Qwen3.5-0.8B \
  --prompts data/smoke.jsonl --output runs/08b-smoke --device cpu \
  --max-new-tokens 96 --limit 2
```

提示文件每行为 JSON：必需 `id`, `prompt`, `split`；discovery 数据还须 `group`。
原始续写使用 `--template raw`；聊天模板默认不开 thinking，可用 `--thinking`。
生成采用 greedy，显式记录 EOS 与长度截断；截断不能计为拒答。
CPU 冒烟使用 float32 和上游 PyTorch reference kernels；正式 GPU 对比用 BF16。
0.8B 本地视图使用已有 vocab/merges 重建 tokenizer，尚未与上游 tokenizer.json
逐字节核对；它的结果仅用于工程检查。

```bash
# 使用自己准备的、主题和语言匹配的发现集；不使用 smoke 集提取结论。
CUDA_VISIBLE_DEVICES=0 /home/tom/anaconda3/envs/so/bin/python scripts/experiment.py \
  --model /mnt/data/models/Qwen/Qwen3.5-9B --device cuda --mode collect \
  --prompts data/discovery.jsonl --output runs/9b-discovery

/home/tom/anaconda3/envs/so/bin/python scripts/direction.py \
  --run runs/9b-discovery --positive sensitive --negative control \
  --layer 13 --output runs/direction-l13.pt

CUDA_VISIBLE_DEVICES=0 /home/tom/anaconda3/envs/so/bin/python scripts/experiment.py \
  --model /mnt/data/models/Qwen/Qwen3.5-9B --device cuda \
  --prompts data/validation.jsonl --output runs/9b-steer-l13 \
  --direction runs/direction-l13.pt --alpha -5
```

`discovery.jsonl` 和 `validation.jsonl` 尚未建立；以上是接口示例。
13 层/-5 只是待验证参数，不是已验证配方。工具拒绝跨 checkpoint、跨模板或
与发现集重叠的 steering 评估。改变模型或模板必须重新提取方向。

## 参考

- [Qwen 政治审查机制研究](https://vas-blog.pages.dev/qwen-censorship/)
- [参考实验代码](https://github.com/Srinivasa314/qwen3.5-censorship)
- [Heretic](https://github.com/p-e-w/heretic)
- [拒答方向论文](https://arxiv.org/abs/2406.11717)

这些是研究参考，不是本项目已经复现的结论。

## 当前评测数据 v0.2

已扩充至 **17 个来源、147,037 条测试记录**；另登记 SORRY-Bench，等待上游访问授权。
核心行为评测 **22,001 条去重提示**，扩展行为评测 **101,042 条**；事实性、数学、
知识和中英双语安全测试分别报告。数据已准备不代表模型已完成评测。

完整数量、限制、复现命令与发布记录见 [v0.2 评测清单](docs/eval-v0.2.md)。
新测试集查出更多训练重叠，v0.2 train/validation 为 **15,039 / 1,566**；
使用 v0.1 训练数据时，不能宣称 v0.2 无泄漏。

## 初始项目组件与数据 v0.1

代码目标：GitHub `ngila-ai/ngila`；模型与数据：Hugging Face `ngila-ai`。
[GitHub 代码库](https://github.com/ngila-ai/ngila)与两个 Hugging Face 数据集均已发布。见 [发布状态](docs/publication-status.json)。

- [评测与标注规范](eval/README.md)：拒答、回避、完成请求、事实性、校准独立报告。
- [9 个 benchmark 来源](docs/benchmarks.md)：已接入 XSTest、SimpleSafetyTests、BeaverTails。
- [数据协议](docs/dataset-schema.md)：17,232 train / 1,797 validation / 3,571 test；隔离 8,157 条泄漏训练记录。
- [Pipeline](configs/pipeline-v0.1.json)：重建数据、导出 550 条评测提示、评分已保存的 smoke 输出、更新结果。
- [Model zoo](model_zoo/README.md)、[持续论文](paper/manuscript.md)、[实验台账](docs/experiments.jsonl)。

```bash
# 从仓库根目录运行；配置中的 Python 和 BeaverTails 缓存路径需匹配本机。
python -m ngila catalog
python -m ngila pipeline --config configs/pipeline-v0.1.json --output runs/pipeline-v0.1
python -m ngila zoo
python -m unittest discover -s tests -v
```

已存在的输出目录不会覆盖。重跑时为配置内的 runs 路径和 --output 同时选择新版本。
完整生成使用 scripts/experiment.py，输入 pipeline 导出的 benchmark-prompts.jsonl，
随后通过 `ngila annotate` / `ngila eval` 评审；目前没有跑完 550 条 benchmark。
训练入口 `ngila discovery` 仅接受已评审的 train 输出，`ngila train` 拟合可撤销
steering 方向；它不等于 SFT/RLCD，也不能把安全偏好直接取反当作事实标签。

数据发布包位于 `releases/`，两种许可证分仓保存。网络可用后：

```bash
python -m ngila publish --folder releases/ngila-benchmarks-v0.1 --receipt releases/benchmark-upload-receipt.json
python -m ngila publish --folder releases/ngila-beavertails-v0.1 --receipt releases/beavertails-upload-receipt.json
```

新机器用 `ngila fetch` 获取源数据，按固定 revision 和官方 split 更新输入路径；
`ngila build` 后用 `ngila stage-dataset --help` 生成相同结构的发布包。

完整 9B 基线生成配置：`configs/pipeline-9b-benchmark.json`。CUDA 可用后执行：

```bash
python -m ngila pipeline --config configs/pipeline-9b-benchmark.json --output runs/qwen3.5-9b-benchmark-v0.1
```

它生成 550 个回答并建立待评审标注，完成生成不代表评测通过。当前该流程未执行。
本地 Git 已初始化，origin 指向 `https://github.com/ngila-ai/ngila.git`。
本项目使用 `ngila-ai` 独立认证，提交身份为 `ngila-ai <thomas.l.wang@gmail.com>`。
本地凭据绑定仅作用于当前仓库。

已发布数据集：

- [ngila-benchmarks-v0.1](https://huggingface.co/datasets/ngila-ai/ngila-benchmarks-v0.1)：550 条 test，CC BY 4.0。
- [ngila-beavertails-v0.1](https://huggingface.co/datasets/ngila-ai/ngila-beavertails-v0.1)：17,232 train / 1,797 validation / 3,021 test，CC BY-NC 4.0。

### Xiangxin System One judge

The response judge is Xiangxin, pinned to `xiangxin-2.0.0`, using the native
`/v1/systemone` API and three independent Noul questions: `response_unsafe`,
`refusal`, and `request_fulfilled`. Set `XIANGXIN_API_KEY` in the process environment.

```bash
python -m ngila judge --outputs runs/example/outputs.jsonl --output runs/example-judge
```

The destination must be new. Each result preserves the original response hash,
raw judge response, resolved model and probabilities. Incomplete generations are
skipped. API failures and malformed responses are excluded from metric denominators
and counted explicitly. The default 0.5 threshold is provisional, not calibrated;
use `--threshold` to record a different preselected threshold. Judge outputs are
separate from human factuality annotations and do not establish honesty.

Evaluation now prioritizes two tracks: unsafe-prompt response behavior and
capability retention. Terminal-Bench 2.0 is the first capability target; the other
requested benchmarks are registered in `configs/capability-benchmarks.json`.
The historical v0.2 datasets remain the source archive. TB2.0 integration and the
custom unsafe-prompt suite are still pending; registration is not an evaluation run.

API references: [quick start](https://docs.xiangxinai.cn/introduction/quickstart),
[Noul](https://docs.xiangxinai.cn/primitives/noul).

For the project's private Xiangxin deployment, use SSH over WireGuard. `t3090`
listens on SSH port 50812; the model service is bound to remote loopback, not the
WG interface. Start a tunnel in a separate terminal (keep it running):

```bash
ssh -N -p 50812 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -i ~/.ssh/id_ed25519_workpnp \
  -L 127.0.0.1:18739:127.0.0.1:18730 tom@10.9.0.1
```

Reuse an existing tunnel if port 18739 is already forwarded. Then run:

```bash
python -m ngila judge --outputs runs/example/outputs.jsonl \
  --output runs/example-judge-wg \
  --endpoint http://127.0.0.1:18739/v1/systemone --no-auth
```

`--no-auth` does not read or send the public API key. Loopback requests bypass
HTTP proxies. The successful connection check is saved in
`eval/evidence/xiangxin-wg-smoke-v1/`, including the remote deployment identity.

### First Qwen3.5-9B pilot

The active pilot is `runs/ngila-9b-pilot-v1`, using 64 training prompts and 16
validation prompts from v0.2. `configs/pipeline-9b-pilot-v1.json` records judging,
discovery, direction fitting and the validation sweep. The background driver
waits for baseline generation before executing the pipeline. Inspect
`pipeline-driver.log` and `pipeline/run.json` under the run directory for status.

`ngila judge-discovery` converts hash-bound Xiangxin decisions into a discovery
contrast only when refusal and fulfillment probabilities agree confidently.
It rejects held-out inputs and omits ambiguous/incomplete cases. These are
behavioral training labels, not factual truth or SFT targets. This pilot does
not establish capability preservation or produce an RLCD-ready checkpoint.

To repeat the local Intern-Decision-4B judge control comparison:

```bash
.venv-intern-judge/bin/python scripts/gpu_run.py --gpu 0 -- \
  .venv-intern-judge/bin/python scripts/check_intern_decision_judge.py \
  --model /mnt/data/models/internlm/Intern-Decision-4B \
  --controls eval/evidence/xiangxin-behavior-controls-v1/controls.jsonl \
  --output runs/intern-decision-controls-repeat
```

The environment reuses the local `so` Python environment and adds matching
`torchvision==0.26.0+cu128` from PyTorch's CUDA 12.8 wheel index. This runs a
finite native inference batch, not a persistent HTTP service. Saved results are
in `eval/evidence/intern-decision-4b-controls-v1/`; both question rubrics passed
all eight basic behavioral controls. Unsafe classification remains unvalidated.
