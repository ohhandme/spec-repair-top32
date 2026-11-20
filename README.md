# 教师 Top-32 分数修复草稿

这个实验沿用 `TOP32_v2` 的修复头。从大模型上一轮验证的合法边界提取概率最高的 32 个候选 token 及其分数，再用原实现编码后喂给修复头。训练从同一初始权重开始，默认把草稿位置按 8 到 1 加权。`--loss equal` 保留旧等权口径，方便复现。

## 独立 A100

使用 Linux、Python 3.11+ 和与 CUDA 驱动匹配的 PyTorch，再安装 `requirements.txt`。把原机冻结输入按 `/data` 原目录布局复制或挂载。`native` 中的 `runtime_freeze.json` 给出文件和哈希。模型、RAW 分片、训练缓存、冻结 503 题及原 Top-32 初始权重从实验输入复制。

```bash
set -e
python experiment.py check-repo
python experiment.py install
python experiment.py preflight --verify-hashes
python experiment.py smoke --out /srv/spec-runs/top32
python experiment.py train --out /srv/spec-runs/top32
python experiment.py eval --out /srv/spec-runs/top32
```

临时训练配置、断点和结果位于 `/srv/spec-runs/top32/front`。使用等权对照时，在三个运行命令上加 `--loss equal`，结果位于 `equal`。两种配置共享初始权重、训练顺序、题单和预算，都按相同的等权校准交叉熵挑选一套权重来评估。评估逐题检查最终输出与封存的 Target 序列完全一致，汇报总接受 token 除以总验证轮数。

这是加权候补配置。已在本机核对冻结源码、原载入行为和两种权重配置。实际 BF16 冒烟和完整评估等待独立 A100。

窗口 87758 的修订使用仓内 `tie_v1/matrix_heads.py`，其 SHA 随新输出的 runtime_freeze.json 绑定。使用新的输出目录（如 /data/Task75-Top32-tie-v1），保留旧目录。原 /data 下的冻结源码继续按原哈希安装和校验。
