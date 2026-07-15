# 项目交接文档 —— NAO 摩擦材料磨损率回归

> **用途**：下次新开 Claude 会话时，把本文件内容作为第一条消息贴入，即可无缝续接。
> **最后更新**：2026-07-15（基线模型完成后）

---

## 0. 我是谁 / 怎么带我

- 材料人（硕士，纤维增强树脂基复合材料 / 摩擦学）
- Python 和命令行**零基础**，做中学
- 带我的方式：
  - **一次只给一步**，我做完回你再给下一步
  - 每步先说「在干什么、为什么」，再给命令
  - 不要一次抛多个工具（终端 / VS Code / Jupyter 来回跳会懵）
  - 目标是「懂材料 + 能建模」，不是转行做 ML 工程师

## 环境

- MacBook Air M4
- conda 环境名：`mat`（包全用 `pip` 装的，**别混用 conda install**）
- 项目目录：`~/projects/friction-ml`
- 已装：jupyterlab 4.6.1 / numpy 2.4.6 / pandas 2.3.3 / scikit-learn 1.9.0 / seaborn 0.13.2 / matplotlib
- **开工老三样**：
  ```bash
  conda activate mat
  cd ~/projects/friction-ml && jupyter lab
  ```
  打开 `01_eda.ipynb` → 菜单 Run → Run All Cells（kernel 重启后变量清空，必须重跑）

---

## 1. 项目概况

NAO（无石棉有机型）纤维增强树脂基摩擦材料：
**配方(19 组分, wt%) + 工艺(4 参数) → 磨损率 wear_rate 回归预测**

- 数据：**合成数据**，200 条，机理已知（生成脚本 `gen_friction_data.py` 在仓库里）
- 数据文件：`friction_formulation.csv`（25 列 = 1 batch_id + 19 配方 + 4 工艺 + 1 wear_rate）

### 真值机理（我自己造的，知道答案 → 用来验收模型）

| # | 机理 | 脚本公式 | 形状 |
|---|---|---|---|
| 1 | 树脂总量 (phenolic + phenolic_modified) | `0.030*(binder-13)²` | **非单调**，最优 ≈ 13 wt% |
| 2 | 芳纶 aramid | `-0.55*log1p(aramid)` | 减磨，对数饱和 |
| 3 | 石墨 graphite | `-0.40*log1p(graphite)` | 减磨，对数饱和 |
| 4 | Al2O3 | `+0.020*Al2O3²` | 增磨，平方（超线性） |
| 5 | 腰果壳粉 friction_dust | `+0.045*friction_dust` | 增磨，线性 |
| 6 | 后固化温度 post_cure_temp_C | `-0.008*(post_cure-160)` | 减磨，线性 |
| 7 | 芳纶 × 树脂 | `-0.010*aramid*(binder-10)` | **交互项** |

- 其余 15 个成分 + 3 个工艺参数(press_temp/pressure/time) = **纯噪声特征**，未进公式，是对照组
- 噪声：10% 对数正态（相对误差）
- **验收标准**：模型特征重要性应能找回上述 7 条，且不把噪声特征错认为重要

### 关键认知：成分闭合

19 个配方列之和恒 = 100 wt% → 列间非独立，只有 18 个自由度。
相关矩阵上会出现「假负相关」，是**数学伪影，不是物理规律**。

---

## 2. 已完成的关键决策（重要，别推翻）

### 决策 A：建模目标用 `log(wear_rate)`，不用原始值
- 证据：wear_rate 偏度 3.14，取 log 后 0.06（对数正态，符合生成机理 `wear=exp(log_wear)` + lognormal 噪声）
- 三个理由：① 真值加性机理在 log 空间（脚本里各项相加）② 10% 相对噪声 → log 后变等宽绝对误差 ③ 摩擦学惯例（Archard 模型本身乘性）
- **坑**：log 空间的 R² 不能直接说成「对磨损率的预测精度」，评估时要单独处理（待做）

### 决策 B：特征方案选「A」——19 列原样全塞（共 23 特征）
- 当时三个选项：A 原样全塞 / B 去掉一列当参照 / C 对数比变换(ILR/CLR)
- **选 A 的理由**：主线是展示「线性 vs 树模型在闭合数据上的差异」，A 让差异最大化
- **C 方案（ILR/CLR 成分数据变换）留到阶段 3 做**，作为学术亮点（「病症」用 A 展示，「药方」用 C）

### 决策 C：新特征 `binder_total = phenolic + phenolic_modified`
- 领域知识注入：机理作用于「树脂总量」，模型不知道这两列该相加
- **坑**：若把 binder_total 和 phenolic+phenolic_modified 三列一起喂线性模型 → 完美共线，模型崩。建模时注意

---

## 3. 关键结果（已验证，可写论文）

### EDA 阶段
1. **目标对数正态**：偏度 3.14 → 0.06
2. **闭合是数学约束**：每列 vs 其余列之和 Pearson r = **-1.000**（19 列全部）；相关矩阵负相关来自组块零和 + 全局闭合，非物理相克
3. **Pearson 漏掉非单调机理**：phenolic 单列 r ≈ 0，但 binder_total 藏着抛物线，二次拟合最优 **13.25 wt%**（真值 13.00）；同时 mineral_fiber 等噪声特征因闭合产生虚假正相关（是 aramid 的「影子」）

### 基线模型阶段（选项 A，23 特征，160 训练 / 40 测试）
| | 训练 R² | 测试 R² | 测试 MAE(log) |
|---|---|---|---|
| 线性回归 | 0.795 | **0.735** | 0.271 |
| 随机森林(默认参数) | 0.949 | **0.627** | 0.287 |

- **随机森林默认参数反而输给线性回归** → 过拟合（训练 0.949 vs 测试 0.627 劈叉）。原因：160 条小样本 + 18 噪声特征，默认 RF 太奔放，把噪声当规律学了
- **线性回归系数不可解读**：最大 5 个系数全挤在 -13.7 附近（aramid 和纯噪声 chromite/MoS2 几乎相同），闭合共线导致系数失去区分度 → **「能预测」≠「能解释」**
- 记住靶子：**测试 R² = 0.735** 是随机森林调参后要打败的目标

---

## 4. 核心教训（概念）

1. **看测试集，不看训练集**（训练 R² 高可能是过拟合）
2. **过拟合信号**：训练 R² 远高于测试 R²
3. **预测 ≠ 解释**：R² 高不代表系数/机理可信
4. **小样本 + 高噪声上，默认随机森林会过拟合**，未必打得过线性回归（很多论文「RF 吊打线性」的结论在小样本材料数据上不成立）
5. **特征工程是材料人的活**：「哪两列该相加」是领域判断，算法自己长不出来

---

## 5. 关键代码片段（可复用）

```python
# ---- 读数 + 自检 ----
import pandas as pd, numpy as np
df = pd.read_csv("friction_formulation.csv")
comp_cols = [c for c in df.columns
             if c not in ["batch_id","press_temp_C","press_pressure_MPa",
                          "press_time_min","post_cure_temp_C","wear_rate"]]  # 19 配方列(排除法)

# ---- 目标 + 领域特征 ----
df["log_wear"] = np.log(df["wear_rate"])
df["binder_total"] = df["phenolic"] + df["phenolic_modified"]

# ---- 建模数据 ----
from sklearn.model_selection import train_test_split
process_cols = ["press_temp_C","press_pressure_MPa","press_time_min","post_cure_temp_C"]
feat_cols = comp_cols + process_cols          # 选项A：23列
X, y = df[feat_cols], df["log_wear"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)     # random_state=42 = 固定切分，可复现

# ---- 两个基线 ----
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
lin = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_train, y_train)
```

---

## 6. 文件 / Git 状态

- 仓库：`~/projects/friction-ml`（本地已建，**GitHub 远程未推**，SSH key 未配 → 待做）
- Commit 历史：
  - `c6d920f` 数据生成脚本
  - `c82e5c1` EDA notebook
  - `6fe8594` 基线模型
- 工作文件：`01_eda.ipynb`（EDA + 基线模型都在里面，6+ cells）
- 数据：`friction_formulation.csv`、脚本 `gen_friction_data.py`

---

## 7. 下一步路线图（未完成）

**阶段 2：模型解释 = 验收机理（核心交付）**
- [ ] 随机森林调参（限制 max_depth、min_samples_leaf 等，踩刹车防过拟合），目标测试 R² > 0.735
- [ ] 特征重要性，逐条对回 7 条机理
- [ ] 部分依赖图(PDP)：一次性验证 aramid/graphite 对数饱和、Al2O3 平方、树脂 U 型等形状
- [ ] 交互项 aramid×binder 专门用 PDP 验（单变量完全看不见）

**阶段 3：闭合数据正确处理（学术亮点）**
- [ ] 成分数据对数比变换（ILR/CLR），对比选项 A 的「病症」

**阶段 4：收尾**
- [ ] log 空间 R² 如何正确解读成磨损率精度
- [ ] 结论整理
- [ ] 配 SSH key，推 GitHub

---

## 下次开场白（建议直接说）

> 「继续这个项目，先把 notebook Run All Cells 恢复环境，然后给随机森林调参 + 拉特征重要性」
