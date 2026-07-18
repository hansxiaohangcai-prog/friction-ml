# 项目交接文档 —— NAO 摩擦材料磨损率回归

> **用途**：下次新开 Claude 会话时，把本文件内容作为第一条消息贴入，即可无缝续接。
> **最后更新**：2026-07-16（阶段 2 主效应机理验收完成，破纪录 R²=0.892）

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

> ⚠️ **重要**：目前所有阶段 2 的成果（best_gb / best_gb_v2 / 各重要性表）都只在 `01_eda.ipynb` 里、且靠 Run All 重建。明天开工第一件事就是 Run All 恢复环境，再往下做。

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
- **坑**：log 空间的 R² 不能直接说成「对磨损率的预测精度」，评估时要单独处理（待做，阶段 4）

### 决策 B：特征方案选「A」——19 列原样全塞（共 23 特征）
- 当时三个选项：A 原样全塞 / B 去掉一列当参照 / C 对数比变换(ILR/CLR)
- **选 A 的理由**：主线是展示「线性 vs 树模型在闭合数据上的差异」，A 让差异最大化
- **C 方案（ILR/CLR 成分数据变换）留到阶段 3 做**，作为学术亮点（「病症」用 A 展示，「药方」用 C）

### 决策 C：新特征 `binder_total = phenolic + phenolic_modified`
- 领域知识注入：机理作用于「树脂总量」，模型不知道这两列该相加
- **坑**：若把 binder_total 和 phenolic+phenolic_modified 三列一起喂 → 完美共线。**正确做法：用 binder_total 替换掉那两列**（drop 掉 phenolic / phenolic_modified）
- ✅ **2026-07-16 已验证并大获成功**（见 §3 决策 C 的实证），这条从「注意事项」升级为「已证实的核心结论」

### 决策 D（2026-07-16 新增）：预测精度阶段就用 GBDT + binder_total 收尾，不再追求更高 R²
- 天花板在数据信噪比（小样本 + 18 噪声特征），不在算法
- 继续换 XGBoost/LightGBM 大概率仍在 0.70~0.74 晃，边际收益低
- 已拿到 R²=0.892（binder_total 版），远超一切基线，够写论文，转向核心交付（机理验收）

---

## 3. 关键结果（已验证，可写论文）

### EDA 阶段（07-15）
1. **目标对数正态**：偏度 3.14 → 0.06
2. **闭合是数学约束**：每列 vs 其余列之和 Pearson r = **-1.000**（19 列全部）；相关矩阵负相关来自组块零和 + 全局闭合，非物理相克
3. **Pearson 漏掉非单调机理**：phenolic 单列 r ≈ 0，但 binder_total 藏着抛物线，二次拟合最优 **13.25 wt%**（真值 13.00）；同时 mineral_fiber 等噪声特征因闭合产生虚假正相关（是 aramid 的「影子」）

### 基线模型阶段（07-15，选项 A，23 特征，160 训练 / 40 测试）
| | 训练 R² | 测试 R² | 测试 MAE(log) |
|---|---|---|---|
| 线性回归 | 0.795 | **0.735** | 0.271 |
| 随机森林(默认) | 0.949 | **0.627** | 0.287 |

- 随机森林默认参数**输给**线性回归 → 过拟合（训练 0.949 vs 测试 0.627）
- 线性回归系数不可解读（闭合共线导致系数失去区分度）→「能预测」≠「能解释」

### 阶段 2 — 模型调参 + 机理验收（07-16，本次核心交付）

#### (1) 调参：越挣扎越印证「小样本+高噪声，树模型难赢线性」

四方对比（测试集 R²，都在选项 A 的 23 特征上）：

| 模型 | 训练 R² | 测试 R² | 备注 |
|---|---|---|---|
| 线性回归 | 0.795 | **0.735** | 靶子 |
| 随机森林(默认) | 0.949 | 0.627 | 过拟合 |
| 随机森林(手调 v1) | 0.834 | 0.555 | 三个刹车一起踩，踩过头（偏差>方差收益） |
| 随机森林(网格搜索最优) | 0.945 | 0.595 | 机器选 `min_samples_leaf=1` = 拒绝踩刹车 |
| **GBDT(网格搜索最优)** | 0.988 | **0.705** | 树模型冠军，逼近线性但仍差 0.03 |

- **RF 网格搜索选出 `min_samples_leaf=1`**：说明这份数据上稍压方差、真信号损失就 > 噪声收益 → RF 结构性打不过线性
- **GBDT 最优参数**：`learning_rate=0.02, max_depth=2, n_estimators=800, subsample=0.7`
  - 每个抗噪旋钮都拧到最保守（最慢学习率、最浅树、subsample 开满）
  - 「浅树 + 慢学习 + 数量补偿」这套抗噪逻辑成立 → GBDT 完胜所有 RF
  - 但训练 0.988 vs 测试 0.705（劈叉 0.28）→ 800 棵树累积灵活性仍略过量，没能翻过线性回归
- **结论固化**：核心教训第 4 条（小样本+高噪声，默认/调参 RF 都未必打得过线性）在本数据上**被三个树模型从三个角度反复证实**，是本工作最硬的实证之一

#### (2) 决策 C 的实证：领域知识 > 蛮力调参（★本项目最强结果）

只做一件事——把 phenolic + phenolic_modified 合体成 binder_total（并 drop 原两列，23→22 特征），
**同样的 GBDT 最优参数**重训：

| | 测试 R² |
|---|---|
| GBDT（拆开两列，23 特征） | 0.705 |
| **GBDT（binder_total，22 特征）** | **0.892** |

- **一个加法（两列合体）+ 一个减法（去共线），测试 R² 暴涨 +0.187**，不换算法、不加数据、不调参
- 前面三个模型 + 两轮网格搜索去追 0.03 差距追不动；一句材料学常识直接 +0.19
- ★ **完美实证核心教训第 5 条**：「哪两列该相加」是领域判断，算法自己长不出来。**领域知识 > 蛮力调参**。这张对比表 = 论文核心图表之一

#### (3) 特征重要性：主效应机理 6/7 命中，噪声全部踩下

两种方法交叉验证（都在 binder_total 版 GBDT 上）：

**方法一：不纯度下降重要性（`feature_importances_`）**——快、免费，但偏爱连续/多取值特征（当初筛）
**方法二：置换重要性（`permutation_importance`，测试集，n_repeats=30）**——更严谨，测试集上算，不偏袒连续特征，还给误差棒（写方法学用这个）

置换重要性排名（值 ± 标准差）：

| 排名 | 特征 | 置换重要性 | 是否真机理 |
|---|---|---|---|
| 1 | aramid | 0.7476 ± 0.1183 | ✅ 断层第一 |
| 2 | binder_total | 0.3512 ± 0.0728 | ✅ 合体后窜到第2 |
| 3 | graphite | 0.0892 ± 0.0266 | ✅ |
| 4 | Al2O3 | 0.0884 ± 0.0218 | ✅ |
| 5 | friction_dust | 0.0824 ± 0.0227 | ✅ |
| 6 | post_cure_temp_C | 0.0485 ± 0.0149 | ✅ |
| 7 | mineral_fiber | 0.0055 ± 0.0018 | ✗ 悬崖下，aramid 闭合影子残留 |
| … | （其余噪声） | ≤ 0.005，多个为负 | ✗ |
| 18-22 | press_temp_C / coke / vermiculite / NBR_powder / zircon | **负值** | ✗ 纯噪声铁证 |

**三条可写进论文的观察**：
1. **两种方法前 6 名完全一致** → 单一方法可能有偏差，两种独立方法交叉一致 = 结论可信
2. **负置换重要性 = 纯噪声正面证据**：打乱它 R² 反而微升，说明该列不含真信号、还偶尔引入干扰。比「重要性≈0」更强
3. **第 6 与第 7 之间有数量级悬崖**（0.048 vs 0.005）→ 干净切开「6 条真机理」与「噪声」；前 6 名值均 ≥ 误差棒的 3 倍，显著非零、稳健

**主效应验收结算：6 个主效应 100% 命中，两种方法交叉确认。仅剩第 7 条交互项（重要性表结构上看不见，须 PDP）**

---

## 4. 核心教训（概念）

1. **看测试集，不看训练集**（训练 R² 高可能是过拟合）
2. **过拟合信号**：训练 R² 远高于测试 R²
3. **预测 ≠ 解释**：R² 高不代表系数/机理可信
4. **小样本 + 高噪声上，默认/调参随机森林都可能打不过线性回归**（本数据用 3 个树模型反复证实）；GBDT「浅树+慢学习」抗噪明显优于 RF，但仍受数据信噪比天花板限制
5. **特征工程是材料人的活**：「哪两列该相加」是领域判断，算法自己长不出来 →★ 本次 binder_total 合体 R² +0.19 是硬证据
6. **重要性要两种方法交叉验**：不纯度重要性偏爱连续特征（当初筛），置换重要性更公正（测试集、带误差棒、能出负值识别噪声）
7. **单变量重要性看不见交互项**：交互是「两特征合作」的效应，逐特征打分显不出来，必须用（交互）PDP

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

# ---- 建模数据（选项A：23列） ----
from sklearn.model_selection import train_test_split
process_cols = ["press_temp_C","press_pressure_MPa","press_time_min","post_cure_temp_C"]
feat_cols = comp_cols + process_cols
X, y = df[feat_cols], df["log_wear"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)     # 固定切分，可复现

# ---- 基线 ----
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
lin = LinearRegression().fit(X_train, y_train)
pred_test = lin.predict(X_test)

# ---- GBDT 网格搜索（阶段2）----
from sklearn.model_selection import GridSearchCV
gb_grid = {"n_estimators":[200,400,800], "learning_rate":[0.02,0.05,0.1],
           "max_depth":[2,3], "subsample":[0.7,1.0]}
gb_search = GridSearchCV(GradientBoostingRegressor(random_state=42),
                         gb_grid, cv=5, scoring="r2", n_jobs=-1).fit(X_train, y_train)
best_gb = gb_search.best_estimator_
# best_params: learning_rate=0.02, max_depth=2, n_estimators=800, subsample=0.7 → 测试 R²=0.705

# ---- ★ binder_total 合体：用它替换两列，勿三列同喂 ----
X_train_v2 = X_train.copy(); X_test_v2 = X_test.copy()
X_train_v2["binder_total"] = X_train_v2["phenolic"] + X_train_v2["phenolic_modified"]
X_test_v2["binder_total"]  = X_test_v2["phenolic"]  + X_test_v2["phenolic_modified"]
X_train_v2 = X_train_v2.drop(columns=["phenolic","phenolic_modified"])
X_test_v2  = X_test_v2.drop(columns=["phenolic","phenolic_modified"])
best_gb_v2 = GradientBoostingRegressor(**gb_search.best_params_, random_state=42).fit(X_train_v2, y_train)
# 测试 R² = 0.892（+0.187 vs 拆开两列）

# ---- 置换重要性（写方法学用这个，测试集）----
from sklearn.inspection import permutation_importance
perm = permutation_importance(best_gb_v2, X_test_v2, y_test,
                              n_repeats=30, random_state=42, scoring="r2", n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=X_test_v2.columns).sort_values(ascending=False)
perm_std = pd.Series(perm.importances_std,  index=X_test_v2.columns)
```

---

## 6. 文件 / Git 状态

- 仓库：`~/projects/friction-ml`（本地已建，**GitHub 远程未推**，SSH key 未配 → 待做，阶段 4）
- Commit 历史（07-15 及以前）：
  - `c6d920f` 数据生成脚本
  - `c82e5c1` EDA notebook
  - `6fe8594` 基线模型
- ⚠️ **07-16 的阶段 2 工作（调参 + binder_total + 两种重要性）尚未 commit** → 明天开工可先补一个 commit
- 工作文件：`01_eda.ipynb`（EDA + 基线 + 阶段2 都在里面，10+ cells）
- 数据：`friction_formulation.csv`、脚本 `gen_friction_data.py`

---

## 7. 下一步路线图

**阶段 2：模型解释 = 验收机理（核心交付）— 进行中**
- [x] 随机森林调参（踩刹车防过拟合）→ 证实结构性打不过线性
- [x] GBDT 网格搜索 → 树模型冠军 R²=0.705；binder_total 合体后 R²=0.892
- [x] 特征重要性（不纯度）逐条对回机理 → 主效应 6/7 命中
- [x] 置换重要性复核（测试集，带误差棒）→ 两法交叉一致，负重要性识别噪声
- [ ] **PDP（部分依赖图）验形状**：aramid/graphite 对数饱和、Al2O3 平方、binder_total U型、friction_dust/post_cure 线性 ← **明天第一件事**
- [ ] **交互 PDP（2D）专验第 7 条 aramid×binder 交互项**（单变量完全看不见）

**阶段 3：闭合数据正确处理（学术亮点）**
- [ ] 成分数据对数比变换（ILR/CLR），对比选项 A 的「病症」

**阶段 4：收尾**
- [ ] log 空间 R² 如何正确解读成磨损率精度
- [ ] 结论整理
- [ ] 补 07-16 的 commit；配 SSH key，推 GitHub

---

## 会话命名索引（每天一条，便于知识库检索）

- NAO摩擦材料ML-01-字段定义+数据生成（07-14 及以前）
- NAO摩擦材料ML-02-基线模型(EDA收尾+线性vs随机森林)（07-15）
- **NAO摩擦材料ML-03-阶段2机理验收(RF/GBDT调参+特征重要性+置换重要性)（07-16，本次）**
- （下次）NAO摩擦材料ML-04-PDP验形状+交互项 …

---

## 下次开场白（建议直接说）

> 「继续这个项目（ML-04），先把 notebook Run All Cells 恢复环境，然后做 PDP 验收 6 条主效应的形状，最后用 2D 交互 PDP 验第 7 条 aramid×binder 交互项。」
