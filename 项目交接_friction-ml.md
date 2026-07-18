# 项目交接文档 —— NAO 摩擦材料磨损率回归

> **用途**：下次新开 Claude 会话时，把本文件内容作为第一条消息贴入，即可无缝续接。
> **最后更新**：2026-07-18（**阶段 2 完结**：PDP 验形状 6/6 + 交互项马鞍面定量验收 r=0.825，7/7 机理全部找回）

---

## 0. 我是谁 / 怎么带我

- 材料人（硕士，纤维增强树脂基复合材料 / 摩擦学）
- Python 和命令行**零基础**，做中学
- 带我的方式：
  - **一次只给一步**，我做完回你再给下一步
  - 每步先说「在干什么、为什么」，再给命令
  - 不要一次抛多个工具（终端 / VS Code / Jupyter 来回跳会懵）
  - 目标是「懂材料 + 能建模」，不是转行做 ML 工程师

### ⏰ 两条固定约定（每次会话都要执行，别等我开口）

**约定 1：阶段性成果做完就提醒我 commit，并把 commit 内容直接写好给我**

- 触发时机：每完成一个有实质产出的小节就提醒一次（拿到一个新结果、跑通一张关键图、确立一个新结论），**不要攒到收工才提**
- 提醒时必须**连同完整命令一起给出**，我直接复制粘贴，不用自己想怎么写：
  - 先提醒我在 Jupyter 里 **⌘S 保存**（Git 存的是硬盘文件，不是内存）
  - 再给 `git add ... ` → `git status` → `git commit -m "..."` 全套
  - commit message 由你起草：**标题一行讲清做了什么，正文列出关键数字**（R²、r、参数等），别写「更新代码」这种废话
- 原因（07-18 的教训）：两天工作堆在同一个 `.ipynb` 里，Git 无法拆成两笔 commit，历史就糊了

**约定 2：每次收工时，给出本次会话的建议命名**

- 收工的判定：我说「收工 / 今天到这 / 先这样」，或一个阶段告一段落时
- 命名格式固定：`NAO摩擦材料ML-<两位序号>-<本次核心内容+关键数字>（<日期>）`
  - 例：`NAO摩擦材料ML-04-PDP验形状+交互项马鞍面定量验收(r=0.825)（07-18）`
- 要求：**带上本次最硬的那个数字**（R²、r、命中率等），方便日后在知识库里一眼检索到
- 同时把这条追加进本文档 §8「会话命名索引」，并更新「下次开场白」

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

> ⚠️ 所有模型对象（best_gb / best_gb_v2 / PDP 结果）都只在 `01_eda.ipynb` 里、靠 Run All 重建。开工第一件事永远是 Run All。

### 已踩过的环境坑（省时间）

| 坑 | 正确做法 |
|---|---|
| `PartialDependenceDisplay.from_estimator` 与 `partial_dependence` 的 features 写法**不一样** | 前者二维交互用**列表套元组** `[("a","b")]`；后者用**平列表** `["a","b"]` |
| NumPy 2.x 删了 `数组.ptp()` | 改用 `np.ptp(数组)` |
| Jupyter 内存 ≠ 硬盘文件 | commit 前必须先在浏览器 **⌘S** 保存，再 `ls -l` 看时间戳确认 |
| 要同时跑 Jupyter 和终端命令 | 终端里 **⌘T** 开新标签页；跑 Jupyter 那个标签**别关别 Ctrl+C** |

---

## 1. 项目概况

NAO（无石棉有机型）纤维增强树脂基摩擦材料：
**配方(19 组分, wt%) + 工艺(4 参数) → 磨损率 wear_rate 回归预测**

- 数据：**合成数据**，200 条，机理已知（生成脚本 `gen_friction_data.py` 在仓库里）
- 数据文件：`friction_formulation.csv`（25 列 = 1 batch_id + 19 配方 + 4 工艺 + 1 wear_rate）

### 真值机理（我自己造的，知道答案 → 用来验收模型）

| # | 机理 | 脚本公式 | 形状 | 验收状态 |
|---|---|---|---|---|
| 1 | 树脂总量 (phenolic + phenolic_modified) | `0.030*(binder-13)²` | **非单调**，最优 ≈ 13 wt% | ✅ |
| 2 | 芳纶 aramid | `-0.55*log1p(aramid)` | 减磨，对数饱和 | ✅ |
| 3 | 石墨 graphite | `-0.40*log1p(graphite)` | 减磨，对数饱和 | ✅ |
| 4 | Al2O3 | `+0.020*Al2O3²` | 增磨，平方（超线性） | ✅ |
| 5 | 腰果壳粉 friction_dust | `+0.045*friction_dust` | 增磨，线性 | ✅ |
| 6 | 后固化温度 post_cure_temp_C | `-0.008*(post_cure-160)` | 减磨，线性 | ✅ |
| 7 | 芳纶 × 树脂 | `-0.010*aramid*(binder-10)` | **交互项** | ✅ |

- 其余 15 个成分 + 3 个工艺参数(press_temp/pressure/time) = **纯噪声特征**，未进公式，是对照组
- 噪声：10% 对数正态（相对误差）
- **验收标准**：模型应能找回上述 7 条，且不把噪声特征错认为重要 → **已 7/7 达成，见 §3.3**

### 关键认知：成分闭合

19 个配方列之和恒 = 100 wt% → 列间非独立，只有 18 个自由度。
相关矩阵上会出现「假负相关」，是**数学伪影，不是物理规律**。

---

## 2. 已完成的关键决策（重要，别推翻）

### 决策 A：建模目标用 `log(wear_rate)`，不用原始值
- 证据：wear_rate 偏度 3.14，取 log 后 0.06
- 三个理由：① 真值加性机理在 log 空间 ② 10% 相对噪声 → log 后变等宽绝对误差 ③ 摩擦学惯例（Archard 模型本身乘性）
- **坑**：log 空间的 R² 不能直接说成「对磨损率的预测精度」→ 阶段 4 待处理

### 决策 B：特征方案选「A」——19 列原样全塞
- **选 A 的理由**：主线是展示「线性 vs 树模型在闭合数据上的差异」，A 让差异最大化
- **C 方案（ILR/CLR 成分数据变换）留到阶段 3**（「病症」用 A 展示，「药方」用 C）

### 决策 C：新特征 `binder_total = phenolic + phenolic_modified`
- **正确做法：用 binder_total 替换掉那两列**（drop 原两列），三列同喂 = 完美共线
- ✅ 已验证，测试 R² +0.187，本项目最强结果

### 决策 D：预测精度阶段就用 GBDT + binder_total 收尾
- 天花板在数据信噪比（小样本 + 18 噪声特征），不在算法；R²=0.892 够写论文

### 决策 E（07-18 新增）：PDP 只验形状，不细读绝对幅度
- 所有 PDP 的幅度都系统性小于真值（收缩效应），如 post_cure 真值跨 0.42、模型只跨 0.23
- 稀疏区（rug 竖线拉开处）的台阶高度不可细读，只读趋势

---

## 3. 关键结果（已验证，可写论文）

### 3.1 EDA 阶段（07-15）
1. **目标对数正态**：偏度 3.14 → 0.06
2. **闭合是数学约束**：每列 vs 其余列之和 Pearson r = **-1.000**（19 列全部）
3. **Pearson 漏掉非单调机理**：phenolic 单列 r ≈ 0，但 binder_total 二次拟合最优 **13.25 wt%**（真值 13.00）

### 3.2 模型阶段（07-15 / 07-16）

| 模型 | 训练 R² | 测试 R² | 备注 |
|---|---|---|---|
| 线性回归 | 0.795 | **0.735** | 靶子 |
| 随机森林(默认) | 0.949 | 0.627 | 过拟合 |
| 随机森林(手调 v1) | 0.834 | 0.555 | 三个刹车一起踩，踩过头 |
| 随机森林(网格最优) | 0.945 | 0.595 | 机器选 `min_samples_leaf=1` = 拒绝踩刹车 |
| GBDT(网格最优，23 特征) | 0.988 | 0.705 | 树模型冠军，仍输线性 |
| **GBDT(binder_total，22 特征)** | — | **0.892** | ★ 一个加法+一个减法，+0.187 |

- GBDT 最优参数：`learning_rate=0.02, max_depth=2, n_estimators=800, subsample=0.7`
- **结论固化**：小样本+高噪声下 RF 结构性打不过线性；GBDT「浅树+慢学习」抗噪更优但仍受信噪比天花板限制
- ★ **领域知识 > 蛮力调参**：三个模型两轮网格搜索追 0.03 追不动，一句材料学常识直接 +0.19

### 3.3 机理验收（07-16 重要性 + 07-18 PDP）★ 核心交付

#### (1) 置换重要性（测试集，n_repeats=30）—— 找出「哪几条重要」

| 排名 | 特征 | 置换重要性 | 真机理 |
|---|---|---|---|
| 1 | aramid | 0.7476 ± 0.1183 | ✅ 断层第一 |
| 2 | binder_total | 0.3512 ± 0.0728 | ✅ |
| 3 | graphite | 0.0892 ± 0.0266 | ✅ |
| 4 | Al2O3 | 0.0884 ± 0.0218 | ✅ |
| 5 | friction_dust | 0.0824 ± 0.0227 | ✅ |
| 6 | post_cure_temp_C | 0.0485 ± 0.0149 | ✅ |
| 7 | mineral_fiber | 0.0055 ± 0.0018 | ✗ aramid 闭合影子 |
| 18–22 | press_temp_C / coke / vermiculite / NBR_powder / zircon | **负值** | ✗ 纯噪声铁证 |

- 不纯度重要性与置换重要性**前 6 名完全一致** → 两法交叉可信
- **负置换重要性 = 纯噪声正面证据**，比「重要性≈0」更强
- **第 6 与第 7 之间数量级悬崖**（0.048 vs 0.005），干净切开真机理与噪声

#### (2) 1D PDP（训练集，grid_resolution=50）—— 验「重要成什么样」

**6/6 形状全部复现**：

| 特征 | 真值形状 | PDP 观察 |
|---|---|---|
| aramid | 对数饱和减磨 | 下降，0→3 陡、6 后压平；纵轴跨度 1.386（最大） |
| binder_total | U 型，谷底 13 | 谷底平坦区 11.5–15，两端翘起；跨度 0.522 |
| graphite | 对数饱和减磨 | 下降+饱和，幅度约 aramid 一半 |
| Al2O3 | 平方增磨 | 上升且右端更陡（凸形明确） |
| friction_dust | 线性增磨 | 近直线上升 |
| post_cure_temp_C | 线性减磨 | 直线下降，跨度最小 0.23 |

**三条可写论文的观察**：
1. **PDP 纵轴跨度排序 = 置换重要性排序**（1.386 / 0.522 / … vs 0.7476 / 0.3512 / …）→ 第三个独立方法交叉一致
2. **曲线呈阶梯状不是 bug**，是树模型本质（分段常数）；台阶密处 = 模型切分多 = 信息量大
3. **幅度普遍收缩**（真值 0.42 → 模型 0.23）→ 噪声大样本少时正则化让模型不敢学满；**形状可靠、幅度保守**

**关于 rug（横轴小竖线）的正确理解**（曾误解，已纠正）：
- 它是**十分位刻度**（10%、20%…90% 分位），**不是数据范围边界**；最后一根竖线右边还有 10% 样本
- PDP 网格取 5%~95% 分位，**曲线全程在数据内，从不外推**
- 补充：**树模型结构上无法外推**——超出训练范围后是水平直线（永远落进同一个叶子），与线性回归飞出去截然不同
- 实测 Al2O3：十分位 `0.45 0.77 1.09 1.45 1.82 2.21 2.75 3.33 4.22`，最大值 7.26，90% 分位右边还有 16 个样本；间距从 0.32 拉大到 0.89 → 右尾稀疏是量化事实 → 台阶被放大

#### (3) 交互项验收（07-18）—— 第 7 条只能靠这个

**为什么 1D PDP 看不见交互**：1D PDP 最后一步「对其余变量取平均」，恰好把「aramid 的效果取决于 binder 多少」抹平成中间曲线。

**方法：交互残差面**
```
纯交互残差 = PD_2D(a,b) − PD_1D(a) − PD_1D(b)
```
右边两项 = 「若只有加性效应该长什么样」，减掉后剩的只能是交互。

> ⚠️ **踩过的坑**：直接看 2D PDP 等高线「不平行」**不能**证明交互。U 型主效应 + 单调主效应加性叠加，等高线本来就会腰部内凹。必须扣掉加性部分才算证据。

**定性结果：马鞍面，四角符号全对**

| 区域 | 应为 | 实测 | 材料学读法 |
|---|---|---|---|
| 右上（高纤维·高树脂） | 蓝 | 深蓝 −0.16 ✅ | 协同减磨：树脂足→纤维锚固牢 |
| 左上（低纤维·高树脂） | 红 | 深红 +0.13 ✅ | |
| 右下（高纤维·低树脂） | 红 | 红 ✅ | 树脂不足堆纤维不划算（拔出/剥落） |
| 左下（低纤维·低树脂） | 蓝 | 淡蓝 ✅ | |

对角同色、相邻异色 = 马鞍面定义；加性组合不可能产生此花纹。

**定量结果（★ 合成数据独有的奢侈）**

真值双重中心化后与模型残差面逐点比对：

| 指标 | 值 | 含义 |
|---|---|---|
| 相关系数 r | **0.825** | 形状一致；r²=0.68，68% 空间变异被真值解释 |
| 幅度还原率 k | **0.575** | 只学到约六成强度（收缩） |
| 真值面幅度 | −0.201 ~ +0.201 | 完美对称（精确双线性） |
| 模型面幅度 | −0.135 ~ +0.185 | **不对称** |

- 交互项量级 ≈ aramid 主效应的 21% → 解释了它为何在单变量重要性表上完全不露脸
- **不对称本身是信息**：红侧还原 92%、蓝侧仅 67% → 「树脂不足时堆纤维的惩罚」比「树脂充足时的协同增效」更易被抓到，因为前者对应高磨损样本、信号显眼，后者在低磨损区被 10% 噪声盖掉更多
- 一句话结论：**方向 100% 正确，形状 83% 一致，强度保守 57%**

**两个诚实的瑕疵（写论文要说）**：
1. 中间大片（binder 12–16）近乎空白 —— 交互项本身在中心趋零 + 中心区拟合最保守
2. aramid≈2.4、binder≈16.3 处有突兀直边 —— **GBDT 切分边界，不是物理临界值**，别解读成「树脂临界 16.3」

**结算：7/7 机理全部找回，18 个噪声特征全部踩下。阶段 2 完结。**
论文核心证据链闭合：**能预测（R²=0.892）→ 能解释（7/7 机理）→ 能量化（r=0.825）**

---

## 4. 核心教训（概念）

1. **看测试集，不看训练集**
2. **过拟合信号**：训练 R² 远高于测试 R²
3. **预测 ≠ 解释**：R² 高不代表机理可信
4. **小样本 + 高噪声上，默认/调参随机森林都可能打不过线性回归**（3 个树模型反复证实）；GBDT「浅树+慢学习」抗噪更优但受信噪比天花板限制
5. **特征工程是材料人的活**：「哪两列该相加」是领域判断 →★ binder_total 合体 R² +0.19
6. **重要性要两种方法交叉验**：不纯度偏爱连续特征，置换重要性更公正（测试集、带误差棒、能出负值识别噪声）
7. **单变量重要性看不见交互项**，必须用交互 PDP
8. **（07-18 新）重要性回答「哪个重要」，PDP 回答「重要成什么样」**——两者是不同层次的问题，缺一不可
9. **（07-18 新）2D PDP 等高线扭曲 ≠ 存在交互**——U型+单调的加性叠加也会扭曲；必须扣掉两条主效应才是证据
10. **（07-18 新）视觉判读要升级成数字**：「看起来像马鞍」会被质疑先入为主，r=0.825 才是客观量。这是合成数据的独有价值
11. **（07-18 新）模型对效应强度系统性收缩**：形状可信、幅度保守，报告时要说明
12. **（07-18 新）commit 要趁热打**：两天工作堆在同一文件里，Git 无法拆成两笔 commit。每完成一小节就提交

---

## 5. 关键代码片段（可复用）

```python
# ---- 读数 + 目标 + 领域特征 ----
import pandas as pd, numpy as np
df = pd.read_csv("friction_formulation.csv")
comp_cols = [c for c in df.columns
             if c not in ["batch_id","press_temp_C","press_pressure_MPa",
                          "press_time_min","post_cure_temp_C","wear_rate"]]
df["log_wear"] = np.log(df["wear_rate"])

# ---- 切分（固定 random_state，可复现）----
from sklearn.model_selection import train_test_split
process_cols = ["press_temp_C","press_pressure_MPa","press_time_min","post_cure_temp_C"]
X, y = df[comp_cols + process_cols], df["log_wear"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- GBDT 网格搜索 ----
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
gb_grid = {"n_estimators":[200,400,800], "learning_rate":[0.02,0.05,0.1],
           "max_depth":[2,3], "subsample":[0.7,1.0]}
gb_search = GridSearchCV(GradientBoostingRegressor(random_state=42),
                         gb_grid, cv=5, scoring="r2", n_jobs=-1).fit(X_train, y_train)

# ---- ★ binder_total 合体：替换两列，勿三列同喂 ----
X_train_v2 = X_train.copy(); X_test_v2 = X_test.copy()
for D in (X_train_v2, X_test_v2):
    D["binder_total"] = D["phenolic"] + D["phenolic_modified"]
X_train_v2 = X_train_v2.drop(columns=["phenolic","phenolic_modified"])
X_test_v2  = X_test_v2.drop(columns=["phenolic","phenolic_modified"])
best_gb_v2 = GradientBoostingRegressor(**gb_search.best_params_,
                                       random_state=42).fit(X_train_v2, y_train)
# 测试 R² = 0.892

# ---- 置换重要性（测试集）----
from sklearn.inspection import permutation_importance
perm = permutation_importance(best_gb_v2, X_test_v2, y_test,
                              n_repeats=30, random_state=42, scoring="r2", n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=X_test_v2.columns).sort_values(ascending=False)

# ---- 1D PDP（训练集）----
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]   # macOS 中文
plt.rcParams["axes.unicode_minus"] = False
six = ["aramid","binder_total","graphite","Al2O3","friction_dust","post_cure_temp_C"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
PartialDependenceDisplay.from_estimator(best_gb_v2, X_train_v2, features=six,
                                        ax=axes.ravel(), grid_resolution=50)
plt.tight_layout(); plt.show()

# ---- ★ 交互残差面（注意：partial_dependence 用平列表，不是元组）----
r2 = partial_dependence(best_gb_v2, X_train_v2, features=["aramid","binder_total"],
                        grid_resolution=20, kind="average")
rA = partial_dependence(best_gb_v2, X_train_v2, features=["aramid"],
                        grid_resolution=20, kind="average")
rB = partial_dependence(best_gb_v2, X_train_v2, features=["binder_total"],
                        grid_resolution=20, kind="average")
gv = r2["grid_values"] if "grid_values" in r2 else r2["values"]
ga, gb = gv[0], gv[1]
Z2, fA, fB = r2["average"][0], rA["average"][0], rB["average"][0]
inter = Z2 - fA[:, None] - fB[None, :]
inter = inter - inter.mean()

# ---- 定量验收：与真值面比 ----
A_grid, B_grid = np.meshgrid(ga, gb, indexing="ij")
T_true = -0.010 * A_grid * (B_grid - 10.0)
def double_center(M):     # 剥掉所有加性成分，只留纯交互
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()
T_true_c, T_model_c = double_center(T_true), double_center(inter)
r = np.corrcoef(T_model_c.ravel(), T_true_c.ravel())[0, 1]              # 0.825
k = np.linalg.lstsq(T_true_c.ravel()[:,None], T_model_c.ravel(), rcond=None)[0][0]  # 0.575
```

---

## 6. 文件 / Git 状态

- 仓库：`~/projects/friction-ml`（本地，**GitHub 远程未推**，SSH key 未配 → 阶段 4）
- Commit 历史（**已全部提交，无未存工作**）：
  ```
  66abf49  阶段2完成：GBDT调参 + 特征重要性 + PDP机理验收 7/7   ← 07-18
  adc681d  添加项目交接文档
  6fe8594  基线模型：线性回归R²=0.735 vs 随机森林R²=0.627(默认参数过拟合)
  c82e5c1  EDA：目标变量对数正态、成分闭合验证、树脂U型曲线还原(13.25 vs 真值13.0)
  c6d920f  feat: NAO friction material formulation data generator
  ```
- 工作文件：`01_eda.ipynb`（EDA + 基线 + 阶段2 全在里面，约 610 KB）
- 数据：`friction_formulation.csv`、脚本 `gen_friction_data.py`
- Git 三步节奏：`git add <文件>` → `git status`（确认变绿）→ `git commit -m "..."`

---

## 7. 下一步路线图

**阶段 2：模型解释 = 验收机理 — ✅ 已完结（07-18）**

**阶段 3：闭合数据正确处理（学术亮点）— 下一站**
- [ ] 理解「闭合数据为什么需要对数比变换」（概念先行）
- [ ] CLR / ILR 变换实现
- [ ] 与选项 A 对比：预测精度、系数可解释性、重要性排序是否更干净
- [ ] 定位：选项 A 展示「病症」，ILR/CLR 是「药方」

**阶段 4：收尾**
- [ ] log 空间 R² 如何正确解读成磨损率精度（决策 A 遗留）
- [ ] 结论整理 / 论文图表清单
- [ ] 配 SSH key，推 GitHub

---

## 8. 会话命名索引（每次收工时由 Claude 追加，见 §0 约定 2）

- NAO摩擦材料ML-01-字段定义+数据生成（07-14 及以前）
- NAO摩擦材料ML-02-基线模型(EDA收尾+线性vs随机森林)（07-15）
- NAO摩擦材料ML-03-阶段2机理验收(RF/GBDT调参+特征重要性+置换重要性)（07-16）
- **NAO摩擦材料ML-04-PDP验形状+交互项马鞍面定量验收(r=0.825)（07-18，本次，阶段2完结）**
- （下次）NAO摩擦材料ML-05-阶段3成分数据对数比变换(CLR/ILR) …

---

## 9. 下次开场白（建议直接说）

> 「继续这个项目（ML-05）。阶段 2 已完结（7/7 机理找回）。今天开阶段 3：成分数据的对数比变换。先给我讲清楚**闭合数据为什么需要 CLR/ILR**、它到底治好了什么病，我听懂了再动手写代码。」
