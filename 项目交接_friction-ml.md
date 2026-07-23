# 项目交接文档 —— NAO 摩擦材料磨损率回归

> **用途**：下次新开 Claude 会话时，把本文件内容作为第一条消息贴入，即可无缝续接。
> **最后更新**：2026-07-23（**ML-06 代码逐行精读**：块 1 读数+闭合校验完结，comp_cols=19、闭合和 100.00）

---

## 0. 我是谁 / 怎么带我

- 材料人（硕士，纤维增强树脂基复合材料 / 摩擦学）
- Python 和命令行**零基础**，做中学
- 带我的方式：
  - **一次只给一步**，我做完回你再给下一步
  - 每步先说「在干什么、为什么」，再给命令
  - 不要一次抛多个工具（终端 / VS Code / Jupyter 来回跳会懵）
  - 目标是「懂材料 + 能建模」，不是转行做 ML 工程师

### ⏰ 三条固定约定（每次会话都要执行，别等我开口）

**约定 1：阶段性成果做完就提醒我 commit，并把 commit 内容直接写好给我**

- 触发时机：每完成一个有实质产出的小节就提醒一次（拿到一个新结果、跑通一张关键图、确立一个新结论），**不要攒到收工才提**
- 提醒时必须**连同完整命令一起给出**，我直接复制粘贴，不用自己想怎么写：
  - 先提醒我在 Jupyter 里 **⌘S 保存**（Git 存的是硬盘文件，不是内存）
  - 再给 `git add ... ` → `git status` → `git commit -m "..."` 全套
  - commit message 由你起草：**标题一行讲清做了什么，正文列出关键数字**（R²、r、参数等），别写「更新代码」这种废话
- 原因（07-18 的教训）：两天工作堆在同一个 `.ipynb` 里，Git 无法拆成两笔 commit，历史就糊了
- **例外**（07-23 新增）：纯读代码 / 纯讨论的会话不动硬盘文件，`git status` 是干净的，**不需要也不应该 commit**。别为了「今天有产出」硬造一笔空提交

**约定 2：每次收工时，给出本次会话的建议命名**

- 收工的判定：我说「收工 / 今天到这 / 先这样」，或一个阶段告一段落时
- 命名格式固定：`NAO摩擦材料ML-<两位序号>-<本次核心内容+关键数字>（<日期>）`
  - 例：`NAO摩擦材料ML-04-PDP验形状+交互项马鞍面定量验收(r=0.825)（07-18）`
- 要求：**带上本次最硬的那个数字**（R²、r、命中率等），方便日后在知识库里一眼检索到
- 同时把这条追加进本文档 §8「会话命名索引」，并更新「下次开场白」

**约定 3（07-23 新增）：代码精读用「四问模板」**

逐行讲代码时，每一行都按这四问过一遍，缺一不可：

| 问 | 内容 | 为什么要问 |
|---|---|---|
| 1 | 这行**在干什么**（语法层） | 认字 |
| 2 | 这里**为什么需要它**（意图层） | 认句 |
| 3 | **删掉 / 写错会怎样** | ★ 把「看懂」变成「能自己写」 |
| 4 | 对应材料工作里的**哪一步** | 我的定位：懂材料 + 能建模 |

- 粒度：**以「代码块」为单位过，块内逐行讲**。孤立一行读不出意图（例：`np.log(wear_rate)` 单看只是取对数，放回块里才知道是决策 A）
- 我的习惯：**先自己读一遍、说出我的理解，再听讲解**。这个顺序比直接听效率高，Claude 应先点评我的理解（对的确认、错的拧正），再展开

---

## 环境

- MacBook Air M4（主力）+ MacBook Air M1（外出备用），两台均可跑
- conda 环境名：`mat`（包全用 `pip` 装的，**别混用 conda install**）
- 项目目录：`~/projects/friction-ml`
- 已装：jupyterlab 4.6.1 / numpy 2.4.6 / pandas 2.3.3 / scikit-learn 1.9.0 / seaborn 0.13.2 / matplotlib
- **环境可一键复现**：仓库已含 `environment.yml`（从 M4 的 mat 环境 `conda env export` 导出）。新机器上重建只需一条命令：
  ```bash
  conda env create -f environment.yml
  ```
  已在 M1（arm64）实测通过：M1 上 conda 用 **Miniforge（arm64 版）** 安装（Anaconda 官方版对机构有授权限制，Miniforge 干净精简、原生 Apple Silicon）；建好后 Run All 结果与 M4 逐点一致（r=0.825、k=0.575 分毫不差）。
- **跨机说明**：M4 与 M1 均为 Apple Silicon（arm64 同架构），同版本包 + 同 random_state → 结果可完全复现。搬仓走 AirDrop（整个 `friction-ml` 文件夹连 `.git` 一起搬，历史不丢）。
- **开工老三样**：
  ```bash
  conda activate mat
  cd ~/projects/friction-ml && jupyter lab
  ```
  打开 `01_eda.ipynb` → 菜单 Run → Run All Cells（kernel 重启后变量清空，必须重跑）
  - `cd` 那一步**不是可有可无**：全项目用的是相对路径（`pd.read_csv("friction_formulation.csv")`），当前目录不对就 `FileNotFoundError`

> ⚠️ 所有模型对象（best_gb / best_gb_v2 / PDP 结果）都只在 `01_eda.ipynb` 里、靠 Run All 重建。开工第一件事永远是 Run All。
> 例外：**纯读代码的会话不用 Run All**，省几十秒。

### 字号调整（07-23）

| 位置 | 临时 | 永久 |
|---|---|---|
| 终端 | ⌘+ / ⌘− 放大缩小，⌘0 复原 | Terminal → 设置(⌘,) → 描述文件 → 选中在用的 Profile → 文本 → 字体 → 更改（13 寸屏建议 14–16） |
| JupyterLab | — | Settings → Theme → Increase **Code** Font Size（代码区）/ Increase **Content** Font Size（文字说明区） |

两者**互相独立**，改一个不影响另一个。

### 运行耗时基准（07-23，别再疑心装坏了）

| 机器 | Run All 耗时 | 说明 |
|---|---|---|
| M4（插电、热跑） | < 10 s | 之前的记忆值 |
| M1（电池、咖啡馆、冷启动） | ~30 s | **正常** |

差距来源拆解：芯片多核约 1.7–1.8×，其余来自 **低电量模式 / 电池供电**（可差 30–50%）、**冷启动**（首次 import 从硬盘读，5–15 s）。交接文档里「网格搜索几十秒属正常」这句本来就是在 M4 上写的，30 s 完全在水位内。真正的健康证明是 **r=0.825 逐点复现**，不是秒数。

### ⛔ 不要搬到 Windows / 拯救者（07-23 结论）

- **sklearn 全程跑 CPU，GPU 贡献为零**。树模型（RF/GBDT）的分裂是大量条件判断+排序，天生不适合 GPU；GPU 只对深度学习的大矩阵乘法有用
- 拯救者 CPU 多核确实可能快 1.2–1.5×（30 s → 15–20 s），**但代价远大于收益**：
  1. 换 x86 架构 → 从「确定逐点一致」退到「应该一致」，ML-05 刚建立的可复现性打折
  2. PowerShell ≠ zsh，零基础学来的命令行手感要重学
  3. 三台机器 + **GitHub 远程还没配**，靠 AirDrop 同步迟早岔开 Git 历史
- **真想多机协作，正确顺序是先配 SSH key 推 GitHub（阶段 4 待办），而不是先加机器**
- 本项目天花板在**数据信噪比**（200 条 + 18 个噪声特征），不在算力 —— 换机器解决不了信噪比（见 §2 决策 D）

### 已踩过的环境坑（省时间）

| 坑 | 正确做法 |
|---|---|
| `PartialDependenceDisplay.from_estimator` 与 `partial_dependence` 的 features 写法**不一样** | 前者二维交互用**列表套元组** `[("a","b")]`；后者用**平列表** `["a","b"]` |
| NumPy 2.x 删了 `数组.ptp()` | 改用 `np.ptp(数组)` |
| Jupyter 内存 ≠ 硬盘文件 | commit 前必须先在浏览器 **⌘S** 保存，再 `ls -l` 看时间戳确认 |
| 要同时跑 Jupyter 和终端命令 | 终端里 **⌘T** 开新标签页；跑 Jupyter 那个标签**别关别 Ctrl+C** |
| cell 左边 `[23]` 以为是「第 23 个 cell」（07-23 误解，已纠正） | 它是**执行序号**：本次 kernel 启动以来第几次执行，**与位置无关、只增不减**，且会随 ⌘S 一起**存进 .ipynb 文件**。`[ ]`=从没跑过，`[*]`=正在跑，`[n]`=跑完 |
| 序号乱序 / 跳跃（上面 17、下面 5） | 说明跳着跑过，内存里可能是老变量 → **结果不可信**。归零：Run All，或 Kernel → Restart Kernel and Clear Outputs |

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
代码里的落点：`df[comp_cols].sum(axis=1).min()/.max()` → 输出 `100.00 ~ 100.00`（见 §5.5 块 1）。

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
13. **（07-23 新）代码里「准备+验收」的行数远多于「做 ML」的行数，这是正常的**：真正建模就 `.fit()` / `.predict()` 几行，其余全是搬数据、验数据、造清单 —— 和做试验一样，压试片几分钟，配料/称量/制样/复核占绝大部分
14. **（07-23 新）算力不是本项目的瓶颈**：sklearn 全 CPU，GPU 零贡献；换更快的机器省的是秒，损失的是可复现性和手感

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

## 5.5 代码逐行精读进度（ML-06 起开的新支线）

> 目的：把 `01_eda.ipynb` 的每一块代码按 §0 约定 3 的「四问模板」吃透，从「能跑」升级到「能自己写」。
> 顺序按执行顺序走。前四块是**建模主干**，后三块是**解释工具**。

| 块 | 内容 | 状态 |
|---|---|---|
| 1 | 读数 + 体检 + comp_cols 清单 + 闭合校验 | ✅ **07-23 完结** |
| 2 | 切分训练/测试集（X/y 约定、random_state） | ⬜ **下次从这里开始** |
| 3 | GBDT 网格搜索 | ⬜ |
| 4 | binder_total 合体 | ⬜ |
| 5 | 置换重要性 | ⬜ |
| 6 | 1D PDP | ⬜ |
| 7 | 交互残差面 + 定量验收 | ⬜ |

### 块 1 精读结论（07-23）

| 行 | 干什么 | 最该记住的 |
|---|---|---|
| `import pandas as pd` | 搬表格工具箱 | kernel 一重启就没 → 必须 Run All |
| `import numpy as np` | 搬数学工具箱 | 本 cell 没用到，是「import 集中在文件顶部」的习惯 |
| `df = pd.read_csv(...)` | 硬盘 → 内存 | **相对路径 = 项目可搬运**（M4→M1 迁移顺利的功臣之一） |
| `df.shape` | 200×25 | **属性不带括号，方法带括号**；25 = 1+19+4+1 对得上 |
| `df.isna().sum().sum()` | 缺失 0 | 两次 sum：先压成每列、再压成总数。它验的是**读取动作**，不是数据本身 |
| `comp_cols = [...]` | 造 19 列清单 | **黑名单**逻辑（踢掉 6 个），不是白名单 |
| `df[comp_cols].sum(axis=1)` | 闭合校验 100.00 | 清单 ≠ 数据；`axis=1` = 横着加 |
| `df.head()` | 肉眼验一眼 | cell 最后一行自动显示；写在中间就没输出 |

**语法要点（这块学到的通用知识）**

1. **属性 vs 方法**：`df.shape` 无括号（表本来就带的事实，像试样尺寸）；`df.head()` 有括号（要它干一件事，像做一次拉伸试验）。**括号 = 干活的启动键**。方法漏括号**不报错**，只打出 `<bound method ...>` —— 见到这个就是漏括号了
2. **列表推导式** `[收什么 for 谁 in 哪里 if 条件]`，**从中间往外读**。等价于「空篮子 + for + if + append」四行
3. **黑名单 vs 白名单**：黑名单（`not in` 6 个非配方列）打字少、新增组分自动纳入；代价是**新增的非配方列会被误收**
4. **`axis` 的记法**：`axis=1` = **消掉「列」这个维度**（19 列压成 1 个总和，每行一个结果）；`axis=0` = 消掉行。记「消掉谁」比记「沿着谁」不容易反
5. **`df[...]` 括号里放列表 → 结果一定是 DataFrame（二维）**；放单个字符串 → Series（一维）。二维才谈得上 axis
6. **`%.2f` 是显示精度**：生成脚本做过 `round(3)`，真实闭合和可能是 99.999，被 `.2f` 抹成 100.00。想看真相把 `.2f` 改 `.6f`。（类比：报密度 1.85 g/cm³ 不代表仪器分辨到 0.01）
7. **Jupyter 自动显示规则**：cell **最后一行**若产生了值就自动漂亮显示；写在中间静悄悄丢弃；一个 cell 写两个只显示最后那个

**结构性智慧（可迁移到任何项目）**

1. **写相对路径，别写绝对路径** → 项目能整个文件夹搬走。和固定 `random_state` 是同一类「为可复现服务」的操作
2. **互相依赖、必须一起更新的几行，焊在同一个 cell 里**。`read_csv` 与 `comp_cols` 同 cell，所以重跑时先拿回干净的 25 列，后加的 `log_wear` 当场蒸发，绝无可能被黑名单误收成第 20 个配方列 —— 从结构上消灭乱序执行的风险，比靠人记住「要按顺序跑」可靠
3. **读完 / 改完就体检一次**（`shape` + `isna` + `head`）。数字体检查不出编码乱码、列串位、数字被读成字符串，这三样只能靠 `head()` 肉眼看。成本一秒，挡掉一整类隐性错误

**为什么现在是合成数据也要查缺失值**：`read_csv` 有很多**静默出错**的方式（编码不对、分隔符猜错、某行多个逗号），出问题时通常不报错，而是错位成一片 NaN。sklearn 的 `fit()` 遇到 NaN 会抛 `ValueError`，等跑到网格搜索才发现就白等几十分钟。

---

## 6. 文件 / Git 状态

- 仓库：`~/projects/friction-ml`（本地，**GitHub 远程未推**，SSH key 未配 → 阶段 4）
- Commit 历史（**已全部提交，无未存工作**）：
  ```
  9abaa86  chore: 移除 01_eda.ipynb 中运行报错的死代码           ← 07-22
  23a31bc  chore: 添加 environment.yml 环境配方单                ← 07-22
  66abf49  阶段2完成：GBDT调参 + 特征重要性 + PDP机理验收 7/7   ← 07-18
  adc681d  添加项目交接文档
  6fe8594  基线模型：线性回归R²=0.735 vs 随机森林R²=0.627(默认参数过拟合)
  c82e5c1  EDA：目标变量对数正态、成分闭合验证、树脂U型曲线还原(13.25 vs 真值13.0)
  c6d920f  feat: NAO friction material formulation data generator
  ```
  > ML-06（07-23）为纯读代码会话，未产生硬盘改动，**无新增 commit**（本文档更新除外）
- 工作文件：`01_eda.ipynb`（EDA + 基线 + 阶段2 全在里面，约 610 KB）
- 数据：`friction_formulation.csv`、脚本 `gen_friction_data.py`、环境 `environment.yml`
- Git 三步节奏：`git add <文件>` → `git status`（确认变绿）→ `git commit -m "..."`
- **本文档存在三处，注意同步**：① M4 本地仓库 ② M1 本地仓库 ③ **Claude 项目知识库（最关键，靠它续接上下文，需手动替换）**

---

## 7. 下一步路线图

**阶段 2：模型解释 = 验收机理 — ✅ 已完结（07-18）**

**支线：代码逐行精读（ML-06 起，与主线并行，见 §5.5）**
- [x] 块 1 读数 + 闭合校验（07-23）
- [ ] 块 2 切分训练/测试集 ← **下一站**
- [ ] 块 3–7（网格搜索 / binder_total / 置换重要性 / PDP / 交互残差面）

**阶段 3：闭合数据正确处理（学术亮点）— 主线下一站**
- [ ] 理解「闭合数据为什么需要对数比变换」（概念先行）
- [ ] CLR / ILR 变换实现
- [ ] 与选项 A 对比：预测精度、系数可解释性、重要性排序是否更干净
- [ ] 定位：选项 A 展示「病症」，ILR/CLR 是「药方」

**阶段 4：收尾**
- [ ] log 空间 R² 如何正确解读成磨损率精度（决策 A 遗留）
- [ ] 结论整理 / 论文图表清单
- [ ] 配 SSH key，推 GitHub（**多机协作的正解，优先级已上调**）
- [x] ~~环境可复现~~ ✅ 已完成（ML-05）：`environment.yml` 进仓库，新机 `conda env create -f environment.yml` 一键重建；M1 已配 Miniforge、mat 环境已建、Run All 复现通过

---

## 8. 会话命名索引（每次收工时由 Claude 追加，见 §0 约定 2）

- NAO摩擦材料ML-01-字段定义+数据生成（07-14 及以前）
- NAO摩擦材料ML-02-基线模型(EDA收尾+线性vs随机森林)（07-15）
- NAO摩擦材料ML-03-阶段2机理验收(RF/GBDT调参+特征重要性+置换重要性)（07-16）
- NAO摩擦材料ML-04-PDP验形状+交互项马鞍面定量验收(r=0.825)（07-18，阶段2完结）
- NAO摩擦材料ML-05-环境迁移M4→M1(environment.yml一键重建, Run All逐点复现r=0.825)（07-22）
- **NAO摩擦材料ML-06-代码块1逐行精读(读数+闭合校验comp_cols=19)（07-23，本次）**
- （下次）NAO摩擦材料ML-07-…（二选一，见 §9）

---

## 9. 下次开场白（建议直接说，二选一）

**选项 A —— 走主线（阶段 3，学术亮点）**

> 「继续这个项目（ML-07）。阶段 2 已完结（7/7 机理找回），环境已可 `environment.yml` 一键复现，代码精读已过完块 1。今天开阶段 3：成分数据的对数比变换。先给我讲清楚**闭合数据为什么需要 CLR/ILR**、它到底治好了什么病，我听懂了再动手写代码。」

**选项 B —— 走支线（接着精读代码）**

> 「继续这个项目（ML-07）。接着 §5.5 的代码精读，从**块 2 切分训练/测试集**开始，还是老规矩：一次一块、块内逐行、四问模板。我先说我的理解，你再拧。」

**建议**：阶段 3 是论文的学术亮点，优先级更高；代码精读可以在阶段 3 每次开工前当「热身」穿插做一块，两条线不冲突。
