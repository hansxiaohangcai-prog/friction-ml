# -*- coding: utf-8 -*-
"""
生成模拟数据：NAO（无石棉有机型）纤维增强树脂基摩擦材料
配方(wt%) + 工艺参数  ->  磨损率 wear_rate [1e-7 cm^3/(N·m)]

设计要点：
1. 配方是成分闭合数据：所有配方组分之和恒为 100 wt%
2. 真值中埋入了非线性 + 交互项，随机森林应当能找回，线性模型应当吃亏
3. 加入 ~10% 相对测量噪声，模拟真实摩擦试验的分散性
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 200

# ---------- 1. 组块定义 ----------
# 每个组块: (组块总量范围 wt%, [组分名], Dirichlet 浓度参数)
BLOCKS = {
    "binder":    ((8, 18),  ["phenolic", "phenolic_modified"],                 [2.0, 1.2]),
    "fiber":     ((10, 22), ["aramid", "ceramic_fiber", "mineral_fiber",
                             "cellulose_fiber"],                               [1.0, 1.5, 2.0, 1.2]),
    "lubricant": ((8, 18),  ["graphite", "MoS2", "Sb2S3", "coke"],             [3.0, 1.0, 1.0, 1.5]),
    "abrasive":  ((2, 8),   ["Al2O3", "zircon", "chromite"],                   [1.5, 1.2, 1.0]),
    "filler":    ((28, 45), ["BaSO4", "CaCO3", "mica", "vermiculite"],         [4.0, 2.0, 1.5, 1.0]),
    "organic":   ((5, 14),  ["NBR_powder", "friction_dust"],                   [1.5, 2.0]),
}

rows = []
for _ in range(N):
    rec = {}
    for _, (rng_tot, names, alpha) in BLOCKS.items():
        total = RNG.uniform(*rng_tot)
        frac = RNG.dirichlet(alpha)
        for name, f in zip(names, frac):
            rec[name] = total * f
    rows.append(rec)

df = pd.DataFrame(rows)
comp_cols = list(df.columns)

# ---------- 2. 闭合归一化：配方总和 = 100 wt% ----------
df[comp_cols] = df[comp_cols].div(df[comp_cols].sum(axis=1), axis=0) * 100.0

# ---------- 3. 工艺参数（独立于配方） ----------
df["press_temp_C"]     = RNG.uniform(150, 175, N)
df["press_pressure_MPa"] = RNG.uniform(10, 25, N)
df["press_time_min"]   = RNG.uniform(6, 15, N)
df["post_cure_temp_C"] = RNG.uniform(160, 220, N)

# ---------- 4. 真值机理（埋进去的“物理”） ----------
binder = df["phenolic"] + df["phenolic_modified"]

# (a) 树脂含量：非单调。过低粘结不足，过高富树脂层热降解 -> 最优约 13 wt%
f_binder = 0.030 * (binder - 13.0) ** 2

# (b) 芳纶纤维：强减磨，但边际递减（饱和）
f_aramid = -0.55 * np.log1p(df["aramid"])

# (c) 石墨：润滑膜，饱和型减磨
f_graphite = -0.40 * np.log1p(df["graphite"])

# (d) 氧化铝磨料：超线性增磨（对偶件也伤，但这里只算自身磨损）
f_abr = 0.020 * df["Al2O3"] ** 2

# (e) 腰果壳粉：软质有机相，热稳定性差 -> 增磨
f_dust = 0.045 * df["friction_dust"]

# (f) 后固化温度：交联更完全 -> 减磨，饱和
f_pc = -0.008 * (df["post_cure_temp_C"] - 160.0)

# (g) 交互项：芳纶必须有足够树脂锚固才能发挥作用
f_inter = -0.010 * df["aramid"] * (binder - 10.0)

log_wear = (0.55 + f_binder + f_aramid + f_graphite
            + f_abr + f_dust + f_pc + f_inter)

wear = np.exp(log_wear)
wear *= RNG.lognormal(0.0, 0.10, N)      # ~10% 相对测量噪声

df["wear_rate"] = wear

# ---------- 5. 输出 ----------
df.insert(0, "batch_id", [f"F{i:03d}" for i in range(1, N + 1)])
df = df.round(3)
df.to_csv("friction_formulation.csv", index=False, encoding="utf-8-sig")

print(f"已生成 {len(df)} 条  ->  friction_formulation.csv")
print(f"配方列 {len(comp_cols)} 个，闭合校验 sum = "
      f"{df[comp_cols].sum(axis=1).min():.2f} ~ {df[comp_cols].sum(axis=1).max():.2f} wt%")
print("\n磨损率分布 [1e-7 cm^3/(N·m)]:")
print(df["wear_rate"].describe().round(3).to_string())
print("\n前 3 行预览:")
print(df.head(3).to_string())
