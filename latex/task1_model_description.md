# Task 1: 投票预估模型

## 1. 问题背景

在《与星共舞》(Dancing with the Stars) 节目中，选手的去留由**评委评分**和**观众投票**共同决定。然而，节目从未公开具体的观众投票数据，我们只能观察到：

- 每周每位选手的评委总分 $S_i$
- 最终被淘汰的选手

**核心问题**：如何根据已知的评委分数和淘汰结果，反推估计每位选手获得的观众投票？

## 2. 计分规则

节目在不同季使用了两种计分方法：

### 2.1 排名法 (Season 1-2, 28-34)

将评委分数和观众投票分别转换为排名，然后相加：

$$
\text{Combined}_i = \text{Rank}_i^{\text{Judge}} + \text{Rank}_i^{\text{Vote}}
$$

**淘汰规则**：综合排名值最大（即排名最靠后）的选手被淘汰。

### 2.2 百分比法 (Season 3-27)

将评委分数和观众投票分别转换为百分比，然后相加：

$$
\text{Combined}_i = \frac{S_i}{\sum_j S_j} + \frac{V_i}{\sum_j V_j}
$$

**淘汰规则**：综合百分比最低的选手被淘汰。

---

## 3. 模型演进历史

### 版本概览

| 版本 | 文件 | 主要特点 | 问题 |
|------|------|----------|------|
| V1 | `task1_vote_estimator.py` | 硬约束MCMC | 一致性100%（循环论证） |
| V2 | `task1_vote_estimator_v2.py` | 软约束MCMC | 接受率低(6.39%) |
| **V3** | `task1_vote_estimator_v3.py` | 自适应MCMC + 特征先验 | **当前版本** |

---

## 4. V1: 硬约束模型（已废弃）

### 4.1 方法

使用硬约束，只接受满足淘汰结果的样本：

```python
if check_elimination(proposal, judge_scores, eliminated_idx):
    # 只有满足约束才考虑接受
    ...
```

### 4.2 问题

- **循环论证**：采样时强制满足约束 → 验证时必然100%一致
- **无法评估模型的真正预测能力**
- **可能低估不确定性**

### 4.3 结果

| 指标 | 值 | 评价 |
|------|-----|------|
| EPA | 100% | ⚠️ 无意义（循环论证） |

---

## 5. V2: 软约束模型

### 5.1 核心改进

使用Softmax软约束替代硬约束：

$$
P(\text{淘汰}_k | V, S) = \frac{\exp(-\tau \cdot C_k)}{\sum_j \exp(-\tau \cdot C_j)}
$$

其中 $\tau$ 是温度参数，$C_i$ 是选手 $i$ 的综合得分。

### 5.2 似然函数

```python
def log_likelihood(votes, judge_scores, eliminated_idx, temperature):
    combined = compute_combined_scores(votes, judge_scores)
    log_probs = -temperature * combined
    probs = softmax(log_probs)
    return np.log(probs[eliminated_idx])
```

### 5.3 实验结果 (全部34季)

| 指标 | 值 | 评价 |
|------|-----|------|
| 总周数 | 301 | - |
| EPA | 84.1% | 合理 |
| 平均淘汰概率 | 0.486 ± 0.306 | 偏低 |
| Top-2准确率 | 90.4% | 良好 |
| Kendall's τ | 0.786 | 良好 |
| **MCMC接受率** | **6.39%** | **⚠️ 过低** |

### 5.4 问题分析

1. **MCMC接受率过低 (6.39%)**
   - 理想值应为 20%-40%
   - 导致采样效率低，后验探索不充分

2. **先验假设过于简单**
   - 仅假设投票与评委分数正相关
   - 未考虑"粉丝效应"（评委分数低但投票高的情况）

3. **未利用选手特征信息**
   - CSV中有职业、年龄等信息但未使用

---

## 6. V3: 改进版模型（当前版本）

### 6.1 改进点总结

| 改进 | V2 | V3 |
|------|----|----|
| MCMC步长 | 固定 | 自适应 |
| 先验信息 | 仅评委分数 | 评委分数 + 职业 + 年龄 |
| 温度参数 | 固定 | 动态（边界敏感） |
| 初始化 | 基于评委分数 | 智能初始化 |

### 6.2 自适应MCMC

根据历史接受率动态调整提议分布的步长：

```python
def adaptive_proposal(current, scale, accept_history, target_rate=0.30):
    recent_rate = np.mean(accept_history[-100:])

    if recent_rate < target_rate - 0.05:
        scale *= 0.95  # 接受率太低，减小步长
    elif recent_rate > target_rate + 0.05:
        scale *= 1.05  # 接受率太高，增大步长

    scale = np.clip(scale, 0.01, 0.3)
    return proposal, scale
```

**目标**：将接受率稳定在 25%-35%

### 6.3 特征增强先验

利用选手特征建立更合理的先验分布：

#### 职业人气权重

| 职业类别 | 权重 | 理由 |
|----------|------|------|
| Singer/Rapper | 1.4 | 粉丝群体最大 |
| Actor/Actress | 1.3 | 知名度高 |
| Athlete | 1.2 | 有粉丝基础 |
| TV Personality | 1.1 | 电视曝光 |
| Model | 1.0 | 基准 |
| News Anchor | 0.9 | 粉丝较少 |

#### 年龄效应

$$
\text{age\_factor} = 1.0 - 0.01 \times |age - 30|
$$

假设25-35岁选手更受观众欢迎（社交媒体活跃度高）。

#### 混合先验

$$
\text{prior}_i = (1-\beta) \cdot \text{feature\_weight}_i + \beta \cdot \text{judge\_norm}_i
$$

其中 $\beta$ 控制评委分数的影响程度。

### 6.4 动态温度似然

边界越清晰，约束越硬：

```python
margin = combined[second_lowest] - combined[lowest]
effective_temp = temperature * (1 + 2 * margin)
```

**效果**：对明确的淘汰更有信心，对边缘情况更谨慎。

### 6.5 智能初始化

降低淘汰者的初始份额，加速收敛：

```python
initial[eliminated_idx] *= 0.7
current = initial / initial.sum()
```

### 6.6 超参数

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 温度 | $\tau$ | 12.0 | 软约束强度 |
| 先验强度 | $\alpha$ | 1.5 | Dirichlet浓度 |
| 先验关联 | $\beta$ | 0.3 | 评委分数权重 |
| 初始步长 | $\lambda_0$ | 0.06 | 提议分布尺度 |
| 采样数 | $N$ | 10000 | MCMC样本量 |
| 预热期 | $B$ | 3000 | 丢弃的初始样本 |

### 6.7 实验结果

#### 一致性指标 (Q1a)

| 指标 | V2 | V3预期 | **V3实际** | 评价 |
|------|-----|--------|------------|------|
| 总周数 | 301 | - | **301** | - |
| EPA | 84.1% | 86-90% | **93.0%** | 超出预期 |
| 淘汰概率 | 0.486 ± 0.306 | 0.55-0.65 | **0.533 ± 0.284** | 接近预期 |
| Top-2准确率 | 90.4% | 92-95% | **99.0%** | 显著提升 |
| Kendall's τ | 0.786 | - | **0.725** | 良好 |
| MCMC接受率 | 6.39% | 25-35% | **28.0%** | 达到目标 |

#### 确定性指标 (Q1b)

| 指标 | V3实际 | 说明 |
|------|--------|------|
| 平均CV | 0.7096 | 不确定性适中 |
| 被淘汰选手CV | 0.7624 | 不确定性略高 |
| 未淘汰选手CV | 0.7028 | 估计更稳定 |

### 6.8 V3 效果分析

#### 主要改进

1. **EPA 从 84.1% 提升到 93.0%**（+8.9个百分点）
   - 自适应MCMC提高了采样效率
   - 特征增强先验捕捉了更多投票模式

2. **MCMC接受率从 6.39% 提升到 28.0%**（+21.6个百分点）
   - 自适应步长调整发挥作用
   - 后验分布探索更充分

3. **Top-2准确率从 90.4% 提升到 99.0%**（+8.6个百分点）
   - 几乎所有被淘汰选手都在模型预测的最低两名中
   - 说明模型很好地识别了"危险区"选手

#### 待改进点

1. **淘汰概率 0.533 略低于预期 0.55-0.65**
   - 模型对淘汰者的识别概率可进一步提高
   - 可考虑增大温度参数 $\tau$

2. **Kendall's τ 从 0.786 降至 0.725**
   - 整体排名相关性略有下降
   - 可能是特征先验引入的偏差
   - 但EPA和Top-2的提升更有实际意义

3. **CV 较高（约 0.71）**
   - 投票份额估计仍有较大不确定性
   - 这是反推问题的固有挑战（不可识别性）

#### 结论

V3 版本在关键指标上取得显著提升：
- **一致性**：EPA 93%，Top-2 99%，模型能准确识别淘汰风险
- **采样效率**：接受率 28%，符合最优区间
- **不确定性量化**：CV ~0.71，合理反映了问题的固有模糊性

---

## 7. 评估指标

### 7.1 一致性指标 (Q1a)

| 指标 | 公式 | 说明 |
|------|------|------|
| EPA | $\frac{\text{正确预测周数}}{\text{总周数}}$ | 淘汰预测准确率 |
| 淘汰概率 | $P(\text{elim} = k \| V, S)$ | 模型对淘汰者的识别概率 |
| Kendall's τ | $\tau(\text{pred\_rank}, \text{actual\_rank})$ | 排名相关性 |
| Top-2准确率 | $P(\text{elim} \in \text{bottom 2})$ | 宽松预测准确率 |
| 淘汰边界 | $C_{\text{second}} - C_{\text{lowest}}$ | 淘汰确信度 |

### 7.2 确定性指标 (Q1b)

| 指标 | 公式 | 说明 |
|------|------|------|
| 后验标准差 | $\sigma_{\pi_i}$ | 投票份额的不确定性 |
| 变异系数 | $CV_i = \sigma_i / \mu_i$ | 标准化不确定性 |
| 95% CI宽度 | $\pi_i^{97.5\%} - \pi_i^{2.5\%}$ | 置信区间宽度 |
| ESS | 有效样本量 | MCMC采样质量 |

---

## 8. 模型输出

对于每周每位选手，模型输出：

| 字段 | 说明 |
|------|------|
| `season`, `week`, `contestant` | 标识信息 |
| `judge_score` | 评委总分 |
| `vote_share_mean` | 投票份额均值 |
| `vote_share_std` | 投票份额标准差 |
| `ci_lower`, `ci_upper` | 95%置信区间 |
| `is_eliminated` | 是否被淘汰 |

输出文件：`vote_estimates_v3.csv`

---

## 9. 使用方法

```bash
# V3版本（推荐）
python scripts/task1_vote_estimator_v3.py

# 指定赛季
python scripts/task1_vote_estimator_v3.py --seasons 1 2 3 4 5

# 调整参数
python scripts/task1_vote_estimator_v3.py --temperature 15 --n-samples 15000

# 不使用选手特征（对比实验）
python scripts/task1_vote_estimator_v3.py --no-features
```

---

## 10. 敏感性分析

使用 `task1_sensitivity_analysis.py` 分析超参数敏感性：

```bash
python scripts/task1_sensitivity_analysis.py --seasons 1 2 3 4 5
```

输出：
- `sensitivity_analysis.csv`: 各参数各取值的详细结果
- `stability_summary.csv`: 稳定性汇总
- `sensitivity_curves.png`: 敏感性曲线图

---

## 11. 符号表

| 符号 | 含义 |
|------|------|
| $n$ | 当周参赛选手数 |
| $S_i$ | 选手 $i$ 的评委总分 |
| $V_i$ | 选手 $i$ 的投票数 |
| $\pi_i = V_i / \sum_j V_j$ | 选手 $i$ 的投票份额 |
| $C_i$ | 选手 $i$ 的综合得分 |
| $\tau$ | Softmax温度参数 |
| $\alpha$ | Dirichlet先验强度 |
| $\beta$ | 评委分数先验权重 |
| $\lambda$ | 提议分布步长 |
| $N$ | MCMC采样数量 |
| $B$ | MCMC预热期 |

---

## 12. 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-01-30 | V1 | 初始版本，硬约束MCMC |
| 2026-01-30 | V2 | 软约束MCMC，添加一致性/确定性指标 |
| 2026-01-30 | V3 | 自适应MCMC，特征增强先验，动态温度 |
| 2026-01-30 | V3结果 | EPA 93%，Top-2 99%，接受率 28% |

---

## 13. 待改进方向

1. **时序信息**：利用同一选手在不同周的投票一致性
2. **分层模型**：建模选手的"基础人气"，在各周间共享
3. **外部数据**：引入Google Trends等外部人气指标
4. **舞伴效应**：分析专业舞伴对投票的影响
