# Scripts 目录

本目录包含 MCM Problem C 的数据分析脚本。

## 文件列表

### Task 1: 投票预估模型

| 文件 | 说明 | 输出 |
|------|------|------|
| `task1_vote_estimator_v3.py` | 贝叶斯MCMC投票估计器 (V3版本) | `results/task1_vote_estimates.csv`, `results/task1_consistency_metrics.csv`, `results/task1_certainty_metrics.csv` |
| `task1_sensitivity_analysis.py` | 超参数敏感性分析 | 敏感性报告 |

### Task 2: 计分方法比较

| 文件 | 说明 | 输出 |
|------|------|------|
| `task2_scoring_methods.py` | 反事实模拟：对每周应用两种计分方法 | `results/task2_counterfactual.csv`, `results/task2_upset_rate.csv` |
| `task2_comparison_analysis.py` | 分组统计：翻转率、相关系数汇总 | `results/task2_comparison_stats.csv` |
| `task2_weight_decomposition.py` | 权重分解：有效权重、边际影响分析 | `results/task2_weight_analysis.csv` |

### Task 3: 争议选手分析

分析评委评分与观众投票存在显著差异的选手，探讨不同评分机制对其命运的影响。

| 文件 | 说明 |
|------|------|
| `task3_controversy_analysis.py` | 分析4位争议选手的每周表现、反事实模拟、评委机制影响 |

**输出文件及用途:**
| 输出文件 | 解决的问题 |
|----------|------------|
| `task3_controversy_metrics.csv` | 量化争议程度：评委与观众的评价差异有多大？ |
| `task3_weekly_performance.csv` | 追踪争议轨迹：选手每周表现如何演变？ |
| `task3_counterfactual_detail.csv` | 评分方法影响：换方法后淘汰结果会改变吗？ |
| `task3_judge_mechanism.csv` | 评委决定权：bottom-2 机制下评委会淘汰谁？ |
| `task3_controversy_summary.csv` | 综合结论：选手被拯救几次？命运如何改变？ |

## 使用方法

```bash
# Task 1: 运行投票估计
python task1_vote_estimator_v3.py

# Task 1: 敏感性分析
python task1_sensitivity_analysis.py --seasons 1 2 3 4 5

# Task 2: 计分方法比较 (按顺序执行)
python task2_scoring_methods.py          # 1. 反事实模拟
python task2_comparison_analysis.py      # 2. 分组统计
python task2_weight_decomposition.py     # 3. 权重分解

# Task 3: 争议选手分析
python task3_controversy_analysis.py     # 生成争议选手相关5个CSV文件
```

## 依赖

- Python 3.8+
- numpy
- pandas
- scipy

---

*最后更新: 2026-01-31*
