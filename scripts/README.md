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
```

## 依赖

- Python 3.8+
- numpy
- pandas
- scipy

---

*最后更新: 2026-01-31*
