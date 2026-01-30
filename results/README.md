# Results 目录

本目录包含模型运行生成的结果文件。

## 文件列表

### Task 1: 投票预估结果

| 文件 | 说明 | 记录数 | 关键字段 |
|------|------|--------|----------|
| `task1_vote_estimates.csv` | 每位选手每周的投票份额估计 | ~3000 | `vote_share_mean`, `vote_share_std`, `ci_lower`, `ci_upper` |
| `task1_consistency_metrics.csv` | 一致性指标 (EPA, Top-2等) | 301周 | `epa`, `elim_prob`, `top2_correct`, `kendall_tau` |
| `task1_certainty_metrics.csv` | 确定性指标 (CV等) | ~3000 | `cv`, `ci_width` |

### Task 2: 计分方法比较结果

| 文件 | 说明 | 记录数 | 关键字段 |
|------|------|--------|----------|
| `task2_comparison_stats.csv` | 两种方法的统计对比 | 34季 | `diff_rate`, `rank_upset_rate`, `pct_upset_rate` |
| `task2_counterfactual.csv` | 反事实分析结果 | 301周 | `actual_elim`, `alt_elim`, `would_change` |
| `task2_weight_analysis.csv` | 隐含权重分析 | 301周 | `judge_weight`, `vote_weight`, `sensitivity` |

## 数据说明

### Task 1 核心指标

- **EPA (Elimination Prediction Accuracy)**: 淘汰预测准确率，V3版本达到 93%
- **Top-2 Accuracy**: 被淘汰者是否在预测的最低两名中，V3版本达到 99%
- **CV (Coefficient of Variation)**: 变异系数，反映估计的不确定性，平均约 0.71

### Task 2 核心指标

- **diff_rate**: 两种方法产生不同淘汰结果的比例
- **upset_rate**: 评委最低分选手未被淘汰的比例（"翻盘率"）

## 生成方式

```bash
# 生成 Task 1 结果
cd scripts && python task1_vote_estimator_v3.py

# 生成 Task 2 结果
cd scripts && python task2_scoring_methods.py
```

---

*最后更新: 2026-01-30*
