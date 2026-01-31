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
| `task2_counterfactual.csv` | 反事实模拟结果 | 301周 | `actual_eliminated`, `cf_eliminated`, `elimination_flipped`, `rank_tau`, `rank_rho` |
| `task2_comparison_stats.csv` | 分组统计汇总 | 38行 | `group`, `flip_count`, `elimination_flip_rate`, `mean_rank_tau` |
| `task2_weight_analysis.csv` | 有效权重分解 | 301周 | `judge_weight_rank`, `vote_weight_rank`, `judge_weight_pct`, `vote_weight_pct` |
| `task2_upset_rate.csv` | 翻盘率分析 | 301周 | `actual_eliminated`, `judge_lowest`, `is_upset` |

## 数据说明

### Task 1 核心指标

- **EPA (Elimination Prediction Accuracy)**: 淘汰预测准确率，V3版本达到 93%
- **Top-2 Accuracy**: 被淘汰者是否在预测的最低两名中，V3版本达到 99%
- **CV (Coefficient of Variation)**: 变异系数，反映估计的不确定性，平均约 0.71

### Task 2 核心指标

- **EFR (Elimination Flip Rate)**: 淘汰翻转率
  - 整体: 21.9% (66/301)
  - 排名法: 0.0% (0/74)
  - 百分比法: 29.1% (66/227)
- **Effective Weight**: 有效权重
  - 排名法观众权重: 51.8%
  - 百分比法观众权重: 77.1% (93.4%周次 > 50%)
- **Upset Rate**: 翻盘率（评委最低分者未被淘汰）
  - 排名法: 64.9% (48/74)
  - 百分比法: 54.6% (124/227)

## 生成方式

```bash
# 生成 Task 1 结果
cd scripts && python task1_vote_estimator_v3.py

# 生成 Task 2 结果 (按顺序执行)
cd scripts && python task2_scoring_methods.py
cd scripts && python task2_comparison_analysis.py
cd scripts && python task2_weight_decomposition.py
```

---

*最后更新: 2026-01-31*
