"""
Task 2: 比较分析 + 统计指标

职责:
1. 计算淘汰翻转率、排名变化率
2. 计算 Kendall's τ / Spearman's ρ 统计
3. 按季节/整体汇总

输入: task2_counterfactual.csv
输出: task2_comparison_stats.csv

使用方法:
    python task2_comparison_analysis.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import os
import warnings
warnings.filterwarnings('ignore')


def compute_flip_rates(cf_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算淘汰翻转率和相关统计

    按以下维度分组:
    - 整体 (overall)
    - 按实际方法 (rank_method / pct_method)
    - 按季节 (season_X)

    Args:
        cf_df: 反事实模拟结果 DataFrame

    Returns:
        分组统计 DataFrame
    """
    results = []

    # 整体统计
    overall_stats = {
        'group': 'overall',
        'n_weeks': len(cf_df),
        'flip_count': cf_df['elimination_flipped'].sum(),
        'elimination_flip_rate': cf_df['elimination_flipped'].mean(),
        'mean_rank_tau': cf_df['rank_tau'].mean(),
        'std_rank_tau': cf_df['rank_tau'].std(),
        'mean_rank_rho': cf_df['rank_rho'].mean(),
        'std_rank_rho': cf_df['rank_rho'].std(),
        'mean_actual_margin': cf_df['actual_margin'].mean(),
        'mean_cf_margin': cf_df['cf_margin'].mean()
    }
    results.append(overall_stats)

    # 按实际方法分组
    for method in ['rank', 'percent']:
        method_df = cf_df[cf_df['actual_method'] == method]
        if len(method_df) == 0:
            continue

        method_stats = {
            'group': f'{method}_method',
            'n_weeks': len(method_df),
            'flip_count': method_df['elimination_flipped'].sum(),
            'elimination_flip_rate': method_df['elimination_flipped'].mean(),
            'mean_rank_tau': method_df['rank_tau'].mean(),
            'std_rank_tau': method_df['rank_tau'].std(),
            'mean_rank_rho': method_df['rank_rho'].mean(),
            'std_rank_rho': method_df['rank_rho'].std(),
            'mean_actual_margin': method_df['actual_margin'].mean(),
            'mean_cf_margin': method_df['cf_margin'].mean()
        }
        results.append(method_stats)

    # 按季节分组
    for season in sorted(cf_df['season'].unique()):
        season_df = cf_df[cf_df['season'] == season]
        if len(season_df) == 0:
            continue

        season_stats = {
            'group': f'season_{season}',
            'n_weeks': len(season_df),
            'flip_count': season_df['elimination_flipped'].sum(),
            'elimination_flip_rate': season_df['elimination_flipped'].mean(),
            'mean_rank_tau': season_df['rank_tau'].mean(),
            'std_rank_tau': season_df['rank_tau'].std(),
            'mean_rank_rho': season_df['rank_rho'].mean(),
            'std_rank_rho': season_df['rank_rho'].std(),
            'mean_actual_margin': season_df['actual_margin'].mean(),
            'mean_cf_margin': season_df['cf_margin'].mean()
        }
        results.append(season_stats)

    return pd.DataFrame(results)


def analyze_flip_patterns(cf_df: pd.DataFrame) -> Dict:
    """
    分析翻转模式

    Args:
        cf_df: 反事实模拟结果

    Returns:
        翻转模式分析结果
    """
    flipped = cf_df[cf_df['elimination_flipped']]

    if len(flipped) == 0:
        return {
            'total_flips': 0,
            'flip_rate': 0.0,
            'patterns': []
        }

    # 分析翻转发生的条件
    patterns = {
        'total_flips': len(flipped),
        'flip_rate': len(flipped) / len(cf_df),
        'by_method': {},
        'by_n_contestants': {},
        'margin_analysis': {}
    }

    # 按方法统计
    for method in ['rank', 'percent']:
        method_flips = flipped[flipped['actual_method'] == method]
        method_total = cf_df[cf_df['actual_method'] == method]
        if len(method_total) > 0:
            patterns['by_method'][method] = {
                'flip_count': len(method_flips),
                'flip_rate': len(method_flips) / len(method_total)
            }

    # 按选手数量统计
    for n in sorted(cf_df['n_contestants'].unique()):
        n_flips = flipped[flipped['n_contestants'] == n]
        n_total = cf_df[cf_df['n_contestants'] == n]
        if len(n_total) > 0:
            patterns['by_n_contestants'][n] = {
                'flip_count': len(n_flips),
                'flip_rate': len(n_flips) / len(n_total)
            }

    # 边距分析
    patterns['margin_analysis'] = {
        'flipped_mean_margin': flipped['actual_margin'].mean(),
        'non_flipped_mean_margin': cf_df[~cf_df['elimination_flipped']]['actual_margin'].mean(),
        'flipped_cf_mean_margin': flipped['cf_margin'].mean()
    }

    return patterns


def compute_correlation_by_week(cf_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每周的排名相关性详情

    Args:
        cf_df: 反事实模拟结果

    Returns:
        每周相关性 DataFrame
    """
    records = []

    for _, row in cf_df.iterrows():
        record = {
            'season': row['season'],
            'week': row['week'],
            'n_contestants': row['n_contestants'],
            'actual_method': row['actual_method'],
            'rank_tau': row['rank_tau'],
            'rank_rho': row['rank_rho'],
            'elimination_flipped': row['elimination_flipped'],
            'actual_margin': row['actual_margin'],
            'cf_margin': row['cf_margin'],
            # 相关性等级
            'correlation_level': categorize_correlation(row['rank_tau'])
        }
        records.append(record)

    return pd.DataFrame(records)


def categorize_correlation(tau: float) -> str:
    """将相关系数分类"""
    if tau >= 0.9:
        return 'very_high'
    elif tau >= 0.7:
        return 'high'
    elif tau >= 0.5:
        return 'moderate'
    elif tau >= 0.3:
        return 'low'
    else:
        return 'very_low'


def hypothesis_tests(cf_df: pd.DataFrame) -> Dict:
    """
    进行假设检验

    H1: 排名法和百分比法的翻转率无显著差异
    H2: 翻转与边距有相关性

    Args:
        cf_df: 反事实模拟结果

    Returns:
        假设检验结果
    """
    results = {}

    # H1: 比较两种方法的翻转率 (卡方检验)
    rank_df = cf_df[cf_df['actual_method'] == 'rank']
    pct_df = cf_df[cf_df['actual_method'] == 'percent']

    if len(rank_df) > 0 and len(pct_df) > 0:
        contingency = [
            [rank_df['elimination_flipped'].sum(),
             len(rank_df) - rank_df['elimination_flipped'].sum()],
            [pct_df['elimination_flipped'].sum(),
             len(pct_df) - pct_df['elimination_flipped'].sum()]
        ]

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            results['method_flip_rate_test'] = {
                'test': 'chi2',
                'statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': '两种方法的翻转率有显著差异' if p_value < 0.05 else '两种方法的翻转率无显著差异'
            }
        except:
            results['method_flip_rate_test'] = {'error': '样本量不足'}

    # H2: 边距与翻转的相关性 (点二列相关)
    margin = cf_df['actual_margin'].values
    flipped = cf_df['elimination_flipped'].astype(int).values

    try:
        corr, p_value = stats.pointbiserialr(flipped, margin)
        results['margin_flip_correlation'] = {
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': '边距小时更容易翻转' if corr < 0 else '边距大时更容易翻转'
        }
    except:
        results['margin_flip_correlation'] = {'error': '计算失败'}

    return results


def generate_summary_report(cf_df: pd.DataFrame) -> str:
    """
    生成文本摘要报告

    Args:
        cf_df: 反事实模拟结果

    Returns:
        报告文本
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Task 2: 计分方法比较分析报告")
    lines.append("=" * 60)

    # 基本统计
    lines.append("\n## 1. 基本统计")
    lines.append(f"总分析周数: {len(cf_df)}")
    lines.append(f"排名法周数: {len(cf_df[cf_df['actual_method'] == 'rank'])}")
    lines.append(f"百分比法周数: {len(cf_df[cf_df['actual_method'] == 'percent'])}")

    # 翻转率
    lines.append("\n## 2. 淘汰翻转率")
    overall_flip = cf_df['elimination_flipped'].mean()
    lines.append(f"整体翻转率: {overall_flip:.1%} ({cf_df['elimination_flipped'].sum()}/{len(cf_df)})")

    for method in ['rank', 'percent']:
        method_df = cf_df[cf_df['actual_method'] == method]
        if len(method_df) > 0:
            flip_rate = method_df['elimination_flipped'].mean()
            lines.append(f"  {method}方法: {flip_rate:.1%} ({method_df['elimination_flipped'].sum()}/{len(method_df)})")

    # 相关性统计
    lines.append("\n## 3. 排名相关性")
    lines.append(f"平均 Kendall's τ: {cf_df['rank_tau'].mean():.3f} (±{cf_df['rank_tau'].std():.3f})")
    lines.append(f"平均 Spearman's ρ: {cf_df['rank_rho'].mean():.3f} (±{cf_df['rank_rho'].std():.3f})")

    # 相关性分布
    high_corr = (cf_df['rank_tau'] >= 0.7).sum()
    low_corr = (cf_df['rank_tau'] < 0.5).sum()
    lines.append(f"高相关 (τ≥0.7): {high_corr}/{len(cf_df)} ({high_corr/len(cf_df):.1%})")
    lines.append(f"低相关 (τ<0.5): {low_corr}/{len(cf_df)} ({low_corr/len(cf_df):.1%})")

    # 边距分析
    lines.append("\n## 4. 边距分析")
    flipped = cf_df[cf_df['elimination_flipped']]
    non_flipped = cf_df[~cf_df['elimination_flipped']]
    lines.append(f"翻转周平均边距: {flipped['actual_margin'].mean():.4f}" if len(flipped) > 0 else "翻转周平均边距: N/A")
    lines.append(f"未翻转周平均边距: {non_flipped['actual_margin'].mean():.4f}" if len(non_flipped) > 0 else "")

    return "\n".join(lines)


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 输入路径
    cf_path = os.path.join(script_dir, 'task2_counterfactual.csv')

    # 检查文件存在
    if not os.path.exists(cf_path):
        print(f"错误：反事实模拟文件不存在: {cf_path}")
        print("请先运行 task2_scoring_methods.py")
        return

    # 加载数据
    cf_df = pd.read_csv(cf_path, encoding='utf-8-sig')
    print(f"加载数据: {len(cf_df)} 周")

    # 计算分组统计
    print("\n计算分组统计...")
    stats_df = compute_flip_rates(cf_df)

    # 分析翻转模式
    print("分析翻转模式...")
    patterns = analyze_flip_patterns(cf_df)

    # 假设检验
    print("进行假设检验...")
    tests = hypothesis_tests(cf_df)

    # 生成报告
    report = generate_summary_report(cf_df)
    print(report)

    # 保存统计结果（先保存再打印，避免编码问题导致丢失数据）
    output_path = os.path.join(script_dir, 'task2_comparison_stats.csv')
    stats_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n统计结果已保存: {output_path}")

    # 打印假设检验结果
    print("\n## 5. 假设检验")
    if 'method_flip_rate_test' in tests:
        test = tests['method_flip_rate_test']
        if 'error' not in test:
            print(f"方法翻转率差异检验: chi2={test['statistic']:.3f}, p={test['p_value']:.4f}")
            print(f"  -> {test['interpretation']}")

    if 'margin_flip_correlation' in tests:
        test = tests['margin_flip_correlation']
        if 'error' not in test:
            print(f"边距-翻转相关性: r={test['correlation']:.3f}, p={test['p_value']:.4f}")
            print(f"  -> {test['interpretation']}")

    # 打印翻转模式
    print("\n## 6. 翻转模式")
    print(f"按选手数量:")
    for n, data in patterns.get('by_n_contestants', {}).items():
        print(f"  {n}人: 翻转率 {data['flip_rate']:.1%} ({data['flip_count']}次)")

    # 打印关键发现
    print("\n" + "=" * 60)
    print("关键发现")
    print("=" * 60)

    overall_flip = cf_df['elimination_flipped'].mean()
    mean_tau = cf_df['rank_tau'].mean()

    findings = []

    if overall_flip < 0.15:
        findings.append(f"1. 两种方法高度一致: 仅 {overall_flip:.1%} 的周产生不同淘汰结果")
    elif overall_flip < 0.25:
        findings.append(f"1. 两种方法较为一致: {overall_flip:.1%} 的周产生不同淘汰结果")
    else:
        findings.append(f"1. 两种方法差异明显: {overall_flip:.1%} 的周产生不同淘汰结果")

    if mean_tau > 0.8:
        findings.append(f"2. 排名相关性很高 (τ={mean_tau:.3f}): 两种方法产生的排名高度一致")
    elif mean_tau > 0.6:
        findings.append(f"2. 排名相关性中等 (τ={mean_tau:.3f}): 两种方法产生的排名较为一致")
    else:
        findings.append(f"2. 排名相关性较低 (τ={mean_tau:.3f}): 两种方法产生的排名差异较大")

    # 边距分析
    flipped = cf_df[cf_df['elimination_flipped']]
    non_flipped = cf_df[~cf_df['elimination_flipped']]
    if len(flipped) > 0 and len(non_flipped) > 0:
        flipped_margin = flipped['actual_margin'].mean()
        non_flipped_margin = non_flipped['actual_margin'].mean()
        if flipped_margin < non_flipped_margin:
            findings.append(f"3. 边界情况更易翻转: 翻转周的平均边距 ({flipped_margin:.4f}) < 未翻转周 ({non_flipped_margin:.4f})")

    for f in findings:
        print(f)


if __name__ == '__main__':
    main()
