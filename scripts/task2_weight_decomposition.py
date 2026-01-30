"""
Task 2: 权重分解 + 边际影响分析

职责:
1. 计算两种方法下评委 vs 观众的有效权重
2. 分析投票变化的边际影响
3. 回归分析权重与分数分布的关系

输入:
- 2026_MCM_Problem_C_Data.csv: 原始评委分数
- vote_estimates_v3.csv: 估计投票份额

输出: task2_weight_analysis.csv

使用方法:
    python task2_weight_decomposition.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')


def compute_effective_weight_rank(judge_scores: np.ndarray,
                                   votes: np.ndarray) -> Dict:
    """
    计算排名法下评委的有效权重

    在排名法中，权重固定为 0.5，因为评委排名和投票排名各贡献一半。
    但我们可以计算"实际影响权重"：谁决定了最终淘汰？

    Args:
        judge_scores: 评委分数
        votes: 投票份额

    Returns:
        权重分析结果
    """
    n = len(judge_scores)

    # 计算排名
    judge_ranks = stats.rankdata(-judge_scores, method='average')
    vote_ranks = stats.rankdata(-votes, method='average')

    # 综合排名值
    combined = judge_ranks + vote_ranks

    # 理论权重 (固定)
    theoretical_weight = 0.5

    # 计算排名的方差贡献
    # 标准化处理
    judge_ranks_std = (judge_ranks - judge_ranks.mean()) / (judge_ranks.std() + 1e-10)
    vote_ranks_std = (vote_ranks - vote_ranks.mean()) / (vote_ranks.std() + 1e-10)
    combined_std = (combined - combined.mean()) / (combined.std() + 1e-10)

    # 回归分析：combined = α * judge + β * vote
    # 由于是简单加法，理论上 α = β = 1
    # 但我们可以看相关性
    judge_corr = np.corrcoef(judge_ranks, combined)[0, 1]
    vote_corr = np.corrcoef(vote_ranks, combined)[0, 1]

    # 相对权重 (基于相关性)
    total_corr = abs(judge_corr) + abs(vote_corr) + 1e-10
    judge_relative_weight = abs(judge_corr) / total_corr

    return {
        'method': 'rank',
        'theoretical_weight': theoretical_weight,
        'judge_corr': judge_corr,
        'vote_corr': vote_corr,
        'judge_relative_weight': judge_relative_weight,
        'vote_relative_weight': 1 - judge_relative_weight,
        'judge_rank_var': judge_ranks.var(),
        'vote_rank_var': vote_ranks.var()
    }


def compute_effective_weight_pct(judge_scores: np.ndarray,
                                  votes: np.ndarray) -> Dict:
    """
    计算百分比法下评委的有效权重

    在百分比法中，权重取决于分数分布的离散程度：
    - 评委分数越分散，评委权重越大
    - 投票越分散，投票权重越大

    Args:
        judge_scores: 评委分数
        votes: 投票份额

    Returns:
        权重分析结果
    """
    n = len(judge_scores)

    # 计算百分比
    judge_pct = judge_scores / (judge_scores.sum() + 1e-10)
    vote_pct = votes / (votes.sum() + 1e-10)

    # 综合百分比
    combined = judge_pct + vote_pct

    # 计算方差
    judge_var = judge_pct.var()
    vote_var = vote_pct.var()

    # 有效权重 = 方差贡献比例
    total_var = judge_var + vote_var + 1e-10
    judge_effective_weight = judge_var / total_var

    # 理论权重 (基于方差)
    theoretical_weight = judge_effective_weight

    # 相关性分析
    judge_corr = np.corrcoef(judge_pct, combined)[0, 1]
    vote_corr = np.corrcoef(vote_pct, combined)[0, 1]

    return {
        'method': 'percent',
        'theoretical_weight': theoretical_weight,
        'judge_corr': judge_corr,
        'vote_corr': vote_corr,
        'judge_effective_weight': judge_effective_weight,
        'vote_effective_weight': 1 - judge_effective_weight,
        'judge_pct_var': judge_var,
        'vote_pct_var': vote_var,
        'judge_pct_std': np.sqrt(judge_var),
        'vote_pct_std': np.sqrt(vote_var)
    }


def compute_marginal_impact_rank(judge_scores: np.ndarray,
                                  votes: np.ndarray,
                                  contestant_idx: int,
                                  delta: float = 0.01) -> Dict:
    """
    计算排名法下投票变化的边际影响

    在排名法中，边际影响是阶跃函数：
    - 如果投票变化导致排名变化，影响为 ±1
    - 否则影响为 0

    Args:
        judge_scores: 评委分数
        votes: 投票份额
        contestant_idx: 要分析的选手索引
        delta: 投票变化量

    Returns:
        边际影响结果
    """
    n = len(votes)

    # 原始排名
    vote_ranks_orig = stats.rankdata(-votes, method='average')
    judge_ranks = stats.rankdata(-judge_scores, method='average')
    combined_orig = judge_ranks + vote_ranks_orig

    # 增加投票后
    votes_plus = votes.copy()
    votes_plus[contestant_idx] += delta
    votes_plus = votes_plus / votes_plus.sum()
    vote_ranks_plus = stats.rankdata(-votes_plus, method='average')
    combined_plus = judge_ranks + vote_ranks_plus

    # 减少投票后
    votes_minus = votes.copy()
    votes_minus[contestant_idx] -= delta
    votes_minus = np.maximum(votes_minus, 0)
    votes_minus = votes_minus / votes_minus.sum()
    vote_ranks_minus = stats.rankdata(-votes_minus, method='average')
    combined_minus = judge_ranks + vote_ranks_minus

    # 边际影响
    combined_change = (combined_minus - combined_plus) / (2 * delta)

    # 淘汰者变化
    elim_orig = np.argmax(combined_orig)
    elim_plus = np.argmax(combined_plus)
    elim_minus = np.argmax(combined_minus)

    # 是否跨越了淘汰边界
    rank_change = vote_ranks_orig[contestant_idx] - vote_ranks_plus[contestant_idx]
    crosses_boundary = (elim_orig != elim_plus) or (elim_orig != elim_minus)

    return {
        'method': 'rank',
        'contestant_idx': contestant_idx,
        'original_vote_rank': vote_ranks_orig[contestant_idx],
        'marginal_impact': combined_change[contestant_idx],
        'rank_change': rank_change,
        'crosses_boundary': crosses_boundary,
        'is_discrete': True
    }


def compute_marginal_impact_pct(judge_scores: np.ndarray,
                                 votes: np.ndarray,
                                 contestant_idx: int,
                                 delta: float = 0.01) -> Dict:
    """
    计算百分比法下投票变化的边际影响

    在百分比法中，边际影响接近常数：
    ∂C_i/∂V_i ≈ 1 (当投票份额小时)

    Args:
        judge_scores: 评委分数
        votes: 投票份额
        contestant_idx: 要分析的选手索引
        delta: 投票变化量

    Returns:
        边际影响结果
    """
    n = len(votes)

    # 原始百分比
    judge_pct = judge_scores / (judge_scores.sum() + 1e-10)
    vote_pct = votes / (votes.sum() + 1e-10)
    combined_orig = judge_pct + vote_pct

    # 增加投票后
    votes_plus = votes.copy()
    votes_plus[contestant_idx] += delta
    vote_pct_plus = votes_plus / (votes_plus.sum() + 1e-10)
    combined_plus = judge_pct + vote_pct_plus

    # 减少投票后
    votes_minus = votes.copy()
    votes_minus[contestant_idx] = max(0, votes_minus[contestant_idx] - delta)
    vote_pct_minus = votes_minus / (votes_minus.sum() + 1e-10)
    combined_minus = judge_pct + vote_pct_minus

    # 边际影响 (数值微分)
    marginal_impact = (combined_plus[contestant_idx] - combined_minus[contestant_idx]) / (2 * delta)

    # 解析边际影响
    # ∂(V_i/ΣV)/∂V_i = (ΣV - V_i) / (ΣV)^2 ≈ 1 - V_i (当 ΣV ≈ 1)
    analytical_impact = 1 - vote_pct[contestant_idx]

    # 淘汰者变化
    elim_orig = np.argmin(combined_orig)
    elim_plus = np.argmin(combined_plus)
    elim_minus = np.argmin(combined_minus)

    crosses_boundary = (elim_orig != elim_plus) or (elim_orig != elim_minus)

    return {
        'method': 'percent',
        'contestant_idx': contestant_idx,
        'original_vote_pct': vote_pct[contestant_idx],
        'marginal_impact': marginal_impact,
        'analytical_impact': analytical_impact,
        'crosses_boundary': crosses_boundary,
        'is_continuous': True
    }


def compute_reversal_threshold(judge_scores: np.ndarray,
                                votes: np.ndarray,
                                method: str) -> Dict:
    """
    计算使淘汰者改变所需的最小投票变化 (逆转阈值)

    Args:
        judge_scores: 评委分数
        votes: 投票份额
        method: 'rank' 或 'percent'

    Returns:
        逆转阈值分析结果
    """
    n = len(votes)

    if method == 'rank':
        judge_ranks = stats.rankdata(-judge_scores, method='average')
        vote_ranks = stats.rankdata(-votes, method='average')
        combined = judge_ranks + vote_ranks
        eliminated_idx = np.argmax(combined)
    else:
        judge_pct = judge_scores / (judge_scores.sum() + 1e-10)
        vote_pct = votes / (votes.sum() + 1e-10)
        combined = judge_pct + vote_pct
        eliminated_idx = np.argmin(combined)

    # 二分搜索找到逆转阈值
    def check_reversal(delta: float) -> bool:
        new_votes = votes.copy()
        new_votes[eliminated_idx] += delta
        new_votes = new_votes / new_votes.sum()

        if method == 'rank':
            new_vote_ranks = stats.rankdata(-new_votes, method='average')
            new_combined = judge_ranks + new_vote_ranks
            new_eliminated = np.argmax(new_combined)
        else:
            new_vote_pct = new_votes / (new_votes.sum() + 1e-10)
            new_combined = judge_pct + new_vote_pct
            new_eliminated = np.argmin(new_combined)

        return new_eliminated != eliminated_idx

    # 二分搜索
    low, high = 0.0, 0.5
    threshold = high

    for _ in range(50):
        mid = (low + high) / 2
        if check_reversal(mid):
            threshold = mid
            high = mid
        else:
            low = mid

        if high - low < 1e-6:
            break

    return {
        'method': method,
        'eliminated_idx': eliminated_idx,
        'reversal_threshold': threshold,
        'threshold_percent': threshold * 100
    }


def load_week_data(raw_df: pd.DataFrame, vote_df: pd.DataFrame,
                   season: int, week: int) -> Optional[Dict]:
    """加载周数据 (复用 task2_scoring_methods.py 的逻辑)"""
    season_df = raw_df[raw_df['season'] == season].copy()
    if len(season_df) == 0:
        return None

    judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
    existing_cols = [col for col in judge_cols if col in season_df.columns]
    if not existing_cols:
        return None

    def is_valid(val):
        if pd.isna(val):
            return False
        if isinstance(val, str) and val.strip().upper() == 'N/A':
            return False
        try:
            return float(val) > 0
        except:
            return False

    valid_mask = season_df[existing_cols[0]].apply(is_valid)
    week_raw = season_df[valid_mask].copy()

    if len(week_raw) == 0:
        return None

    names = week_raw['celebrity_name'].tolist()

    scores = []
    for _, row in week_raw.iterrows():
        total = sum(float(row[c]) for c in existing_cols if is_valid(row[c]))
        scores.append(total)
    judge_scores = np.array(scores)

    vote_week = vote_df[(vote_df['season'] == season) & (vote_df['week'] == week)]
    if len(vote_week) == 0:
        return None

    votes = []
    for name in names:
        vote_row = vote_week[vote_week['contestant'] == name]
        if len(vote_row) > 0:
            votes.append(vote_row.iloc[0]['vote_share_mean'])
        else:
            votes.append(1.0 / len(names))

    votes = np.array(votes)
    votes = votes / votes.sum()

    return {
        'names': names,
        'judge_scores': judge_scores,
        'votes': votes
    }


def get_scoring_method(season: int) -> str:
    """根据赛季获取计分方法"""
    if season <= 2 or season >= 28:
        return 'rank'
    return 'percent'


def run_weight_analysis(data_path: str, vote_estimates_path: str) -> pd.DataFrame:
    """
    权重分解主函数

    对每个季/周计算:
    1. 排名法下的权重
    2. 百分比法下的权重
    3. 边际影响
    4. 逆转阈值

    Returns:
        权重分析结果 DataFrame
    """
    raw_df = pd.read_csv(data_path, encoding='utf-8-sig')
    vote_df = pd.read_csv(vote_estimates_path, encoding='utf-8-sig')

    seasons = sorted(raw_df['season'].unique())

    results = []

    for season in seasons:
        actual_method = get_scoring_method(season)

        for week in range(1, 12):
            week_data = load_week_data(raw_df, vote_df, season, week)
            if week_data is None:
                continue

            names = week_data['names']
            judge_scores = week_data['judge_scores']
            votes = week_data['votes']
            n = len(names)

            # 排名法权重
            rank_weight = compute_effective_weight_rank(judge_scores, votes)

            # 百分比法权重
            pct_weight = compute_effective_weight_pct(judge_scores, votes)

            # 逆转阈值
            rank_threshold = compute_reversal_threshold(judge_scores, votes, 'rank')
            pct_threshold = compute_reversal_threshold(judge_scores, votes, 'percent')

            # 边际影响 (对淘汰者)
            if actual_method == 'rank':
                judge_ranks = stats.rankdata(-judge_scores, method='average')
                vote_ranks = stats.rankdata(-votes, method='average')
                combined = judge_ranks + vote_ranks
                elim_idx = int(np.argmax(combined))
            else:
                judge_pct = judge_scores / (judge_scores.sum() + 1e-10)
                vote_pct = votes / (votes.sum() + 1e-10)
                combined = judge_pct + vote_pct
                elim_idx = int(np.argmin(combined))

            rank_marginal = compute_marginal_impact_rank(judge_scores, votes, elim_idx)
            pct_marginal = compute_marginal_impact_pct(judge_scores, votes, elim_idx)

            # 记录结果
            results.append({
                'season': season,
                'week': week,
                'n_contestants': n,
                'actual_method': actual_method,

                # 排名法权重
                'judge_weight_rank': rank_weight['judge_relative_weight'],
                'vote_weight_rank': 1 - rank_weight['judge_relative_weight'],
                'judge_corr_rank': rank_weight['judge_corr'],
                'vote_corr_rank': rank_weight['vote_corr'],

                # 百分比法权重
                'judge_weight_pct': pct_weight['judge_effective_weight'],
                'vote_weight_pct': 1 - pct_weight['judge_effective_weight'],
                'judge_pct_var': pct_weight['judge_pct_var'],
                'vote_pct_var': pct_weight['vote_pct_var'],

                # 权重差异
                'weight_difference': pct_weight['judge_effective_weight'] - rank_weight['judge_relative_weight'],

                # 边际影响
                'marginal_impact_rank': rank_marginal['marginal_impact'],
                'marginal_impact_pct': pct_marginal['marginal_impact'],
                'marginal_diff': pct_marginal['marginal_impact'] - rank_marginal['marginal_impact'],

                # 逆转阈值
                'reversal_threshold_rank': rank_threshold['reversal_threshold'],
                'reversal_threshold_pct': pct_threshold['reversal_threshold'],
                'threshold_diff': rank_threshold['reversal_threshold'] - pct_threshold['reversal_threshold']
            })

    return pd.DataFrame(results)


def analyze_weight_patterns(weight_df: pd.DataFrame) -> Dict:
    """分析权重模式"""
    patterns = {}

    # 整体统计
    patterns['overall'] = {
        'mean_judge_weight_rank': weight_df['judge_weight_rank'].mean(),
        'mean_judge_weight_pct': weight_df['judge_weight_pct'].mean(),
        'mean_weight_difference': weight_df['weight_difference'].mean(),
        'std_weight_difference': weight_df['weight_difference'].std()
    }

    # 按方法分组
    for method in ['rank', 'percent']:
        method_df = weight_df[weight_df['actual_method'] == method]
        if len(method_df) > 0:
            patterns[f'{method}_method'] = {
                'n_weeks': len(method_df),
                'mean_judge_weight_rank': method_df['judge_weight_rank'].mean(),
                'mean_judge_weight_pct': method_df['judge_weight_pct'].mean(),
                'mean_reversal_threshold_rank': method_df['reversal_threshold_rank'].mean(),
                'mean_reversal_threshold_pct': method_df['reversal_threshold_pct'].mean()
            }

    # 相关性分析：评委分数方差 vs 有效权重
    judge_var = weight_df['judge_pct_var'].values
    judge_weight = weight_df['judge_weight_pct'].values
    if len(judge_var) > 2:
        corr, p_value = stats.pearsonr(judge_var, judge_weight)
        patterns['variance_weight_correlation'] = {
            'correlation': corr,
            'p_value': p_value,
            'interpretation': '评委分数越分散，评委权重越大' if corr > 0 else '负相关'
        }

    return patterns


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(script_dir, '..', '2026_MCM-ICM_Problems',
                             '2026_MCM_Problem_C_Data.csv')
    vote_path = os.path.join(script_dir, 'task1_vote_estimates.csv')

    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在: {data_path}")
        return

    if not os.path.exists(vote_path):
        print(f"错误：投票估计文件不存在: {vote_path}")
        return

    print("=" * 60)
    print("Task 2: 权重分解与边际影响分析")
    print("=" * 60)

    # 运行权重分析
    print("\n正在分析权重...")
    weight_df = run_weight_analysis(data_path, vote_path)
    print(f"分析周数: {len(weight_df)}")

    # 分析模式
    patterns = analyze_weight_patterns(weight_df)

    # 打印结果
    print("\n## 权重分析结果")
    print(f"\n整体统计:")
    overall = patterns['overall']
    print(f"  排名法平均评委权重: {overall['mean_judge_weight_rank']:.3f}")
    print(f"  百分比法平均评委权重: {overall['mean_judge_weight_pct']:.3f}")
    print(f"  权重差异 (百分比法 - 排名法): {overall['mean_weight_difference']:.3f} (±{overall['std_weight_difference']:.3f})")

    # 按方法分组
    for method in ['rank', 'percent']:
        key = f'{method}_method'
        if key in patterns:
            stats_data = patterns[key]
            print(f"\n{method}方法周 ({stats_data['n_weeks']}周):")
            print(f"  平均逆转阈值 (排名法): {stats_data['mean_reversal_threshold_rank']:.4f}")
            print(f"  平均逆转阈值 (百分比法): {stats_data['mean_reversal_threshold_pct']:.4f}")

    # 方差-权重相关性
    if 'variance_weight_correlation' in patterns:
        var_corr = patterns['variance_weight_correlation']
        print(f"\n评委分数方差 vs 评委权重:")
        print(f"  相关系数: {var_corr['correlation']:.3f} (p={var_corr['p_value']:.4f})")
        print(f"  解释: {var_corr['interpretation']}")

    # 边际影响对比
    print("\n## 边际影响对比")
    print(f"排名法平均边际影响: {weight_df['marginal_impact_rank'].mean():.4f}")
    print(f"百分比法平均边际影响: {weight_df['marginal_impact_pct'].mean():.4f}")

    # 逆转阈值对比
    print("\n## 逆转阈值对比 (使淘汰者改变所需的投票增加)")
    print(f"排名法平均阈值: {weight_df['reversal_threshold_rank'].mean():.4f} ({weight_df['reversal_threshold_rank'].mean()*100:.2f}%)")
    print(f"百分比法平均阈值: {weight_df['reversal_threshold_pct'].mean():.4f} ({weight_df['reversal_threshold_pct'].mean()*100:.2f}%)")

    # 假设验证
    print("\n" + "=" * 60)
    print("假设验证")
    print("=" * 60)

    # 假设1：排名法更稳定
    rank_threshold = weight_df['reversal_threshold_rank'].mean()
    pct_threshold = weight_df['reversal_threshold_pct'].mean()
    if rank_threshold > pct_threshold:
        print(f"[OK] 假设1成立：排名法更稳定 (逆转阈值 {rank_threshold:.4f} > {pct_threshold:.4f})")
    else:
        print(f"[X] 假设1不成立：百分比法更稳定 (逆转阈值 {pct_threshold:.4f} > {rank_threshold:.4f})")

    # 假设2：百分比法更敏感
    rank_marginal = abs(weight_df['marginal_impact_rank'].mean())
    pct_marginal = abs(weight_df['marginal_impact_pct'].mean())
    if pct_marginal > rank_marginal:
        print(f"[OK] 假设2成立：百分比法更敏感 (边际影响 {pct_marginal:.4f} > {rank_marginal:.4f})")
    else:
        print(f"[X] 假设2不成立：排名法更敏感 (边际影响 {rank_marginal:.4f} > {pct_marginal:.4f})")

    # 假设3：评委分数离散时百分比法权重更大
    if 'variance_weight_correlation' in patterns:
        if patterns['variance_weight_correlation']['correlation'] > 0 and \
           patterns['variance_weight_correlation']['p_value'] < 0.05:
            print(f"[OK] 假设3成立：评委分数越分散，百分比法中评委权重越大")
        else:
            print(f"[?] 假设3待验证：相关不显著或方向相反")

    # 保存结果
    output_path = os.path.join(script_dir, 'task2_weight_analysis.csv')
    weight_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
