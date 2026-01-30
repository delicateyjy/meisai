"""
Task 2: 计分方法核心函数库 + 反事实模拟

职责:
1. compute_ranking_method(): 排名法计算
2. compute_percentage_method(): 百分比法计算
3. run_counterfactual(): 反事实模拟主函数

输出: task2_counterfactual.csv

使用方法:
    python task2_scoring_methods.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import os
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ScoringResult:
    """单周计分结果"""
    combined_scores: np.ndarray  # 综合得分
    ranks: np.ndarray            # 综合排名
    eliminated_idx: int          # 被淘汰选手索引
    eliminated_name: str         # 被淘汰选手姓名
    margin: float                # 淘汰边距


def compute_ranking_method(judge_scores: np.ndarray,
                           votes: np.ndarray,
                           names: List[str] = None) -> ScoringResult:
    """
    排名法计算

    排名法规则 (Season 1-2, 28-34):
    1. 将评委分数转换为排名 (分数越高排名越小)
    2. 将观众投票转换为排名 (投票越高排名越小)
    3. 综合排名值 = 评委排名 + 投票排名
    4. 综合排名值最大的选手被淘汰

    Args:
        judge_scores: 各选手的评委总分
        votes: 各选手的投票份额 (归一化后)
        names: 选手姓名列表 (可选)

    Returns:
        ScoringResult: 包含综合得分、排名、淘汰者等信息
    """
    n = len(judge_scores)

    # 计算排名 (使用 'average' 处理平局)
    # rankdata 默认是升序，所以取负值使高分排名小
    judge_ranks = stats.rankdata(-judge_scores, method='average')
    vote_ranks = stats.rankdata(-votes, method='average')

    # 综合排名值 = 评委排名 + 投票排名
    combined = judge_ranks + vote_ranks

    # 被淘汰者: 综合排名值最大
    eliminated_idx = int(np.argmax(combined))

    # 计算最终排名 (按综合值升序)
    final_ranks = stats.rankdata(combined, method='average')

    # 淘汰边距: 第二大与最大的差
    sorted_combined = np.sort(combined)
    if n > 1:
        margin = sorted_combined[-1] - sorted_combined[-2]
    else:
        margin = 0.0

    eliminated_name = names[eliminated_idx] if names else f"选手{eliminated_idx}"

    return ScoringResult(
        combined_scores=combined,
        ranks=final_ranks,
        eliminated_idx=eliminated_idx,
        eliminated_name=eliminated_name,
        margin=margin
    )


def compute_percentage_method(judge_scores: np.ndarray,
                              votes: np.ndarray,
                              names: List[str] = None) -> ScoringResult:
    """
    百分比法计算

    百分比法规则 (Season 3-27):
    1. 评委分数百分比 = S_i / sum(S)
    2. 投票百分比 = V_i (假设已归一化)
    3. 综合百分比 = 评委百分比 + 投票百分比
    4. 综合百分比最低的选手被淘汰

    Args:
        judge_scores: 各选手的评委总分
        votes: 各选手的投票份额 (归一化后)
        names: 选手姓名列表 (可选)

    Returns:
        ScoringResult: 包含综合得分、排名、淘汰者等信息
    """
    n = len(judge_scores)

    # 计算百分比
    judge_pct = judge_scores / (judge_scores.sum() + 1e-10)
    vote_pct = votes / (votes.sum() + 1e-10)  # 确保归一化

    # 综合百分比
    combined = judge_pct + vote_pct

    # 被淘汰者: 综合百分比最低
    eliminated_idx = int(np.argmin(combined))

    # 计算最终排名 (按综合值降序，百分比高排名小)
    final_ranks = stats.rankdata(-combined, method='average')

    # 淘汰边距: 第二低与最低的差
    sorted_combined = np.sort(combined)
    if n > 1:
        margin = sorted_combined[1] - sorted_combined[0]
    else:
        margin = 0.0

    eliminated_name = names[eliminated_idx] if names else f"选手{eliminated_idx}"

    return ScoringResult(
        combined_scores=combined,
        ranks=final_ranks,
        eliminated_idx=eliminated_idx,
        eliminated_name=eliminated_name,
        margin=margin
    )


def get_scoring_method(season: int) -> str:
    """
    根据赛季获取计分方法

    - Season 1-2, 28-34: 排名法 (rank)
    - Season 3-27: 百分比法 (percent)
    """
    if season <= 2 or season >= 28:
        return 'rank'
    return 'percent'


def load_week_data(raw_df: pd.DataFrame, vote_df: pd.DataFrame,
                   season: int, week: int) -> Optional[Dict]:
    """
    加载指定季/周的数据

    Args:
        raw_df: 原始数据 DataFrame
        vote_df: 投票估计 DataFrame
        season: 赛季号
        week: 周次

    Returns:
        包含 judge_scores, votes, names, eliminated_idx 的字典，或 None
    """
    # 筛选原始数据
    season_df = raw_df[raw_df['season'] == season].copy()
    if len(season_df) == 0:
        return None

    # 评委分数列
    judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
    existing_cols = [col for col in judge_cols if col in season_df.columns]
    if not existing_cols:
        return None

    # 筛选有效分数
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

    # 获取选手姓名
    names = week_raw['celebrity_name'].tolist()

    # 计算评委总分
    scores = []
    for _, row in week_raw.iterrows():
        total = sum(float(row[c]) for c in existing_cols if is_valid(row[c]))
        scores.append(total)
    judge_scores = np.array(scores)

    # 从投票估计中获取投票份额
    vote_week = vote_df[(vote_df['season'] == season) & (vote_df['week'] == week)]
    if len(vote_week) == 0:
        return None

    # 按选手名匹配
    votes = []
    found_eliminated_idx = None
    for i, name in enumerate(names):
        vote_row = vote_week[vote_week['contestant'] == name]
        if len(vote_row) > 0:
            votes.append(vote_row.iloc[0]['vote_share_mean'])
            if vote_row.iloc[0]['is_eliminated']:
                found_eliminated_idx = i
        else:
            # 如果找不到，使用均匀分布
            votes.append(1.0 / len(names))

    votes = np.array(votes)
    votes = votes / votes.sum()  # 确保归一化

    # 从原始数据确定淘汰者
    actual_eliminated_idx = None
    for i, (_, row) in enumerate(week_raw.iterrows()):
        result = str(row['results']).lower()
        if f'eliminated week {week}' in result:
            actual_eliminated_idx = i
            break

    if actual_eliminated_idx is None:
        actual_eliminated_idx = found_eliminated_idx

    if actual_eliminated_idx is None:
        return None

    return {
        'names': names,
        'judge_scores': judge_scores,
        'votes': votes,
        'actual_eliminated_idx': actual_eliminated_idx,
        'actual_eliminated_name': names[actual_eliminated_idx]
    }


def run_counterfactual(data_path: str, vote_estimates_path: str) -> pd.DataFrame:
    """
    反事实模拟主函数

    对每个季/周：
    1. 用实际方法计算结果
    2. 用反事实方法计算结果
    3. 比较差异

    Args:
        data_path: 原始数据文件路径
        vote_estimates_path: 投票估计文件路径

    Returns:
        DataFrame 包含反事实模拟结果
    """
    # 加载数据
    raw_df = pd.read_csv(data_path, encoding='utf-8-sig')
    vote_df = pd.read_csv(vote_estimates_path, encoding='utf-8-sig')

    seasons = sorted(raw_df['season'].unique())

    results = []

    for season in seasons:
        actual_method = get_scoring_method(season)
        cf_method = 'percent' if actual_method == 'rank' else 'rank'

        for week in range(1, 12):
            week_data = load_week_data(raw_df, vote_df, season, week)
            if week_data is None:
                continue

            names = week_data['names']
            judge_scores = week_data['judge_scores']
            votes = week_data['votes']
            actual_elim_idx = week_data['actual_eliminated_idx']

            # 用实际方法计算
            if actual_method == 'rank':
                actual_result = compute_ranking_method(judge_scores, votes, names)
            else:
                actual_result = compute_percentage_method(judge_scores, votes, names)

            # 用反事实方法计算
            if cf_method == 'rank':
                cf_result = compute_ranking_method(judge_scores, votes, names)
            else:
                cf_result = compute_percentage_method(judge_scores, votes, names)

            # 计算排名相关性
            tau, _ = stats.kendalltau(actual_result.ranks, cf_result.ranks)
            rho, _ = stats.spearmanr(actual_result.ranks, cf_result.ranks)

            # 记录结果
            results.append({
                'season': season,
                'week': week,
                'n_contestants': len(names),
                'actual_method': actual_method,
                'cf_method': cf_method,
                'actual_eliminated': actual_result.eliminated_name,
                'cf_eliminated': cf_result.eliminated_name,
                'actual_eliminated_idx': actual_result.eliminated_idx,
                'cf_eliminated_idx': cf_result.eliminated_idx,
                'elimination_flipped': actual_result.eliminated_idx != cf_result.eliminated_idx,
                'actual_margin': actual_result.margin,
                'cf_margin': cf_result.margin,
                'rank_tau': tau if not np.isnan(tau) else 1.0,
                'rank_rho': rho if not np.isnan(rho) else 1.0,
                # 存储详细排名
                'actual_ranks': actual_result.ranks.tolist(),
                'cf_ranks': cf_result.ranks.tolist()
            })

    return pd.DataFrame(results)


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 数据路径
    data_path = os.path.join(script_dir, '..', '2026_MCM-ICM_Problems',
                             '2026_MCM_Problem_C_Data.csv')
    vote_path = os.path.join(script_dir, 'task1_vote_estimates.csv')

    # 检查文件存在
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在: {data_path}")
        return

    if not os.path.exists(vote_path):
        print(f"错误：投票估计文件不存在: {vote_path}")
        print("请先运行 task1_vote_estimator_v3.py 生成投票估计")
        return

    print("=" * 60)
    print("Task 2: 计分方法反事实模拟")
    print("=" * 60)

    # 运行反事实模拟
    print("\n正在运行反事实模拟...")
    cf_df = run_counterfactual(data_path, vote_path)

    # 输出统计
    print(f"\n分析周数: {len(cf_df)}")

    # 按实际方法分组
    rank_weeks = cf_df[cf_df['actual_method'] == 'rank']
    pct_weeks = cf_df[cf_df['actual_method'] == 'percent']

    print(f"\n排名法周数 (S1-2, 28-34): {len(rank_weeks)}")
    print(f"百分比法周数 (S3-27): {len(pct_weeks)}")

    # 翻转率
    overall_flip = cf_df['elimination_flipped'].mean()
    rank_flip = rank_weeks['elimination_flipped'].mean() if len(rank_weeks) > 0 else 0
    pct_flip = pct_weeks['elimination_flipped'].mean() if len(pct_weeks) > 0 else 0

    print(f"\n淘汰翻转率:")
    print(f"  整体: {overall_flip:.1%} ({cf_df['elimination_flipped'].sum()}/{len(cf_df)})")
    print(f"  排名法→百分比法: {rank_flip:.1%}")
    print(f"  百分比法→排名法: {pct_flip:.1%}")

    # 排名相关性
    print(f"\n排名相关性:")
    print(f"  平均 Kendall's τ: {cf_df['rank_tau'].mean():.3f}")
    print(f"  平均 Spearman's ρ: {cf_df['rank_rho'].mean():.3f}")

    # 保存结果 (移除 list 类型的列)
    output_df = cf_df.drop(columns=['actual_ranks', 'cf_ranks'])
    output_path = os.path.join(script_dir, 'task2_counterfactual.csv')
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {output_path}")

    # 打印一些翻转的例子
    flipped = cf_df[cf_df['elimination_flipped']]
    if len(flipped) > 0:
        print(f"\n翻转示例 (前5个):")
        for _, row in flipped.head(5).iterrows():
            print(f"  S{row['season']}W{row['week']}: "
                  f"{row['actual_eliminated']} ({row['actual_method']}) → "
                  f"{row['cf_eliminated']} ({row['cf_method']})")


if __name__ == '__main__':
    main()
