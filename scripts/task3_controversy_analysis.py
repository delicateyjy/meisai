"""
Task 3: 争议案例分析

分析4个争议选手：
1. Jerry Rice (Season 2, 排名法) - 5周评委最低分
2. Billy Ray Cyrus (Season 4, 百分比法) - 6周评委最低分
3. Bristol Palin (Season 11, 百分比法) - 多次评委最低分
4. Bobby Bones (Season 27, 百分比法) - 持续低评委分夺冠

分析内容：
1. 争议度指标计算
2. 反事实分析：如果使用另一种计分方法
3. 评委淘汰机制模拟

输出:
- task3_controversy_metrics.csv
- task3_counterfactual_detail.csv
- task3_judge_mechanism.csv
- task3_controversy_summary.csv
- figures/task3_*.png

使用方法:
    python task3_controversy_analysis.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互后端
warnings.filterwarnings('ignore')

# 导入Task 2的计分函数
from task2_scoring_methods import compute_ranking_method, compute_percentage_method, ScoringResult


# ============================================================================
# 数据路径配置
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "2026_MCM-ICM_Problems", "2026_MCM_Problem_C_Data.csv")
VOTE_ESTIMATES_PATH = os.path.join(PROJECT_ROOT, "results", "task1_vote_estimates.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================================
# 争议选手定义
# ============================================================================
CONTROVERSY_CASES = [
    {
        "name": "Jerry Rice",
        "season": 2,
        "actual_method": "rank",       # Season 2 使用排名法
        "final_placement": 2,          # 亚军
        "controversy_point": "5 weeks with lowest judge scores",
        "eliminated": False
    },
    {
        "name": "Billy Ray Cyrus",
        "season": 4,
        "actual_method": "percent",    # Season 4 使用百分比法
        "final_placement": 5,          # 第5名
        "controversy_point": "6 weeks with lowest judge scores",
        "eliminated": True
    },
    {
        "name": "Bristol Palin",
        "season": 11,
        "actual_method": "percent",    # Season 11 使用百分比法
        "final_placement": 3,          # 季军
        "controversy_point": "Multiple weeks with lowest judge scores",
        "eliminated": False
    },
    {
        "name": "Bobby Bones",
        "season": 27,
        "actual_method": "percent",    # Season 27 使用百分比法
        "final_placement": 1,          # 冠军
        "controversy_point": "Consistently low judge scores yet won",
        "eliminated": False
    }
]


# ============================================================================
# 数据加载函数
# ============================================================================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载原始数据和投票估计数据"""
    # 原始数据
    raw_df = pd.read_csv(DATA_PATH)

    # 投票估计
    vote_df = pd.read_csv(VOTE_ESTIMATES_PATH)

    return raw_df, vote_df


def get_season_weeks(vote_df: pd.DataFrame, season: int) -> List[int]:
    """获取某季的所有周数"""
    season_data = vote_df[vote_df['season'] == season]
    return sorted(season_data['week'].unique())


def get_week_data(vote_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """获取某季某周的所有选手数据"""
    return vote_df[(vote_df['season'] == season) & (vote_df['week'] == week)].copy()


# ============================================================================
# 争议度指标计算
# ============================================================================
@dataclass
class ControversyMetrics:
    """争议度指标"""
    contestant: str
    season: int
    weeks_participated: int
    times_judge_last: int           # 评委最低分次数
    times_saved_by_fans: int        # 被粉丝救回次数 (评委最低但未淘汰)
    avg_judge_rank: float           # 平均评委排名
    avg_vote_rank: float            # 平均投票排名
    controversy_intensity: float    # 争议强度 = avg_judge_rank - avg_vote_rank
    best_judge_rank: int            # 最佳评委排名
    worst_judge_rank: int           # 最差评委排名


def calculate_controversy_metrics(vote_df: pd.DataFrame,
                                  contestant: str,
                                  season: int) -> ControversyMetrics:
    """计算单个选手的争议度指标"""
    # 获取选手参赛周数据
    contestant_data = vote_df[
        (vote_df['season'] == season) &
        (vote_df['contestant'] == contestant)
    ].copy()

    weeks = sorted(contestant_data['week'].unique())

    times_judge_last = 0
    times_saved_by_fans = 0
    judge_ranks = []
    vote_ranks = []

    for week in weeks:
        week_data = get_week_data(vote_df, season, week)
        n_contestants = len(week_data)

        # 计算评委排名
        week_data['judge_rank'] = stats.rankdata(-week_data['judge_score'].values, method='average')
        week_data['vote_rank'] = stats.rankdata(-week_data['vote_share_mean'].values, method='average')

        # 获取该选手的排名
        contestant_row = week_data[week_data['contestant'] == contestant].iloc[0]
        judge_rank = contestant_row['judge_rank']
        vote_rank = contestant_row['vote_rank']
        is_eliminated = contestant_row['is_eliminated']

        judge_ranks.append(judge_rank)
        vote_ranks.append(vote_rank)

        # 是否评委最低
        if judge_rank == n_contestants:
            times_judge_last += 1
            # 如果评委最低但未被淘汰，则被粉丝救回
            if not is_eliminated:
                times_saved_by_fans += 1

    avg_judge_rank = np.mean(judge_ranks) if judge_ranks else 0
    avg_vote_rank = np.mean(vote_ranks) if vote_ranks else 0
    controversy_intensity = avg_judge_rank - avg_vote_rank

    return ControversyMetrics(
        contestant=contestant,
        season=season,
        weeks_participated=len(weeks),
        times_judge_last=times_judge_last,
        times_saved_by_fans=times_saved_by_fans,
        avg_judge_rank=avg_judge_rank,
        avg_vote_rank=avg_vote_rank,
        controversy_intensity=controversy_intensity,
        best_judge_rank=int(min(judge_ranks)) if judge_ranks else 0,
        worst_judge_rank=int(max(judge_ranks)) if judge_ranks else 0
    )


def calculate_all_controversy_metrics(vote_df: pd.DataFrame) -> pd.DataFrame:
    """计算所有争议选手的争议度指标"""
    results = []

    for case in CONTROVERSY_CASES:
        metrics = calculate_controversy_metrics(
            vote_df,
            case["name"],
            case["season"]
        )
        results.append({
            "contestant": metrics.contestant,
            "season": metrics.season,
            "actual_method": case["actual_method"],
            "final_placement": case["final_placement"],
            "weeks_participated": metrics.weeks_participated,
            "times_judge_last": metrics.times_judge_last,
            "times_saved_by_fans": metrics.times_saved_by_fans,
            "avg_judge_rank": round(metrics.avg_judge_rank, 2),
            "avg_vote_rank": round(metrics.avg_vote_rank, 2),
            "controversy_intensity": round(metrics.controversy_intensity, 2),
            "best_judge_rank": metrics.best_judge_rank,
            "worst_judge_rank": metrics.worst_judge_rank
        })

    return pd.DataFrame(results)


# ============================================================================
# 周度详细分析
# ============================================================================
def analyze_contestant_weekly(vote_df: pd.DataFrame,
                               contestant: str,
                               season: int) -> pd.DataFrame:
    """分析选手周度表现"""
    contestant_data = vote_df[
        (vote_df['season'] == season) &
        (vote_df['contestant'] == contestant)
    ].copy()

    weeks = sorted(contestant_data['week'].unique())

    results = []
    for week in weeks:
        week_data = get_week_data(vote_df, season, week)
        n = len(week_data)

        # 计算排名
        week_data['judge_rank'] = stats.rankdata(-week_data['judge_score'].values, method='average')
        week_data['vote_rank'] = stats.rankdata(-week_data['vote_share_mean'].values, method='average')

        # 获取选手数据
        row = week_data[week_data['contestant'] == contestant].iloc[0]

        # 确定实际淘汰者
        actual_eliminated = week_data[week_data['is_eliminated'] == True]
        eliminated_name = actual_eliminated['contestant'].values[0] if len(actual_eliminated) > 0 else "None"

        results.append({
            "contestant": contestant,
            "season": season,
            "week": week,
            "n_contestants": n,
            "judge_score": row['judge_score'],
            "judge_rank": int(row['judge_rank']),
            "vote_share": round(row['vote_share_mean'], 4),
            "vote_rank": int(row['vote_rank']),
            "is_judge_last": row['judge_rank'] == n,
            "is_eliminated": row['is_eliminated'],
            "week_eliminated": eliminated_name
        })

    return pd.DataFrame(results)


# ============================================================================
# 反事实分析
# ============================================================================
def counterfactual_week_analysis(vote_df: pd.DataFrame,
                                  season: int,
                                  week: int,
                                  contestant: str) -> Dict:
    """
    单周反事实分析：如果使用另一种方法，结果会如何变化？

    Returns:
        Dict with keys:
        - actual_method: 实际使用的方法
        - cf_method: 反事实方法
        - actual_eliminated: 实际淘汰者
        - cf_eliminated: 反事实淘汰者
        - contestant_would_be_eliminated: 该选手在反事实下是否会被淘汰
        - actual_rank: 实际综合排名
        - cf_rank: 反事实综合排名
    """
    week_data = get_week_data(vote_df, season, week)
    n = len(week_data)

    if n < 2:
        return None

    # 准备数据
    names = week_data['contestant'].tolist()
    judge_scores = week_data['judge_score'].values
    votes = week_data['vote_share_mean'].values

    # 确定实际方法
    if season in [1, 2] or season >= 28:
        actual_method = "rank"
        cf_method = "percent"
    else:
        actual_method = "percent"
        cf_method = "rank"

    # 计算实际结果
    if actual_method == "rank":
        actual_result = compute_ranking_method(judge_scores, votes, names)
        cf_result = compute_percentage_method(judge_scores, votes, names)
    else:
        actual_result = compute_percentage_method(judge_scores, votes, names)
        cf_result = compute_ranking_method(judge_scores, votes, names)

    # 获取选手索引
    contestant_idx = names.index(contestant) if contestant in names else -1
    if contestant_idx == -1:
        return None

    # 选手在两种方法下的排名
    actual_rank = actual_result.ranks[contestant_idx]
    cf_rank = cf_result.ranks[contestant_idx]

    return {
        "season": season,
        "week": week,
        "contestant": contestant,
        "n_contestants": n,
        "actual_method": actual_method,
        "cf_method": cf_method,
        "actual_eliminated": actual_result.eliminated_name,
        "cf_eliminated": cf_result.eliminated_name,
        "actual_rank": int(actual_rank),
        "cf_rank": int(cf_rank),
        "contestant_actual_eliminated": actual_result.eliminated_name == contestant,
        "contestant_cf_eliminated": cf_result.eliminated_name == contestant,
        "elimination_changed": actual_result.eliminated_name != cf_result.eliminated_name
    }


def run_contestant_counterfactual(vote_df: pd.DataFrame,
                                   contestant: str,
                                   season: int) -> pd.DataFrame:
    """对单个选手运行完整反事实分析"""
    contestant_data = vote_df[
        (vote_df['season'] == season) &
        (vote_df['contestant'] == contestant)
    ]

    weeks = sorted(contestant_data['week'].unique())
    results = []

    for week in weeks:
        cf_result = counterfactual_week_analysis(vote_df, season, week, contestant)
        if cf_result:
            results.append(cf_result)

    return pd.DataFrame(results)


# ============================================================================
# 评委淘汰机制模拟
# ============================================================================
def simulate_judge_mechanism(vote_df: pd.DataFrame,
                              season: int,
                              week: int) -> Dict:
    """
    模拟评委淘汰机制：从底部两人中选择

    假设：评委会选择评委分数更低的那位淘汰

    Returns:
        Dict with keys:
        - bottom_two: 底部两位选手
        - judge_choice: 评委选择淘汰谁
        - actual_eliminated: 实际淘汰者
        - mechanism_matches_actual: 机制结果是否与实际一致
    """
    week_data = get_week_data(vote_df, season, week)
    n = len(week_data)

    if n < 3:  # 需要至少3人才有底部两人的概念
        return None

    # 准备数据
    names = week_data['contestant'].tolist()
    judge_scores = week_data['judge_score'].values
    votes = week_data['vote_share_mean'].values

    # 使用排名法计算综合排名确定底部两人
    # (这是评委淘汰机制的标准做法)
    result = compute_ranking_method(judge_scores, votes, names)

    # 找到综合排名最低的两人 (排名值最大的两人)
    sorted_indices = np.argsort(result.combined_scores)[::-1]  # 降序
    bottom_two_indices = sorted_indices[:2]
    bottom_two_names = [names[i] for i in bottom_two_indices]
    bottom_two_scores = [judge_scores[i] for i in bottom_two_indices]

    # 评委选择评委分数更低的那位
    if bottom_two_scores[0] <= bottom_two_scores[1]:
        judge_choice = bottom_two_names[0]
    else:
        judge_choice = bottom_two_names[1]

    # 确定实际淘汰者
    actual_eliminated_row = week_data[week_data['is_eliminated'] == True]
    actual_eliminated = actual_eliminated_row['contestant'].values[0] if len(actual_eliminated_row) > 0 else "None"

    return {
        "season": season,
        "week": week,
        "n_contestants": n,
        "bottom_two": f"{bottom_two_names[0]} vs {bottom_two_names[1]}",
        "bottom_1_score": bottom_two_scores[0],
        "bottom_2_score": bottom_two_scores[1],
        "judge_choice": judge_choice,
        "actual_eliminated": actual_eliminated,
        "mechanism_matches_actual": judge_choice == actual_eliminated
    }


def run_judge_mechanism_analysis(vote_df: pd.DataFrame,
                                  contestant: str,
                                  season: int) -> pd.DataFrame:
    """对单个选手参与的周运行评委机制分析"""
    contestant_data = vote_df[
        (vote_df['season'] == season) &
        (vote_df['contestant'] == contestant)
    ]

    weeks = sorted(contestant_data['week'].unique())
    results = []

    for week in weeks:
        mech_result = simulate_judge_mechanism(vote_df, season, week)
        if mech_result:
            # 判断选手是否在底部两人中
            in_bottom_two = contestant in mech_result["bottom_two"]
            would_be_eliminated = mech_result["judge_choice"] == contestant

            results.append({
                **mech_result,
                "focus_contestant": contestant,
                "contestant_in_bottom_two": in_bottom_two,
                "contestant_would_be_eliminated": would_be_eliminated
            })

    return pd.DataFrame(results)


# ============================================================================
# 可视化
# ============================================================================
def plot_judge_rank_history(vote_df: pd.DataFrame) -> str:
    """绘制争议选手的评委排名历史"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, case in enumerate(CONTROVERSY_CASES):
        ax = axes[idx]
        contestant = case["name"]
        season = case["season"]

        contestant_data = vote_df[
            (vote_df['season'] == season) &
            (vote_df['contestant'] == contestant)
        ].copy()

        weeks = sorted(contestant_data['week'].unique())
        judge_ranks = []
        vote_ranks = []
        n_contestants_list = []

        for week in weeks:
            week_data = get_week_data(vote_df, season, week)
            n = len(week_data)
            n_contestants_list.append(n)

            week_data['judge_rank'] = stats.rankdata(-week_data['judge_score'].values, method='average')
            week_data['vote_rank'] = stats.rankdata(-week_data['vote_share_mean'].values, method='average')

            row = week_data[week_data['contestant'] == contestant].iloc[0]
            judge_ranks.append(row['judge_rank'])
            vote_ranks.append(row['vote_rank'])

        # 绘制
        ax.plot(weeks, judge_ranks, 'o-', color=colors[idx], linewidth=2,
                markersize=8, label='Judge Rank $R_i^S$')
        ax.plot(weeks, vote_ranks, 's--', color=colors[idx], linewidth=2,
                markersize=6, alpha=0.6, label='Vote Rank $R_i^\\pi$')

        # 标记最后名
        ax.fill_between(weeks, n_contestants_list, max(n_contestants_list) + 0.5,
                        alpha=0.2, color='red', label='Bottom Zone')

        ax.set_xlabel('Week', fontsize=11)
        ax.set_ylabel('Rank (1 = best)', fontsize=11)
        ax.set_title(f'{contestant} (Season {season})\nPlacement: {case["final_placement"]}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.invert_yaxis()  # 排名1在上
        ax.set_ylim(max(n_contestants_list) + 1, 0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "task3_judge_rank_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_controversy_comparison(metrics_df: pd.DataFrame) -> str:
    """绘制争议度指标对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    contestants = metrics_df['contestant'].tolist()
    x = np.arange(len(contestants))
    width = 0.6

    # 图1: 评委最低分次数 vs 被救回次数
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/4, metrics_df['times_judge_last'], width/2,
                    label='Times Judge Last', color='#ff6b6b')
    bars2 = ax1.bar(x + width/4, metrics_df['times_saved_by_fans'], width/2,
                    label='Times Saved by Fans', color='#4ecdc4')
    ax1.set_xlabel('Contestant')
    ax1.set_ylabel('Count')
    ax1.set_title('Judge Last vs Saved by Fans')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.split()[0] for c in contestants], rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 图2: 平均排名对比
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/4, metrics_df['avg_judge_rank'], width/2,
                    label='Avg Judge Rank', color='#ff6b6b')
    bars4 = ax2.bar(x + width/4, metrics_df['avg_vote_rank'], width/2,
                    label='Avg Vote Rank', color='#4ecdc4')
    ax2.set_xlabel('Contestant')
    ax2.set_ylabel('Average Rank')
    ax2.set_title('Average Judge Rank vs Vote Rank')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.split()[0] for c in contestants], rotation=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 图3: 争议强度
    ax3 = axes[2]
    colors = ['#d62728' if v > 0 else '#2ca02c' for v in metrics_df['controversy_intensity']]
    bars5 = ax3.bar(x, metrics_df['controversy_intensity'], width, color=colors)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Contestant')
    ax3.set_ylabel('Controversy Intensity\n(Judge Rank - Vote Rank)')
    ax3.set_title('Controversy Intensity\n(Positive = Fans favor more)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.split()[0] for c in contestants], rotation=15)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "task3_controversy_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_counterfactual_impact(all_cf_results: pd.DataFrame) -> str:
    """绘制反事实分析影响图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 为每个选手绘制周度排名变化
    contestants = all_cf_results['contestant'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, contestant in enumerate(contestants):
        cf_data = all_cf_results[all_cf_results['contestant'] == contestant]

        # 计算排名变化
        rank_change = cf_data['cf_rank'].values - cf_data['actual_rank'].values
        weeks = cf_data['week'].values

        ax.plot(weeks, rank_change, 'o-', color=colors[idx], linewidth=2,
                markersize=8, label=contestant.split()[0], alpha=0.8)

        # 标记可能被淘汰的周
        cf_eliminated = cf_data[cf_data['contestant_cf_eliminated'] == True]
        if len(cf_eliminated) > 0:
            for _, row in cf_eliminated.iterrows():
                ax.scatter([row['week']], [row['cf_rank'] - row['actual_rank']],
                          s=200, c='red', marker='x', zorder=5, linewidths=3)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Rank Change (CF - Actual)\nPositive = Worse in CF', fontsize=11)
    ax.set_title('Counterfactual Analysis: Rank Change Under Alternative Method\n(X marks weeks where contestant would be eliminated)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "task3_counterfactual_impact.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


# ============================================================================
# 汇总分析
# ============================================================================
def create_summary_table(metrics_df: pd.DataFrame,
                          all_cf_results: pd.DataFrame,
                          all_mech_results: pd.DataFrame) -> pd.DataFrame:
    """创建汇总表格"""
    summary = []

    for _, row in metrics_df.iterrows():
        contestant = row['contestant']
        season = row['season']

        # 反事实分析结果
        cf_data = all_cf_results[all_cf_results['contestant'] == contestant]
        weeks_cf_eliminated = len(cf_data[cf_data['contestant_cf_eliminated'] == True])
        weeks_changed = len(cf_data[cf_data['elimination_changed'] == True])

        # 评委机制分析
        mech_data = all_mech_results[all_mech_results['focus_contestant'] == contestant]
        weeks_in_bottom_two = len(mech_data[mech_data['contestant_in_bottom_two'] == True])
        weeks_would_be_eliminated_by_judges = len(mech_data[mech_data['contestant_would_be_eliminated'] == True])

        summary.append({
            "contestant": contestant,
            "season": season,
            "actual_method": row['actual_method'],
            "final_placement": row['final_placement'],
            "weeks_participated": row['weeks_participated'],
            "times_judge_last": row['times_judge_last'],
            "times_saved_by_fans": row['times_saved_by_fans'],
            "controversy_intensity": row['controversy_intensity'],
            "weeks_cf_would_eliminate": weeks_cf_eliminated,
            "weeks_elimination_changed": weeks_changed,
            "weeks_in_bottom_two": weeks_in_bottom_two,
            "weeks_judges_would_eliminate": weeks_would_be_eliminated_by_judges
        })

    return pd.DataFrame(summary)


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 60)
    print("Task 3: Controversy Case Analysis")
    print("=" * 60)

    # 加载数据
    print("\n[1/7] Loading data...")
    raw_df, vote_df = load_data()
    print(f"  - Loaded {len(vote_df)} vote estimates")

    # 计算争议度指标
    print("\n[2/7] Calculating controversy metrics...")
    metrics_df = calculate_all_controversy_metrics(vote_df)
    metrics_path = os.path.join(RESULTS_DIR, "task3_controversy_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  - Saved to: {metrics_path}")
    print("\nControversy Metrics Summary:")
    print(metrics_df.to_string(index=False))

    # 周度详细分析
    print("\n[3/7] Analyzing weekly performance...")
    all_weekly = []
    for case in CONTROVERSY_CASES:
        weekly = analyze_contestant_weekly(vote_df, case["name"], case["season"])
        all_weekly.append(weekly)
    all_weekly_df = pd.concat(all_weekly, ignore_index=True)
    weekly_path = os.path.join(RESULTS_DIR, "task3_weekly_performance.csv")
    all_weekly_df.to_csv(weekly_path, index=False)
    print(f"  - Saved to: {weekly_path}")

    # 反事实分析
    print("\n[4/7] Running counterfactual analysis...")
    all_cf = []
    for case in CONTROVERSY_CASES:
        cf = run_contestant_counterfactual(vote_df, case["name"], case["season"])
        all_cf.append(cf)
    all_cf_df = pd.concat(all_cf, ignore_index=True)
    cf_path = os.path.join(RESULTS_DIR, "task3_counterfactual_detail.csv")
    all_cf_df.to_csv(cf_path, index=False)
    print(f"  - Saved to: {cf_path}")

    # 统计反事实结果
    for case in CONTROVERSY_CASES:
        cf_data = all_cf_df[all_cf_df['contestant'] == case["name"]]
        cf_eliminated_weeks = cf_data[cf_data['contestant_cf_eliminated'] == True]['week'].tolist()
        if cf_eliminated_weeks:
            print(f"  - {case['name']}: Would be eliminated in week(s) {cf_eliminated_weeks} under {cf_data['cf_method'].iloc[0]}")
        else:
            print(f"  - {case['name']}: Would NOT be eliminated under alternative method")

    # 评委淘汰机制分析
    print("\n[5/7] Simulating judge elimination mechanism...")
    all_mech = []
    for case in CONTROVERSY_CASES:
        mech = run_judge_mechanism_analysis(vote_df, case["name"], case["season"])
        all_mech.append(mech)
    all_mech_df = pd.concat(all_mech, ignore_index=True)
    mech_path = os.path.join(RESULTS_DIR, "task3_judge_mechanism.csv")
    all_mech_df.to_csv(mech_path, index=False)
    print(f"  - Saved to: {mech_path}")

    # 统计评委机制结果
    for case in CONTROVERSY_CASES:
        mech_data = all_mech_df[all_mech_df['focus_contestant'] == case["name"]]
        in_bottom = mech_data[mech_data['contestant_in_bottom_two'] == True]
        would_eliminate = mech_data[mech_data['contestant_would_be_eliminated'] == True]
        print(f"  - {case['name']}: In bottom-two {len(in_bottom)} times, judges would eliminate {len(would_eliminate)} times")

    # 创建汇总表
    print("\n[6/7] Creating summary table...")
    summary_df = create_summary_table(metrics_df, all_cf_df, all_mech_df)
    summary_path = os.path.join(RESULTS_DIR, "task3_controversy_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  - Saved to: {summary_path}")
    print("\nFinal Summary:")
    print(summary_df.to_string(index=False))

    # 生成可视化
    print("\n[7/7] Generating visualizations...")
    fig1 = plot_judge_rank_history(vote_df)
    print(f"  - Saved: {fig1}")

    fig2 = plot_controversy_comparison(metrics_df)
    print(f"  - Saved: {fig2}")

    fig3 = plot_counterfactual_impact(all_cf_df)
    print(f"  - Saved: {fig3}")

    print("\n" + "=" * 60)
    print("Task 3 Analysis Complete!")
    print("=" * 60)

    # 打印关键发现
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print("\n1. JERRY RICE (Season 2, Rank Method):")
    jerry_cf = all_cf_df[all_cf_df['contestant'] == 'Jerry Rice']
    jerry_mech = all_mech_df[all_mech_df['focus_contestant'] == 'Jerry Rice']
    print(f"   - 5 weeks with lowest judge scores")
    print(f"   - Under percentage method: Would be eliminated in {jerry_cf[jerry_cf['contestant_cf_eliminated']==True]['week'].tolist()} weeks")
    print(f"   - Judge mechanism: In bottom-two {len(jerry_mech[jerry_mech['contestant_in_bottom_two']==True])} times")

    print("\n2. BILLY RAY CYRUS (Season 4, Percentage Method):")
    billy_cf = all_cf_df[all_cf_df['contestant'] == 'Billy Ray Cyrus']
    billy_mech = all_mech_df[all_mech_df['focus_contestant'] == 'Billy Ray Cyrus']
    print(f"   - 6 weeks with lowest judge scores")
    print(f"   - Under ranking method: Would be eliminated in {billy_cf[billy_cf['contestant_cf_eliminated']==True]['week'].tolist()} weeks")
    print(f"   - Judge mechanism: Would be eliminated {len(billy_mech[billy_mech['contestant_would_be_eliminated']==True])} times")

    print("\n3. BRISTOL PALIN (Season 11, Percentage Method):")
    bristol_cf = all_cf_df[all_cf_df['contestant'] == 'Bristol Palin']
    bristol_mech = all_mech_df[all_mech_df['focus_contestant'] == 'Bristol Palin']
    print(f"   - Multiple weeks with lowest judge scores")
    print(f"   - Under ranking method: Would be eliminated in {bristol_cf[bristol_cf['contestant_cf_eliminated']==True]['week'].tolist()} weeks")
    print(f"   - Judge mechanism: Would be eliminated {len(bristol_mech[bristol_mech['contestant_would_be_eliminated']==True])} times")

    print("\n4. BOBBY BONES (Season 27, Percentage Method) - MOST CONTROVERSIAL:")
    bobby_cf = all_cf_df[all_cf_df['contestant'] == 'Bobby Bones']
    bobby_mech = all_mech_df[all_mech_df['focus_contestant'] == 'Bobby Bones']
    bobby_metrics = metrics_df[metrics_df['contestant'] == 'Bobby Bones'].iloc[0]
    print(f"   - WON despite consistently low judge scores")
    print(f"   - Controversy intensity: {bobby_metrics['controversy_intensity']} (judge rank - vote rank)")
    print(f"   - Under ranking method: Would be eliminated in {bobby_cf[bobby_cf['contestant_cf_eliminated']==True]['week'].tolist()} weeks")
    print(f"   - Judge mechanism: Would be eliminated {len(bobby_mech[bobby_mech['contestant_would_be_eliminated']==True])} times")

    return metrics_df, all_cf_df, all_mech_df, summary_df


if __name__ == "__main__":
    main()
