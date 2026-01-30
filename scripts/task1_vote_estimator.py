"""
Task 1: 投票预估模型 - 贝叶斯MCMC蒙特卡洛模拟
基于评委分数和淘汰结果，反推估计每位选手的投票数

使用方法:
    python task1_vote_estimator.py

输出:
    - 每周每位选手的投票估计
    - 95%置信区间
    - 模型验证结果
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class WeekData:
    """单周比赛数据"""
    season: int
    week: int
    names: List[str]
    judge_scores: np.ndarray
    eliminated_idx: Optional[int]
    eliminated_name: Optional[str]


class VoteEstimator:
    """
    贝叶斯投票估计器

    使用MCMC采样估计观众投票分布，满足淘汰约束
    """

    def __init__(self, method: str = 'rank'):
        """
        初始化估计器

        Args:
            method: 'rank' (排名法, Season 1-2, 28-34)
                   或 'percent' (百分比法, Season 3-27)
        """
        self.method = method
        self.samples = None
        self.acceptance_rate = None

    def log_prior(self, votes: np.ndarray, judge_scores: np.ndarray,
                  prior_params: Tuple[float, float, float]) -> float:
        """
        计算先验对数概率

        假设投票服从对数正态分布，均值与评委分数相关

        Args:
            votes: 投票数组
            judge_scores: 评委分数数组
            prior_params: (alpha, beta_s, sigma) 先验参数

        Returns:
            对数先验概率
        """
        alpha, beta_s, sigma = prior_params

        # 标准化评委分数
        s_std = judge_scores.std()
        if s_std < 1e-10:
            s_norm = np.zeros_like(judge_scores)
        else:
            s_norm = (judge_scores - judge_scores.mean()) / s_std

        # 先验均值: 评委分数高 -> 预期投票高
        mu = alpha + beta_s * s_norm

        # 对数正态先验
        log_v = np.log(np.maximum(votes, 1))
        log_prob = -0.5 * np.sum((log_v - mu) ** 2 / sigma ** 2)

        return log_prob

    def check_elimination(self, votes: np.ndarray, judge_scores: np.ndarray,
                          eliminated_idx: int) -> bool:
        """
        检查给定投票是否能产生正确的淘汰结果

        Args:
            votes: 投票数组
            judge_scores: 评委分数数组
            eliminated_idx: 实际被淘汰选手的索引

        Returns:
            是否满足淘汰约束
        """
        n = len(votes)

        if self.method == 'rank':
            # 排名法: 评委排名 + 投票排名
            # 分数高 -> 排名小(靠前)
            judge_ranks = stats.rankdata(-judge_scores)
            vote_ranks = stats.rankdata(-votes)
            combined = judge_ranks + vote_ranks

            # combined最高的被淘汰
            return np.argmax(combined) == eliminated_idx

        else:  # percent
            # 百分比法: 评委百分比 + 投票百分比
            judge_sum = judge_scores.sum()
            vote_sum = votes.sum()

            if judge_sum < 1e-10 or vote_sum < 1e-10:
                return False

            judge_pct = judge_scores / judge_sum
            vote_pct = votes / vote_sum
            combined = judge_pct + vote_pct

            # combined最低的被淘汰
            return np.argmin(combined) == eliminated_idx

    def mcmc_sample(self, judge_scores: np.ndarray, eliminated_idx: int,
                    n_samples: int = 10000, burnin: int = 2000,
                    prior_params: Tuple[float, float, float] = (10, 2, 1),
                    proposal_scale: float = 0.05) -> Tuple[np.ndarray, float]:
        """
        MCMC采样估计投票分布

        使用Metropolis-Hastings算法在满足淘汰约束的条件下采样

        Args:
            judge_scores: 评委分数数组
            eliminated_idx: 被淘汰选手索引
            n_samples: 采样数量
            burnin: 预热期长度
            prior_params: 先验参数 (alpha, beta_s, sigma)
            proposal_scale: 提议分布的尺度

        Returns:
            (samples, acceptance_rate): 样本数组和接受率
        """
        n = len(judge_scores)

        # 归一化评委分数到 [0.1, 1] 区间，用作投票份额的基础
        s_min, s_max = judge_scores.min(), judge_scores.max()
        if s_max - s_min > 1e-10:
            s_norm = 0.1 + 0.9 * (judge_scores - s_min) / (s_max - s_min)
        else:
            s_norm = np.ones(n) / n

        # 初始化为投票份额 (归一化到和为1)
        current = s_norm / s_norm.sum()

        # 调整初始状态使其满足约束
        max_init_attempts = 1000
        for attempt in range(max_init_attempts):
            if self.check_elimination(current, judge_scores, eliminated_idx):
                break
            # 减少被淘汰者的份额
            current[eliminated_idx] *= 0.8
            current = current / current.sum()  # 重新归一化

        samples = []
        accepted = 0
        total_proposals = 0

        for i in range(n_samples + burnin):
            # 在单纯形上进行随机游走 (保持份额和为1)
            # 使用Dirichlet扰动
            alpha = current * (1 / proposal_scale)  # 浓度参数
            alpha = np.maximum(alpha, 0.1)  # 防止过小
            proposal = np.random.dirichlet(alpha)

            total_proposals += 1

            # 检查约束
            if self.check_elimination(proposal, judge_scores, eliminated_idx):
                # 计算接受概率
                # 使用对数份额作为先验
                log_prior_current = self.log_prior_share(current, judge_scores, prior_params)
                log_prior_proposal = self.log_prior_share(proposal, judge_scores, prior_params)

                # 提议分布比 (Dirichlet是对称的，当使用相同的浓度参数时)
                log_q_forward = self._log_dirichlet_pdf(proposal, current * (1 / proposal_scale))
                log_q_backward = self._log_dirichlet_pdf(current, proposal * (1 / proposal_scale))

                # Metropolis-Hastings接受率
                log_alpha = (log_prior_proposal - log_prior_current +
                            log_q_backward - log_q_forward)

                if np.log(np.random.rand()) < min(0, log_alpha):
                    current = proposal
                    accepted += 1

            # 保存样本 (burnin后)
            if i >= burnin:
                samples.append(current.copy())

        acceptance_rate = accepted / total_proposals
        return np.array(samples), acceptance_rate

    def log_prior_share(self, shares: np.ndarray, judge_scores: np.ndarray,
                        prior_params: Tuple[float, float, float]) -> float:
        """计算份额的对数先验概率"""
        alpha, beta_s, sigma = prior_params

        # 归一化评委分数
        s_sum = judge_scores.sum()
        if s_sum > 1e-10:
            expected_shares = judge_scores / s_sum
        else:
            expected_shares = np.ones(len(shares)) / len(shares)

        # 先验: 份额应该接近评委分数的归一化值
        # 使用对数份额的正态分布
        log_shares = np.log(np.maximum(shares, 1e-10))
        log_expected = np.log(np.maximum(expected_shares, 1e-10))

        log_prob = -0.5 * np.sum((log_shares - log_expected) ** 2 / sigma ** 2)
        return log_prob

    def _log_dirichlet_pdf(self, x: np.ndarray, alpha: np.ndarray) -> float:
        """计算Dirichlet分布的对数概率密度"""
        from scipy.special import gammaln
        alpha = np.maximum(alpha, 1e-10)
        x = np.maximum(x, 1e-10)
        return (gammaln(alpha.sum()) - gammaln(alpha).sum() +
                np.sum((alpha - 1) * np.log(x)))

    def estimate_votes(self, judge_scores: np.ndarray, eliminated_idx: int,
                       **kwargs) -> Dict:
        """
        估计投票并返回统计量

        Args:
            judge_scores: 评委分数数组
            eliminated_idx: 被淘汰选手索引
            **kwargs: 传递给mcmc_sample的参数

        Returns:
            包含估计结果的字典
        """
        samples, acc_rate = self.mcmc_sample(judge_scores, eliminated_idx, **kwargs)

        self.samples = samples
        self.acceptance_rate = acc_rate

        # samples现在是投票份额（和为1）
        vote_shares = samples

        # 转换为假设的绝对投票数（假设总投票为100万）
        total_votes = 1_000_000
        vote_counts = samples * total_votes

        return {
            'mean': vote_counts.mean(axis=0),
            'std': vote_counts.std(axis=0),
            'median': np.median(vote_counts, axis=0),
            'ci_lower': np.percentile(vote_counts, 2.5, axis=0),
            'ci_upper': np.percentile(vote_counts, 97.5, axis=0),
            'vote_share_mean': vote_shares.mean(axis=0),
            'vote_share_std': vote_shares.std(axis=0),
            'samples': samples,
            'acceptance_rate': acc_rate
        }


class DataLoader:
    """数据加载与预处理"""

    def __init__(self, csv_path: str):
        """
        加载CSV数据

        Args:
            csv_path: CSV文件路径
        """
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.seasons = sorted(self.df['season'].unique())

    def get_scoring_method(self, season: int) -> str:
        """
        获取特定季节使用的计分方法

        Season 1-2, 28-34: 排名法
        Season 3-27: 百分比法
        """
        if season <= 2 or season >= 28:
            return 'rank'
        else:
            return 'percent'

    def get_week_data(self, season: int, week: int) -> Optional[WeekData]:
        """
        获取特定季特定周的比赛数据

        Args:
            season: 季数
            week: 周数

        Returns:
            WeekData对象或None
        """
        season_df = self.df[self.df['season'] == season].copy()

        if len(season_df) == 0:
            return None

        # 构建评委分数列名
        judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]

        # 检查列是否存在
        existing_cols = [col for col in judge_cols if col in season_df.columns]
        if len(existing_cols) == 0:
            return None

        # 筛选该周有参赛的选手 (分数不为0且非N/A)
        def is_valid_score(val):
            if pd.isna(val):
                return False
            if isinstance(val, str) and val.strip().upper() == 'N/A':
                return False
            try:
                return float(val) > 0
            except:
                return False

        # 检查第一个评委列的有效性
        first_col = existing_cols[0]
        valid_mask = season_df[first_col].apply(is_valid_score)
        week_df = season_df[valid_mask].copy()

        if len(week_df) == 0:
            return None

        # 计算每位选手的总分
        names = week_df['celebrity_name'].tolist()
        scores = []

        for _, row in week_df.iterrows():
            total = 0
            count = 0
            for col in existing_cols:
                val = row[col]
                if is_valid_score(val):
                    total += float(val)
                    count += 1
            scores.append(total)

        judge_scores = np.array(scores)

        # 确定被淘汰的选手
        # 根据results列判断淘汰周
        eliminated_idx = None
        eliminated_name = None

        for i, (_, row) in enumerate(week_df.iterrows()):
            result = str(row['results']).lower()
            if f'eliminated week {week}' in result or f'week {week}' in result:
                # 检查下一周是否有分数，如果没有则本周被淘汰
                next_week_col = f'week{week+1}_judge1_score'
                if next_week_col in season_df.columns:
                    next_val = row[next_week_col]
                    if not is_valid_score(next_val):
                        eliminated_idx = i
                        eliminated_name = names[i]
                        break
                else:
                    # 没有下一周数据，检查result
                    if f'eliminated week {week}' in result:
                        eliminated_idx = i
                        eliminated_name = names[i]
                        break

        # 如果通过result列没找到，尝试通过placement推断
        if eliminated_idx is None:
            placements = week_df['placement'].values
            n_contestants = len(week_df)

            # 找出本周后被淘汰的选手
            for i, (_, row) in enumerate(week_df.iterrows()):
                placement = row['placement']
                n_remaining = n_contestants - week + 1

                # 如果placement等于n_remaining，说明这周被淘汰
                if placement == n_remaining:
                    eliminated_idx = i
                    eliminated_name = names[i]
                    break

        return WeekData(
            season=season,
            week=week,
            names=names,
            judge_scores=judge_scores,
            eliminated_idx=eliminated_idx,
            eliminated_name=eliminated_name
        )

    def get_all_weeks(self, season: int) -> List[WeekData]:
        """获取某季所有周的数据"""
        weeks = []
        for week in range(1, 12):
            week_data = self.get_week_data(season, week)
            if week_data is not None and week_data.eliminated_idx is not None:
                weeks.append(week_data)
        return weeks


def run_monte_carlo_estimation(data_path: str,
                                seasons: Optional[List[int]] = None,
                                n_samples: int = 5000,
                                burnin: int = 1000) -> pd.DataFrame:
    """
    运行蒙特卡洛模拟估计投票

    Args:
        data_path: 数据CSV路径
        seasons: 要分析的季数列表，None表示全部
        n_samples: MCMC采样数量
        burnin: 预热期长度

    Returns:
        包含所有估计结果的DataFrame
    """
    loader = DataLoader(data_path)

    if seasons is None:
        seasons = loader.seasons

    all_results = []

    for season in seasons:
        print(f"\n{'='*50}")
        print(f"处理 Season {season}")
        print(f"{'='*50}")

        method = loader.get_scoring_method(season)
        print(f"计分方法: {'排名法' if method == 'rank' else '百分比法'}")

        estimator = VoteEstimator(method=method)
        weeks = loader.get_all_weeks(season)

        for week_data in weeks:
            if week_data.eliminated_idx is None:
                print(f"  Week {week_data.week}: 无法确定淘汰选手，跳过")
                continue

            print(f"\n  Week {week_data.week}:")
            print(f"    选手数: {len(week_data.names)}")
            print(f"    被淘汰: {week_data.eliminated_name}")

            try:
                result = estimator.estimate_votes(
                    week_data.judge_scores,
                    week_data.eliminated_idx,
                    n_samples=n_samples,
                    burnin=burnin
                )

                print(f"    MCMC接受率: {result['acceptance_rate']:.3f}")

                # 保存每位选手的结果
                for i, name in enumerate(week_data.names):
                    all_results.append({
                        'season': season,
                        'week': week_data.week,
                        'contestant': name,
                        'judge_score': week_data.judge_scores[i],
                        'vote_mean': result['mean'][i],
                        'vote_median': result['median'][i],
                        'vote_std': result['std'][i],
                        'vote_ci_lower': result['ci_lower'][i],
                        'vote_ci_upper': result['ci_upper'][i],
                        'vote_share_mean': result['vote_share_mean'][i],
                        'vote_share_std': result['vote_share_std'][i],
                        'is_eliminated': i == week_data.eliminated_idx,
                        'scoring_method': method,
                        'mcmc_acceptance_rate': result['acceptance_rate']
                    })

            except Exception as e:
                print(f"    错误: {e}")
                continue

    df = pd.DataFrame(all_results)
    return df


def validate_model(results_df: pd.DataFrame, data_path: str) -> Dict:
    """
    验证模型: 检查估计的投票是否能正确预测淘汰结果

    Args:
        results_df: 估计结果DataFrame
        data_path: 原始数据路径

    Returns:
        验证结果字典
    """
    loader = DataLoader(data_path)

    correct = 0
    total = 0

    for (season, week), group in results_df.groupby(['season', 'week']):
        method = loader.get_scoring_method(season)
        estimator = VoteEstimator(method=method)

        judge_scores = group['judge_score'].values
        votes = group['vote_mean'].values
        actual_eliminated = group[group['is_eliminated']].index[0] - group.index[0]

        # 用估计的投票重新计算
        if estimator.check_elimination(votes, judge_scores, actual_eliminated):
            correct += 1

        total += 1

    return {
        'total_weeks': total,
        'correct_predictions': correct,
        'accuracy': correct / total if total > 0 else 0
    }


def plot_vote_distribution(results_df: pd.DataFrame, output_path: str):
    """
    绘制投票分布可视化

    Args:
        results_df: 估计结果DataFrame
        output_path: 输出图片路径
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
    except ImportError:
        print("matplotlib未安装，跳过可视化")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 评委分数 vs 投票份额
    ax1 = axes[0, 0]
    ax1.scatter(results_df['judge_score'], results_df['vote_share_mean'],
                c=results_df['is_eliminated'].map({True: 'red', False: 'blue'}),
                alpha=0.6)
    ax1.set_xlabel('Judge Score')
    ax1.set_ylabel('Estimated Vote Share')
    ax1.set_title('Judge Score vs Vote Share')
    ax1.legend(['Not Eliminated', 'Eliminated'], loc='upper left')

    # 2. 投票份额分布 (按是否淘汰)
    ax2 = axes[0, 1]
    eliminated = results_df[results_df['is_eliminated']]['vote_share_mean']
    not_eliminated = results_df[~results_df['is_eliminated']]['vote_share_mean']
    ax2.hist([eliminated, not_eliminated], bins=20, label=['Eliminated', 'Not Eliminated'],
             color=['red', 'blue'], alpha=0.7)
    ax2.set_xlabel('Vote Share')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Vote Share Distribution')
    ax2.legend()

    # 3. 每季平均接受率
    ax3 = axes[1, 0]
    acc_by_season = results_df.groupby('season')['mcmc_acceptance_rate'].mean()
    ax3.bar(acc_by_season.index, acc_by_season.values)
    ax3.set_xlabel('Season')
    ax3.set_ylabel('MCMC Acceptance Rate')
    ax3.set_title('MCMC Acceptance Rate by Season')
    ax3.axhline(y=0.234, color='r', linestyle='--', label='Optimal (0.234)')
    ax3.legend()

    # 4. 投票估计不确定性
    ax4 = axes[1, 1]
    ax4.errorbar(range(len(results_df[:20])),
                 results_df['vote_share_mean'][:20],
                 yerr=results_df['vote_share_std'][:20] * 1.96,
                 fmt='o', capsize=3, alpha=0.7)
    ax4.set_xlabel('Contestant Index')
    ax4.set_ylabel('Vote Share (with 95% CI)')
    ax4.set_title('Vote Share Estimates with Uncertainty (First 20)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存到: {output_path}")


def main():
    """主函数"""
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Task 1: 投票预估模型 - 贝叶斯MCMC蒙特卡洛模拟')
    parser.add_argument('--seasons', type=int, nargs='+', default=[1, 2, 3],
                        help='要分析的季数，默认为[1,2,3]')
    parser.add_argument('--all-seasons', action='store_true',
                        help='分析所有季')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='MCMC采样数量')
    parser.add_argument('--burnin', type=int, default=1000,
                        help='MCMC预热期长度')
    parser.add_argument('--no-plot', action='store_true',
                        help='不生成可视化图')

    args = parser.parse_args()

    # 数据路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '2026_MCM-ICM_Problems',
                             '2026_MCM_Problem_C_Data.csv')

    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        print("请确保数据文件路径正确")
        return

    print("="*60)
    print("Task 1: 投票预估模型 - 贝叶斯MCMC蒙特卡洛模拟")
    print("="*60)

    # 确定分析的季数
    seasons = None if args.all_seasons else args.seasons

    # 运行估计
    results_df = run_monte_carlo_estimation(
        data_path,
        seasons=seasons,
        n_samples=args.n_samples,
        burnin=args.burnin
    )

    # 保存结果
    output_path = os.path.join(script_dir, 'vote_estimates.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_path}")

    # 验证模型
    print("\n" + "="*60)
    print("模型验证")
    print("="*60)
    validation = validate_model(results_df, data_path)
    print(f"总周数: {validation['total_weeks']}")
    print(f"正确预测: {validation['correct_predictions']}")
    print(f"准确率: {validation['accuracy']:.2%}")

    # 生成可视化
    if not args.no_plot:
        plot_path = os.path.join(script_dir, 'vote_distribution.png')
        plot_vote_distribution(results_df, plot_path)

    # 输出示例结果
    print("\n" + "="*60)
    print("示例结果 (Season 1, Week 1)")
    print("="*60)
    sample = results_df[(results_df['season'] == 1) & (results_df['week'] == 1)]
    if len(sample) > 0:
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(sample[['contestant', 'judge_score', 'vote_share_mean',
                      'vote_share_std', 'is_eliminated']].to_string(index=False))

    # 输出汇总统计
    print("\n" + "="*60)
    print("汇总统计")
    print("="*60)
    print(f"总季数: {results_df['season'].nunique()}")
    print(f"总周数: {len(results_df.groupby(['season', 'week']))}")
    print(f"总选手-周记录: {len(results_df)}")
    print(f"平均MCMC接受率: {results_df['mcmc_acceptance_rate'].mean():.3f}")
    print(f"被淘汰选手平均投票份额: {results_df[results_df['is_eliminated']]['vote_share_mean'].mean():.4f}")
    print(f"未淘汰选手平均投票份额: {results_df[~results_df['is_eliminated']]['vote_share_mean'].mean():.4f}")


if __name__ == '__main__':
    main()
