"""
Task 1: 投票预估模型 V2 - 软约束贝叶斯MCMC

改进点：
1. 使用软约束（Softmax似然）而非硬约束
2. 提供一致性度量指标 (Consistency Metrics)
3. 提供确定性度量指标 (Certainty Metrics)
4. 支持跨选手/周的不确定性分析

使用方法:
    python task1_vote_estimator_v2.py --seasons 1 2 3

输出:
    - vote_estimates_v2.csv: 投票估计结果
    - consistency_metrics.csv: 一致性指标
    - certainty_analysis.csv: 确定性分析
    - 可视化图表
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln, softmax
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 数据结构
# ============================================================

@dataclass
class WeekData:
    """单周比赛数据"""
    season: int
    week: int
    names: List[str]
    judge_scores: np.ndarray
    eliminated_idx: Optional[int]
    eliminated_name: Optional[str]


@dataclass
class EstimationResult:
    """单周估计结果"""
    season: int
    week: int
    names: List[str]
    judge_scores: np.ndarray
    eliminated_idx: int

    # 投票估计
    vote_share_mean: np.ndarray
    vote_share_std: np.ndarray
    vote_share_median: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    samples: np.ndarray

    # 一致性指标
    predicted_eliminated_idx: int
    is_correct: bool
    elimination_probability: float  # 模型预测该选手被淘汰的概率
    margin: float  # 淘汰边界距离
    rank_correlation: float  # 排名相关性

    # MCMC诊断
    acceptance_rate: float
    effective_sample_size: np.ndarray


# ============================================================
# 软约束贝叶斯估计器
# ============================================================

class SoftConstraintVoteEstimator:
    """
    软约束贝叶斯投票估计器

    使用Softmax似然函数，允许模型"犯错"，从而可以评估预测能力
    """

    def __init__(self, method: str = 'percent', temperature: float = 10.0):
        """
        初始化估计器

        Args:
            method: 'rank' 或 'percent'
            temperature: 软约束温度参数
                        - 越大约束越硬（趋近硬约束）
                        - 越小约束越软（趋近均匀分布）
                        - 推荐值: 5-20
        """
        self.method = method
        self.temperature = temperature

    def compute_combined_scores(self, votes: np.ndarray,
                                 judge_scores: np.ndarray) -> np.ndarray:
        """
        计算综合得分

        Args:
            votes: 投票份额 (和为1)
            judge_scores: 评委分数

        Returns:
            combined_scores: 综合得分（越低越容易被淘汰）
        """
        n = len(votes)

        if self.method == 'rank':
            # 排名法: 分数高 -> 排名小 -> 综合排名小 -> 不容易淘汰
            judge_ranks = stats.rankdata(-judge_scores)  # 高分排名靠前
            vote_ranks = stats.rankdata(-votes)
            # 排名和越大越容易被淘汰，取负使其与百分比法方向一致
            combined = -(judge_ranks + vote_ranks)
        else:
            # 百分比法: 百分比和越大越不容易被淘汰
            judge_pct = judge_scores / (judge_scores.sum() + 1e-10)
            vote_pct = votes / (votes.sum() + 1e-10)
            combined = judge_pct + vote_pct

        return combined

    def log_likelihood(self, votes: np.ndarray, judge_scores: np.ndarray,
                       eliminated_idx: int) -> float:
        """
        软约束对数似然函数

        使用Softmax将综合得分转化为淘汰概率
        得分越低，被淘汰概率越高

        Args:
            votes: 投票份额
            judge_scores: 评委分数
            eliminated_idx: 实际被淘汰者索引

        Returns:
            log_likelihood: 对数似然
        """
        combined = self.compute_combined_scores(votes, judge_scores)

        # Softmax: 得分越低，被淘汰概率越高
        # 使用负分数，这样低分对应高概率
        log_probs = -self.temperature * combined
        log_probs = log_probs - np.max(log_probs)  # 数值稳定
        probs = np.exp(log_probs) / np.sum(np.exp(log_probs))

        return np.log(probs[eliminated_idx] + 1e-10)

    def log_prior(self, votes: np.ndarray, judge_scores: np.ndarray,
                  alpha: float = 2.0, beta: float = 0.5) -> float:
        """
        对数先验概率

        先验假设: 投票份额与评委分数正相关，但允许偏离
        使用Dirichlet先验，浓度参数与评委分数相关

        Args:
            votes: 投票份额
            judge_scores: 评委分数
            alpha: 先验强度
            beta: 评委分数影响系数

        Returns:
            log_prior: 对数先验概率
        """
        # 归一化评委分数作为Dirichlet浓度参数的基础
        s_norm = judge_scores / (judge_scores.sum() + 1e-10)

        # Dirichlet浓度参数: 基础 + 评委分数贡献
        concentration = alpha * (1 + beta * s_norm * len(votes))
        concentration = np.maximum(concentration, 0.1)  # 确保正值

        # Dirichlet对数概率
        log_prob = (gammaln(concentration.sum()) -
                    gammaln(concentration).sum() +
                    np.sum((concentration - 1) * np.log(votes + 1e-10)))

        return log_prob

    def propose(self, current: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """
        Dirichlet提议分布

        在单纯形上进行随机游走
        """
        concentration = current * (1 / scale)
        concentration = np.maximum(concentration, 0.1)
        proposal = np.random.dirichlet(concentration)
        return proposal

    def log_proposal(self, x: np.ndarray, given: np.ndarray,
                     scale: float = 0.1) -> float:
        """提议分布的对数概率密度"""
        concentration = given * (1 / scale)
        concentration = np.maximum(concentration, 0.1)

        log_prob = (gammaln(concentration.sum()) -
                    gammaln(concentration).sum() +
                    np.sum((concentration - 1) * np.log(x + 1e-10)))
        return log_prob

    def mcmc_sample(self, judge_scores: np.ndarray, eliminated_idx: int,
                    n_samples: int = 10000, burnin: int = 2000,
                    thin: int = 2, proposal_scale: float = 0.08,
                    prior_alpha: float = 2.0, prior_beta: float = 0.5
                    ) -> Tuple[np.ndarray, float]:
        """
        Metropolis-Hastings MCMC采样

        Args:
            judge_scores: 评委分数
            eliminated_idx: 被淘汰者索引
            n_samples: 采样数量
            burnin: 预热期
            thin: 稀疏化间隔
            proposal_scale: 提议分布尺度
            prior_alpha, prior_beta: 先验参数

        Returns:
            samples: 后验样本
            acceptance_rate: 接受率
        """
        n = len(judge_scores)
        total_iterations = burnin + n_samples * thin

        # 初始化: 基于评委分数的软启动
        s_norm = judge_scores / judge_scores.sum()
        current = s_norm * 0.7 + np.ones(n) / n * 0.3  # 混合
        current = current / current.sum()

        samples = []
        accepted = 0

        # 当前状态的对数后验
        log_posterior_current = (
            self.log_likelihood(current, judge_scores, eliminated_idx) +
            self.log_prior(current, judge_scores, prior_alpha, prior_beta)
        )

        for i in range(total_iterations):
            # 提议新状态
            proposal = self.propose(current, proposal_scale)

            # 计算提议状态的对数后验
            log_posterior_proposal = (
                self.log_likelihood(proposal, judge_scores, eliminated_idx) +
                self.log_prior(proposal, judge_scores, prior_alpha, prior_beta)
            )

            # 提议分布比 (非对称提议需要修正)
            log_q_forward = self.log_proposal(proposal, current, proposal_scale)
            log_q_backward = self.log_proposal(current, proposal, proposal_scale)

            # Metropolis-Hastings接受概率
            log_alpha = (log_posterior_proposal - log_posterior_current +
                        log_q_backward - log_q_forward)

            # 接受或拒绝
            if np.log(np.random.rand() + 1e-10) < min(0, log_alpha):
                current = proposal
                log_posterior_current = log_posterior_proposal
                accepted += 1

            # 保存样本 (burnin后，稀疏化)
            if i >= burnin and (i - burnin) % thin == 0:
                samples.append(current.copy())

        acceptance_rate = accepted / total_iterations
        return np.array(samples), acceptance_rate

    def compute_effective_sample_size(self, samples: np.ndarray) -> np.ndarray:
        """
        计算有效样本量 (ESS)

        基于自相关函数估计
        """
        n_samples, n_dims = samples.shape
        ess = np.zeros(n_dims)

        for d in range(n_dims):
            x = samples[:, d]
            x = x - x.mean()

            # 自相关函数
            n = len(x)
            acf = np.correlate(x, x, mode='full')[n-1:] / (np.var(x) * n)

            # 找到第一个负值或截断点
            cutoff = n // 2
            for k in range(1, cutoff):
                if acf[k] < 0:
                    cutoff = k
                    break

            # ESS = n / (1 + 2 * sum(acf))
            tau = 1 + 2 * np.sum(acf[1:cutoff])
            ess[d] = n / max(tau, 1)

        return ess

    def estimate(self, week_data: WeekData, **kwargs) -> Optional[EstimationResult]:
        """
        估计单周的投票分布并计算所有指标

        Args:
            week_data: 周数据
            **kwargs: MCMC参数

        Returns:
            EstimationResult: 包含估计结果和所有指标
        """
        if week_data.eliminated_idx is None:
            return None

        judge_scores = week_data.judge_scores
        eliminated_idx = week_data.eliminated_idx
        n = len(judge_scores)

        # MCMC采样
        samples, acceptance_rate = self.mcmc_sample(
            judge_scores, eliminated_idx, **kwargs
        )

        # 基本统计量
        vote_share_mean = samples.mean(axis=0)
        vote_share_std = samples.std(axis=0)
        vote_share_median = np.median(samples, axis=0)
        ci_lower = np.percentile(samples, 2.5, axis=0)
        ci_upper = np.percentile(samples, 97.5, axis=0)

        # ESS
        ess = self.compute_effective_sample_size(samples)

        # ============ 一致性指标 ============

        # 1. 预测淘汰者 (使用后验均值)
        combined_mean = self.compute_combined_scores(vote_share_mean, judge_scores)
        predicted_eliminated_idx = np.argmin(combined_mean)
        is_correct = (predicted_eliminated_idx == eliminated_idx)

        # 2. 淘汰概率 (蒙特卡洛估计)
        elimination_probs = np.zeros(n)
        for sample in samples:
            combined = self.compute_combined_scores(sample, judge_scores)
            pred_elim = np.argmin(combined)
            elimination_probs[pred_elim] += 1
        elimination_probs /= len(samples)
        elimination_probability = elimination_probs[eliminated_idx]

        # 3. 淘汰边界距离
        combined_sorted = np.sort(combined_mean)
        if len(combined_sorted) >= 2:
            margin = combined_sorted[1] - combined_sorted[0]  # 次低 - 最低
        else:
            margin = 0.0

        # 4. 排名相关性
        predicted_ranks = stats.rankdata(-combined_mean)  # 高分排名靠前
        # 实际排名: 被淘汰者排名最后
        actual_ranks = np.ones(n)
        actual_ranks[eliminated_idx] = n  # 淘汰者最后
        # 简化: 其他人按评委分数排
        other_mask = np.arange(n) != eliminated_idx
        actual_ranks[other_mask] = stats.rankdata(-judge_scores[other_mask])

        rank_correlation, _ = stats.kendalltau(predicted_ranks, actual_ranks)

        return EstimationResult(
            season=week_data.season,
            week=week_data.week,
            names=week_data.names,
            judge_scores=judge_scores,
            eliminated_idx=eliminated_idx,
            vote_share_mean=vote_share_mean,
            vote_share_std=vote_share_std,
            vote_share_median=vote_share_median,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            samples=samples,
            predicted_eliminated_idx=predicted_eliminated_idx,
            is_correct=is_correct,
            elimination_probability=elimination_probability,
            margin=margin,
            rank_correlation=rank_correlation if not np.isnan(rank_correlation) else 0.0,
            acceptance_rate=acceptance_rate,
            effective_sample_size=ess
        )


# ============================================================
# 数据加载器
# ============================================================

class DataLoader:
    """数据加载与预处理"""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.seasons = sorted(self.df['season'].unique())

    def get_scoring_method(self, season: int) -> str:
        """Season 1-2, 28-34: rank; Season 3-27: percent"""
        if season <= 2 or season >= 28:
            return 'rank'
        return 'percent'

    def get_week_data(self, season: int, week: int) -> Optional[WeekData]:
        """获取特定季特定周的数据"""
        season_df = self.df[self.df['season'] == season].copy()
        if len(season_df) == 0:
            return None

        judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        existing_cols = [col for col in judge_cols if col in season_df.columns]
        if len(existing_cols) == 0:
            return None

        def is_valid_score(val):
            if pd.isna(val):
                return False
            if isinstance(val, str) and val.strip().upper() == 'N/A':
                return False
            try:
                return float(val) > 0
            except:
                return False

        first_col = existing_cols[0]
        valid_mask = season_df[first_col].apply(is_valid_score)
        week_df = season_df[valid_mask].copy()

        if len(week_df) == 0:
            return None

        names = week_df['celebrity_name'].tolist()
        scores = []

        for _, row in week_df.iterrows():
            total = 0
            for col in existing_cols:
                val = row[col]
                if is_valid_score(val):
                    total += float(val)
            scores.append(total)

        judge_scores = np.array(scores)

        # 确定被淘汰的选手
        eliminated_idx = None
        eliminated_name = None

        for i, (_, row) in enumerate(week_df.iterrows()):
            result = str(row['results']).lower()
            if f'eliminated week {week}' in result:
                eliminated_idx = i
                eliminated_name = names[i]
                break

        if eliminated_idx is None:
            for i, (_, row) in enumerate(week_df.iterrows()):
                placement = row['placement']
                n_contestants = len(week_df)
                n_remaining = n_contestants - week + 1
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
        """获取某季所有有淘汰的周"""
        weeks = []
        for week in range(1, 12):
            week_data = self.get_week_data(season, week)
            if week_data is not None and week_data.eliminated_idx is not None:
                weeks.append(week_data)
        return weeks


# ============================================================
# 指标计算与汇总
# ============================================================

class MetricsCalculator:
    """计算和汇总各类指标"""

    @staticmethod
    def compute_consistency_metrics(results: List[EstimationResult]) -> pd.DataFrame:
        """
        计算一致性指标汇总

        指标:
        1. EPA (Elimination Prediction Accuracy): 淘汰预测准确率
        2. Mean Elimination Probability: 平均淘汰概率
        3. Mean Kendall's τ: 平均排名相关性
        4. Top-2 Accuracy: 预测最低2人包含实际淘汰者的比例
        """
        records = []

        for r in results:
            # Top-2 检查
            combined = np.zeros(len(r.vote_share_mean))
            for i, (v, s) in enumerate(zip(r.vote_share_mean, r.judge_scores)):
                # 简化计算
                combined[i] = s / r.judge_scores.sum() + v
            bottom_2 = np.argsort(combined)[:2]
            is_top2 = r.eliminated_idx in bottom_2

            records.append({
                'season': r.season,
                'week': r.week,
                'n_contestants': len(r.names),
                'is_correct': r.is_correct,
                'elimination_probability': r.elimination_probability,
                'rank_correlation': r.rank_correlation,
                'margin': r.margin,
                'is_top2': is_top2,
                'acceptance_rate': r.acceptance_rate
            })

        df = pd.DataFrame(records)

        # 汇总统计
        summary = {
            'total_weeks': len(df),
            'EPA': df['is_correct'].mean(),
            'mean_elimination_prob': df['elimination_probability'].mean(),
            'std_elimination_prob': df['elimination_probability'].std(),
            'mean_kendall_tau': df['rank_correlation'].mean(),
            'top2_accuracy': df['is_top2'].mean(),
            'mean_margin': df['margin'].mean(),
            'mean_acceptance_rate': df['acceptance_rate'].mean()
        }

        return df, summary

    @staticmethod
    def compute_certainty_metrics(results: List[EstimationResult]) -> pd.DataFrame:
        """
        计算确定性指标

        分析不确定性在不同维度上的变化:
        - 被淘汰 vs 未淘汰选手
        - 不同周
        - 不同赛季
        - 评委分数高低
        """
        records = []

        for r in results:
            for i, name in enumerate(r.names):
                cv = r.vote_share_std[i] / (r.vote_share_mean[i] + 1e-10)
                ci_width = r.ci_upper[i] - r.ci_lower[i]
                ess_ratio = r.effective_sample_size[i] / len(r.samples)

                # 评委分数分位数
                judge_percentile = stats.percentileofscore(
                    r.judge_scores, r.judge_scores[i]
                )

                records.append({
                    'season': r.season,
                    'week': r.week,
                    'contestant': name,
                    'is_eliminated': i == r.eliminated_idx,
                    'judge_score': r.judge_scores[i],
                    'judge_percentile': judge_percentile,
                    'vote_share_mean': r.vote_share_mean[i],
                    'vote_share_std': r.vote_share_std[i],
                    'cv': cv,
                    'ci_width': ci_width,
                    'ci_lower': r.ci_lower[i],
                    'ci_upper': r.ci_upper[i],
                    'ess': r.effective_sample_size[i],
                    'ess_ratio': ess_ratio
                })

        df = pd.DataFrame(records)

        # 分组分析
        analysis = {}

        # 1. 被淘汰 vs 未淘汰
        analysis['by_elimination'] = df.groupby('is_eliminated').agg({
            'cv': ['mean', 'std'],
            'ci_width': ['mean', 'std'],
            'ess_ratio': 'mean'
        }).round(4)

        # 2. 按周
        analysis['by_week'] = df.groupby('week').agg({
            'cv': 'mean',
            'ci_width': 'mean'
        }).round(4)

        # 3. 按评委分数分位
        df['judge_quartile'] = pd.cut(df['judge_percentile'],
                                       bins=[0, 25, 50, 75, 100],
                                       labels=['Q1(低)', 'Q2', 'Q3', 'Q4(高)'])
        analysis['by_judge_score'] = df.groupby('judge_quartile').agg({
            'cv': 'mean',
            'ci_width': 'mean'
        }).round(4)

        return df, analysis


# ============================================================
# 可视化
# ============================================================

def create_visualizations(results: List[EstimationResult],
                          consistency_df: pd.DataFrame,
                          certainty_df: pd.DataFrame,
                          output_dir: str):
    """生成可视化图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
    except ImportError:
        print("matplotlib未安装，跳过可视化")
        return

    import os

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 淘汰预测准确率按赛季
    ax1 = axes[0, 0]
    epa_by_season = consistency_df.groupby('season')['is_correct'].mean()
    ax1.bar(epa_by_season.index, epa_by_season.values, color='steelblue', alpha=0.7)
    ax1.axhline(y=epa_by_season.mean(), color='red', linestyle='--',
                label=f'Mean: {epa_by_season.mean():.1%}')
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Elimination Prediction Accuracy')
    ax1.set_title('Q1a: Consistency - EPA by Season')
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # 2. 淘汰概率分布
    ax2 = axes[0, 1]
    ax2.hist(consistency_df['elimination_probability'], bins=20,
             color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=consistency_df['elimination_probability'].mean(),
                color='red', linestyle='--',
                label=f'Mean: {consistency_df["elimination_probability"].mean():.2f}')
    ax2.set_xlabel('Elimination Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Q1a: Distribution of Elimination Probability')
    ax2.legend()

    # 3. 排名相关性
    ax3 = axes[0, 2]
    ax3.hist(consistency_df['rank_correlation'], bins=20,
             color='green', alpha=0.7, edgecolor='black')
    ax3.axvline(x=consistency_df['rank_correlation'].mean(),
                color='red', linestyle='--',
                label=f'Mean τ: {consistency_df["rank_correlation"].mean():.2f}')
    ax3.set_xlabel("Kendall's τ")
    ax3.set_ylabel('Frequency')
    ax3.set_title("Q1a: Rank Correlation Distribution")
    ax3.legend()

    # 4. 确定性: CV按淘汰状态
    ax4 = axes[1, 0]
    eliminated = certainty_df[certainty_df['is_eliminated']]['cv']
    not_eliminated = certainty_df[~certainty_df['is_eliminated']]['cv']
    ax4.boxplot([eliminated, not_eliminated], labels=['Eliminated', 'Not Eliminated'])
    ax4.set_ylabel('Coefficient of Variation (CV)')
    ax4.set_title('Q1b: Certainty - CV by Elimination Status')

    # 5. 确定性: CI宽度按周
    ax5 = axes[1, 1]
    ci_by_week = certainty_df.groupby('week')['ci_width'].mean()
    ax5.plot(ci_by_week.index, ci_by_week.values, 'o-', color='purple')
    ax5.set_xlabel('Week')
    ax5.set_ylabel('Mean 95% CI Width')
    ax5.set_title('Q1b: Certainty - CI Width by Week')

    # 6. 确定性: CV vs 评委分数
    ax6 = axes[1, 2]
    ax6.scatter(certainty_df['judge_percentile'], certainty_df['cv'],
                c=certainty_df['is_eliminated'].map({True: 'red', False: 'blue'}),
                alpha=0.5, s=20)
    ax6.set_xlabel('Judge Score Percentile')
    ax6.set_ylabel('CV')
    ax6.set_title('Q1b: Certainty vs Judge Score')

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='red', label='Eliminated'),
                       Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='blue', label='Not Eliminated')]
    ax6.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task1_metrics_v2.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存")


# ============================================================
# 主函数
# ============================================================

def run_estimation(data_path: str,
                   seasons: Optional[List[int]] = None,
                   temperature: float = 10.0,
                   n_samples: int = 8000,
                   burnin: int = 2000) -> Tuple[List[EstimationResult], pd.DataFrame, pd.DataFrame]:
    """
    运行完整的投票估计流程
    """
    loader = DataLoader(data_path)

    if seasons is None:
        seasons = loader.seasons

    all_results = []

    for season in seasons:
        print(f"\n{'='*50}")
        print(f"Season {season}")
        print(f"{'='*50}")

        method = loader.get_scoring_method(season)
        print(f"计分方法: {'排名法' if method == 'rank' else '百分比法'}")

        estimator = SoftConstraintVoteEstimator(method=method, temperature=temperature)
        weeks = loader.get_all_weeks(season)

        for week_data in weeks:
            print(f"\n  Week {week_data.week}: {len(week_data.names)}人, "
                  f"淘汰: {week_data.eliminated_name}")

            try:
                result = estimator.estimate(
                    week_data,
                    n_samples=n_samples,
                    burnin=burnin
                )

                if result:
                    all_results.append(result)
                    status = "✓" if result.is_correct else "✗"
                    print(f"    预测: {status} | "
                          f"淘汰概率: {result.elimination_probability:.2f} | "
                          f"τ: {result.rank_correlation:.2f} | "
                          f"接受率: {result.acceptance_rate:.2f}")

            except Exception as e:
                print(f"    错误: {e}")

    # 计算汇总指标
    calc = MetricsCalculator()
    consistency_df, consistency_summary = calc.compute_consistency_metrics(all_results)
    certainty_df, certainty_analysis = calc.compute_certainty_metrics(all_results)

    return all_results, consistency_df, certainty_df, consistency_summary, certainty_analysis


def main():
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description='Task 1 V2: 软约束贝叶斯投票估计'
    )
    parser.add_argument('--seasons', type=int, nargs='+', default=None,
                        help='要分析的季数，默认全部')
    parser.add_argument('--temperature', type=float, default=10.0,
                        help='软约束温度参数 (5-20推荐)')
    parser.add_argument('--n-samples', type=int, default=8000,
                        help='MCMC采样数')
    parser.add_argument('--burnin', type=int, default=2000,
                        help='预热期')
    parser.add_argument('--no-plot', action='store_true',
                        help='不生成图表')

    args = parser.parse_args()

    # 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '2026_MCM-ICM_Problems',
                             '2026_MCM_Problem_C_Data.csv')

    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return

    print("="*60)
    print("Task 1 V2: 软约束贝叶斯投票估计")
    print(f"温度参数: {args.temperature}")
    print("="*60)

    # 运行估计
    results, consistency_df, certainty_df, con_summary, cert_analysis = run_estimation(
        data_path,
        seasons=args.seasons,
        temperature=args.temperature,
        n_samples=args.n_samples,
        burnin=args.burnin
    )

    # ============ 输出结果 ============

    print("\n" + "="*60)
    print("Q1a: 一致性指标汇总 (Consistency Metrics)")
    print("="*60)
    print(f"总周数: {con_summary['total_weeks']}")
    print(f"淘汰预测准确率 (EPA): {con_summary['EPA']:.1%}")
    print(f"平均淘汰概率: {con_summary['mean_elimination_prob']:.3f} "
          f"(±{con_summary['std_elimination_prob']:.3f})")
    print(f"Top-2准确率: {con_summary['top2_accuracy']:.1%}")
    print(f"平均Kendall's τ: {con_summary['mean_kendall_tau']:.3f}")
    print(f"平均淘汰边界: {con_summary['mean_margin']:.4f}")
    print(f"平均MCMC接受率: {con_summary['mean_acceptance_rate']:.2%}")

    print("\n" + "="*60)
    print("Q1b: 确定性指标汇总 (Certainty Metrics)")
    print("="*60)
    print("\n按淘汰状态:")
    print(cert_analysis['by_elimination'])
    print("\n按周:")
    print(cert_analysis['by_week'])
    print("\n按评委分数分位:")
    print(cert_analysis['by_judge_score'])

    # 保存结果
    # 1. 详细投票估计
    vote_records = []
    for r in results:
        for i, name in enumerate(r.names):
            vote_records.append({
                'season': r.season,
                'week': r.week,
                'contestant': name,
                'judge_score': r.judge_scores[i],
                'vote_share_mean': r.vote_share_mean[i],
                'vote_share_std': r.vote_share_std[i],
                'vote_share_median': r.vote_share_median[i],
                'ci_lower': r.ci_lower[i],
                'ci_upper': r.ci_upper[i],
                'is_eliminated': i == r.eliminated_idx,
                'ess': r.effective_sample_size[i]
            })

    votes_df = pd.DataFrame(vote_records)
    votes_df.to_csv(os.path.join(script_dir, 'vote_estimates_v2.csv'),
                    index=False, encoding='utf-8-sig')

    # 2. 一致性指标
    consistency_df.to_csv(os.path.join(script_dir, 'consistency_metrics.csv'),
                          index=False, encoding='utf-8-sig')

    # 3. 确定性指标
    certainty_df.to_csv(os.path.join(script_dir, 'certainty_analysis.csv'),
                        index=False, encoding='utf-8-sig')

    print(f"\n结果已保存到 {script_dir}")

    # 可视化
    if not args.no_plot:
        create_visualizations(results, consistency_df, certainty_df, script_dir)


if __name__ == '__main__':
    main()
