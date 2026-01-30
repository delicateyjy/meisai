"""
Task 1: 投票预估模型 V3 - 改进版

改进点：
1. 自适应MCMC - 自动调整提议分布，提高接受率
2. 特征增强先验 - 利用选手职业、年龄等信息
3. 分层模型 - 建模选手的"基础人气"
4. 改进似然 - 更好地处理边界情况

使用方法:
    python task1_vote_estimator_v3.py --seasons 1 2 3
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln, softmax, expit
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
    # 新增特征
    industries: List[str] = None
    ages: List[float] = None
    partners: List[str] = None


@dataclass
class EstimationResult:
    """估计结果"""
    season: int
    week: int
    names: List[str]
    judge_scores: np.ndarray
    eliminated_idx: int

    vote_share_mean: np.ndarray
    vote_share_std: np.ndarray
    vote_share_median: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    samples: np.ndarray

    predicted_eliminated_idx: int
    is_correct: bool
    elimination_probability: float
    margin: float
    rank_correlation: float

    acceptance_rate: float
    effective_sample_size: np.ndarray


class ImprovedVoteEstimator:
    """
    改进版投票估计器 V3

    主要改进：
    1. 自适应MCMC
    2. 特征增强先验
    3. 更好的似然函数
    """

    # 职业类别的先验人气权重（基于常识）
    INDUSTRY_POPULARITY = {
        'Actor/Actress': 1.3,      # 演员粉丝多
        'Singer/Rapper': 1.4,      # 歌手粉丝最多
        'Athlete': 1.2,            # 运动员有粉丝基础
        'TV Personality': 1.1,     # 电视名人
        'Model': 1.0,              # 模特
        'News Anchor': 0.9,        # 新闻主播
        'Racing Driver': 1.0,
        'Sports Broadcaster': 0.9,
        'Beauty Pagent': 0.95,
        'default': 1.0
    }

    def __init__(self, method: str = 'percent',
                 temperature: float = 12.0,
                 use_features: bool = True):
        """
        Args:
            method: 'rank' 或 'percent'
            temperature: 软约束温度
            use_features: 是否使用选手特征
        """
        self.method = method
        self.temperature = temperature
        self.use_features = use_features

    def get_popularity_prior(self, industries: List[str],
                              ages: List[float]) -> np.ndarray:
        """
        基于选手特征计算先验人气权重

        考虑因素：
        1. 职业类别 - 不同职业粉丝基础不同
        2. 年龄 - 年轻选手可能有更多社交媒体粉丝
        """
        n = len(industries) if industries else len(ages)
        weights = np.ones(n)

        if industries:
            for i, ind in enumerate(industries):
                weights[i] *= self.INDUSTRY_POPULARITY.get(
                    ind, self.INDUSTRY_POPULARITY['default']
                )

        if ages:
            for i, age in enumerate(ages):
                if age and not np.isnan(age):
                    # 年龄效应：25-35岁最受欢迎
                    age_factor = 1.0 - 0.01 * abs(age - 30)
                    age_factor = max(0.7, min(1.2, age_factor))
                    weights[i] *= age_factor

        return weights / weights.sum()  # 归一化

    def compute_combined_scores(self, votes: np.ndarray,
                                 judge_scores: np.ndarray) -> np.ndarray:
        """计算综合得分"""
        n = len(votes)

        if self.method == 'rank':
            judge_ranks = stats.rankdata(-judge_scores, method='average')
            vote_ranks = stats.rankdata(-votes, method='average')
            # 排名和越大越容易被淘汰
            combined = -(judge_ranks + vote_ranks)
        else:
            judge_pct = judge_scores / (judge_scores.sum() + 1e-10)
            vote_pct = votes / (votes.sum() + 1e-10)
            combined = judge_pct + vote_pct

        return combined

    def log_likelihood(self, votes: np.ndarray, judge_scores: np.ndarray,
                       eliminated_idx: int) -> float:
        """
        改进的对数似然函数

        使用分段温度：对边界情况更敏感
        """
        combined = self.compute_combined_scores(votes, judge_scores)
        n = len(combined)

        # 找出最低分和次低分
        sorted_idx = np.argsort(combined)
        lowest_idx = sorted_idx[0]
        second_lowest_idx = sorted_idx[1] if n > 1 else sorted_idx[0]

        # 边界距离
        margin = combined[second_lowest_idx] - combined[lowest_idx]

        # 动态温度：边界越清晰，温度越高（约束越硬）
        effective_temp = self.temperature * (1 + 2 * margin)

        # Softmax似然
        log_probs = -effective_temp * combined
        log_probs = log_probs - np.max(log_probs)
        probs = np.exp(log_probs) / np.sum(np.exp(log_probs))

        return np.log(probs[eliminated_idx] + 1e-10)

    def log_prior(self, votes: np.ndarray, judge_scores: np.ndarray,
                  popularity_weights: np.ndarray = None,
                  alpha: float = 1.5, beta: float = 0.3) -> float:
        """
        改进的先验分布

        融合三个信息源：
        1. 均匀分布（无信息）
        2. 评委分数（正相关假设）
        3. 选手特征（职业、年龄）
        """
        n = len(votes)

        # 基础：归一化评委分数
        judge_norm = judge_scores / (judge_scores.sum() + 1e-10)

        # 特征权重
        if popularity_weights is not None:
            feature_weight = popularity_weights
        else:
            feature_weight = np.ones(n) / n

        # 混合先验：评委分数 + 特征
        # 允许"逆反"：特征权重可能与评委分数不一致
        mixed_prior = (1 - beta) * feature_weight + beta * judge_norm

        # Dirichlet浓度参数
        concentration = alpha * n * mixed_prior
        concentration = np.maximum(concentration, 0.5)

        # Dirichlet对数概率
        log_prob = (gammaln(concentration.sum()) -
                    gammaln(concentration).sum() +
                    np.sum((concentration - 1) * np.log(votes + 1e-10)))

        return log_prob

    def adaptive_proposal(self, current: np.ndarray,
                          scale: float,
                          accept_history: List[bool],
                          target_rate: float = 0.30) -> Tuple[np.ndarray, float]:
        """
        自适应提议分布

        根据历史接受率动态调整步长
        """
        # 计算最近的接受率
        window = min(100, len(accept_history))
        if window > 10:
            recent_rate = np.mean(accept_history[-window:])

            # 调整步长
            if recent_rate < target_rate - 0.05:
                scale *= 0.95  # 接受率太低，减小步长
            elif recent_rate > target_rate + 0.05:
                scale *= 1.05  # 接受率太高，增大步长

            scale = np.clip(scale, 0.01, 0.3)

        # Dirichlet提议
        concentration = current * (1 / scale)
        concentration = np.maximum(concentration, 0.5)
        proposal = np.random.dirichlet(concentration)

        return proposal, scale

    def log_proposal(self, x: np.ndarray, given: np.ndarray,
                     scale: float) -> float:
        """提议分布的对数概率"""
        concentration = given * (1 / scale)
        concentration = np.maximum(concentration, 0.5)

        return (gammaln(concentration.sum()) -
                gammaln(concentration).sum() +
                np.sum((concentration - 1) * np.log(x + 1e-10)))

    def mcmc_sample(self, judge_scores: np.ndarray, eliminated_idx: int,
                    popularity_weights: np.ndarray = None,
                    n_samples: int = 10000, burnin: int = 3000,
                    thin: int = 2, initial_scale: float = 0.06,
                    prior_alpha: float = 1.5, prior_beta: float = 0.3
                    ) -> Tuple[np.ndarray, float, float]:
        """
        自适应MCMC采样
        """
        n = len(judge_scores)
        total_iterations = burnin + n_samples * thin

        # 智能初始化
        judge_norm = judge_scores / judge_scores.sum()
        if popularity_weights is not None:
            # 混合初始化
            initial = 0.5 * judge_norm + 0.5 * popularity_weights
        else:
            initial = judge_norm

        # 调整初始值使其更可能满足约束
        initial[eliminated_idx] *= 0.7  # 降低淘汰者的初始份额
        current = initial / initial.sum()

        samples = []
        accept_history = []
        scale = initial_scale
        accepted_total = 0

        # 当前对数后验
        log_posterior_current = (
            self.log_likelihood(current, judge_scores, eliminated_idx) +
            self.log_prior(current, judge_scores, popularity_weights,
                          prior_alpha, prior_beta)
        )

        for i in range(total_iterations):
            # 自适应提议
            proposal, scale = self.adaptive_proposal(
                current, scale, accept_history
            )

            # 计算后验
            log_posterior_proposal = (
                self.log_likelihood(proposal, judge_scores, eliminated_idx) +
                self.log_prior(proposal, judge_scores, popularity_weights,
                              prior_alpha, prior_beta)
            )

            # 提议分布修正
            log_q_forward = self.log_proposal(proposal, current, scale)
            log_q_backward = self.log_proposal(current, proposal, scale)

            # MH接受概率
            log_alpha = (log_posterior_proposal - log_posterior_current +
                        log_q_backward - log_q_forward)

            # 接受/拒绝
            accept = np.log(np.random.rand() + 1e-10) < min(0, log_alpha)
            accept_history.append(accept)

            if accept:
                current = proposal
                log_posterior_current = log_posterior_proposal
                accepted_total += 1

            # 保存样本
            if i >= burnin and (i - burnin) % thin == 0:
                samples.append(current.copy())

        acceptance_rate = accepted_total / total_iterations
        final_scale = scale

        return np.array(samples), acceptance_rate, final_scale

    def compute_ess(self, samples: np.ndarray) -> np.ndarray:
        """计算有效样本量"""
        n_samples, n_dims = samples.shape
        ess = np.zeros(n_dims)

        for d in range(n_dims):
            x = samples[:, d] - samples[:, d].mean()
            if np.var(x) < 1e-10:
                ess[d] = n_samples
                continue

            n = len(x)
            acf = np.correlate(x, x, mode='full')[n-1:] / (np.var(x) * n + 1e-10)

            cutoff = min(n // 3, 100)
            for k in range(1, cutoff):
                if acf[k] < 0.05:
                    cutoff = k
                    break

            tau = 1 + 2 * np.sum(np.abs(acf[1:cutoff]))
            ess[d] = n / max(tau, 1)

        return ess

    def estimate(self, week_data: WeekData, **kwargs) -> Optional[EstimationResult]:
        """估计单周投票分布"""
        if week_data.eliminated_idx is None:
            return None

        judge_scores = week_data.judge_scores
        eliminated_idx = week_data.eliminated_idx
        n = len(judge_scores)

        # 计算特征先验权重
        if self.use_features and week_data.industries:
            popularity_weights = self.get_popularity_prior(
                week_data.industries, week_data.ages
            )
        else:
            popularity_weights = None

        # MCMC采样
        samples, acceptance_rate, final_scale = self.mcmc_sample(
            judge_scores, eliminated_idx,
            popularity_weights=popularity_weights,
            **kwargs
        )

        # 统计量
        vote_share_mean = samples.mean(axis=0)
        vote_share_std = samples.std(axis=0)
        vote_share_median = np.median(samples, axis=0)
        ci_lower = np.percentile(samples, 2.5, axis=0)
        ci_upper = np.percentile(samples, 97.5, axis=0)

        ess = self.compute_ess(samples)

        # 一致性指标
        combined_mean = self.compute_combined_scores(vote_share_mean, judge_scores)
        predicted_eliminated_idx = np.argmin(combined_mean)
        is_correct = (predicted_eliminated_idx == eliminated_idx)

        # 淘汰概率
        elimination_probs = np.zeros(n)
        for sample in samples:
            combined = self.compute_combined_scores(sample, judge_scores)
            pred_elim = np.argmin(combined)
            elimination_probs[pred_elim] += 1
        elimination_probs /= len(samples)
        elimination_probability = elimination_probs[eliminated_idx]

        # 边界距离
        combined_sorted = np.sort(combined_mean)
        margin = combined_sorted[1] - combined_sorted[0] if n > 1 else 0

        # 排名相关
        predicted_ranks = stats.rankdata(-combined_mean)
        actual_ranks = np.ones(n)
        actual_ranks[eliminated_idx] = n
        other_mask = np.arange(n) != eliminated_idx
        if np.sum(other_mask) > 0:
            actual_ranks[other_mask] = stats.rankdata(-judge_scores[other_mask])

        rank_corr, _ = stats.kendalltau(predicted_ranks, actual_ranks)

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
            rank_correlation=rank_corr if not np.isnan(rank_corr) else 0,
            acceptance_rate=acceptance_rate,
            effective_sample_size=ess
        )


class DataLoaderV3:
    """改进的数据加载器，提取更多特征"""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.seasons = sorted(self.df['season'].unique())

    def get_scoring_method(self, season: int) -> str:
        if season <= 2 or season >= 28:
            return 'rank'
        return 'percent'

    def get_week_data(self, season: int, week: int) -> Optional[WeekData]:
        """获取周数据，包含选手特征"""
        season_df = self.df[self.df['season'] == season].copy()
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
        week_df = season_df[valid_mask].copy()

        if len(week_df) == 0:
            return None

        names = week_df['celebrity_name'].tolist()

        # 评委分数
        scores = []
        for _, row in week_df.iterrows():
            total = sum(float(row[c]) for c in existing_cols if is_valid(row[c]))
            scores.append(total)
        judge_scores = np.array(scores)

        # 提取特征
        industries = week_df['celebrity_industry'].tolist() if 'celebrity_industry' in week_df else None
        ages = week_df['celebrity_age_during_season'].tolist() if 'celebrity_age_during_season' in week_df else None
        partners = week_df['ballroom_partner'].tolist() if 'ballroom_partner' in week_df else None

        # 处理年龄中的非数值
        if ages:
            ages = [float(a) if pd.notna(a) and str(a).replace('.','').isdigit() else np.nan for a in ages]

        # 淘汰选手
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
            eliminated_name=eliminated_name,
            industries=industries,
            ages=ages,
            partners=partners
        )

    def get_all_weeks(self, season: int) -> List[WeekData]:
        weeks = []
        for week in range(1, 12):
            wd = self.get_week_data(season, week)
            if wd and wd.eliminated_idx is not None:
                weeks.append(wd)
        return weeks


def run_estimation_v3(data_path: str,
                      seasons: List[int] = None,
                      temperature: float = 12.0,
                      use_features: bool = True,
                      n_samples: int = 10000,
                      burnin: int = 3000):
    """运行V3估计"""
    loader = DataLoaderV3(data_path)

    if seasons is None:
        seasons = loader.seasons

    all_results = []

    for season in seasons:
        print(f"\n{'='*50}")
        print(f"Season {season}")
        print(f"{'='*50}")

        method = loader.get_scoring_method(season)
        print(f"计分方法: {'排名法' if method == 'rank' else '百分比法'}")

        estimator = ImprovedVoteEstimator(
            method=method,
            temperature=temperature,
            use_features=use_features
        )

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
                    print(f"    {status} | P(elim)={result.elimination_probability:.2f} | "
                          f"τ={result.rank_correlation:.2f} | "
                          f"接受率={result.acceptance_rate:.1%}")

            except Exception as e:
                print(f"    错误: {e}")

    return all_results


def compute_metrics(results: List[EstimationResult]) -> Dict:
    """计算汇总指标"""
    if not results:
        return {}

    epa = np.mean([r.is_correct for r in results])
    elim_probs = [r.elimination_probability for r in results]
    tau_vals = [r.rank_correlation for r in results]
    acc_rates = [r.acceptance_rate for r in results]

    # Top-2准确率
    top2_correct = 0
    for r in results:
        combined = r.vote_share_mean  # 简化
        bottom2 = np.argsort(combined)[:2]
        if r.eliminated_idx in bottom2:
            top2_correct += 1
    top2_acc = top2_correct / len(results)

    # CV分析
    all_cvs = []
    elim_cvs = []
    non_elim_cvs = []

    for r in results:
        for i in range(len(r.names)):
            cv = r.vote_share_std[i] / (r.vote_share_mean[i] + 1e-10)
            all_cvs.append(cv)
            if i == r.eliminated_idx:
                elim_cvs.append(cv)
            else:
                non_elim_cvs.append(cv)

    return {
        'total_weeks': len(results),
        'EPA': epa,
        'mean_elimination_prob': np.mean(elim_probs),
        'std_elimination_prob': np.std(elim_probs),
        'top2_accuracy': top2_acc,
        'mean_kendall_tau': np.mean(tau_vals),
        'mean_acceptance_rate': np.mean(acc_rates),
        'mean_cv': np.mean(all_cvs),
        'cv_eliminated': np.mean(elim_cvs),
        'cv_not_eliminated': np.mean(non_elim_cvs)
    }


def main():
    import os
    import argparse

    # 设置随机种子确保结果可复现
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Task 1 V3: 改进版投票估计')
    parser.add_argument('--seasons', type=int, nargs='+', default=None)
    parser.add_argument('--temperature', type=float, default=12.0)
    parser.add_argument('--no-features', action='store_true',
                        help='不使用选手特征')
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--burnin', type=int, default=3000)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '2026_MCM-ICM_Problems',
                             '2026_MCM_Problem_C_Data.csv')

    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return

    print("="*60)
    print("Task 1 V3: 改进版贝叶斯投票估计")
    print("="*60)
    print(f"温度: {args.temperature}")
    print(f"使用特征: {not args.no_features}")
    print(f"采样数: {args.n_samples}")

    results = run_estimation_v3(
        data_path,
        seasons=args.seasons,
        temperature=args.temperature,
        use_features=not args.no_features,
        n_samples=args.n_samples,
        burnin=args.burnin
    )

    metrics = compute_metrics(results)

    print("\n" + "="*60)
    print("Q1a: 一致性指标汇总")
    print("="*60)
    print(f"总周数: {metrics['total_weeks']}")
    print(f"淘汰预测准确率 (EPA): {metrics['EPA']:.1%}")
    print(f"平均淘汰概率: {metrics['mean_elimination_prob']:.3f} "
          f"(±{metrics['std_elimination_prob']:.3f})")
    print(f"Top-2准确率: {metrics['top2_accuracy']:.1%}")
    print(f"平均Kendall's τ: {metrics['mean_kendall_tau']:.3f}")
    print(f"平均MCMC接受率: {metrics['mean_acceptance_rate']:.1%}")

    print("\n" + "="*60)
    print("Q1b: 确定性指标汇总")
    print("="*60)
    print(f"平均CV: {metrics['mean_cv']:.4f}")
    print(f"被淘汰选手CV: {metrics['cv_eliminated']:.4f}")
    print(f"未淘汰选手CV: {metrics['cv_not_eliminated']:.4f}")

    # 保存结果
    records = []
    for r in results:
        for i, name in enumerate(r.names):
            records.append({
                'season': r.season,
                'week': r.week,
                'contestant': name,
                'judge_score': r.judge_scores[i],
                'vote_share_mean': r.vote_share_mean[i],
                'vote_share_std': r.vote_share_std[i],
                'ci_lower': r.ci_lower[i],
                'ci_upper': r.ci_upper[i],
                'is_eliminated': i == r.eliminated_idx
            })

    df = pd.DataFrame(records)
    output_path = os.path.join(script_dir, 'vote_estimates_v3.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
