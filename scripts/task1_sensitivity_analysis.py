"""
Task 1 补充: 超参数敏感性分析

分析模型输出对各超参数变化的稳定性
证明模型的鲁棒性

使用方法:
    python task1_sensitivity_analysis.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
import sys

# 导入主模型 (V3版本)
from task1_vote_estimator_v3 import (
    ImprovedVoteEstimator as SoftConstraintVoteEstimator,
    DataLoaderV3 as DataLoader,
    WeekData,
)


class SensitivityAnalyzer:
    """超参数敏感性分析器 (适配V3模型)"""

    def __init__(self, data_path: str):
        self.loader = DataLoader(data_path)
        # V3模型的默认参数
        self.base_params = {
            'temperature': 12.0,      # V3默认温度
            'prior_alpha': 1.5,       # V3默认先验强度
            'prior_beta': 0.3,        # V3默认先验关联
            'initial_scale': 0.06,    # V3使用initial_scale而非proposal_scale
            'n_samples': 5000,
            'burnin': 1000
        }

    def run_with_params(self, seasons: List[int],
                        params: Dict) -> Tuple[float, float, float]:
        """
        使用指定参数运行模型，返回关键指标

        Returns:
            (EPA, mean_elimination_prob, mean_cv)
        """
        results = []

        for season in seasons:
            method = self.loader.get_scoring_method(season)
            estimator = SoftConstraintVoteEstimator(
                method=method,
                temperature=params['temperature']
            )

            weeks = self.loader.get_all_weeks(season)

            for week_data in weeks:
                if week_data.eliminated_idx is None:
                    continue

                try:
                    # V3的estimate方法参数
                    result = estimator.estimate(
                        week_data,
                        n_samples=params['n_samples'],
                        burnin=params['burnin'],
                        initial_scale=params['initial_scale'],
                        prior_alpha=params['prior_alpha'],
                        prior_beta=params['prior_beta']
                    )
                    if result:
                        results.append(result)
                except:
                    continue

        if not results:
            return 0, 0, 0

        # 计算指标
        epa = np.mean([r.is_correct for r in results])
        mean_elim_prob = np.mean([r.elimination_probability for r in results])

        # 计算平均CV
        all_cvs = []
        for r in results:
            cvs = r.vote_share_std / (r.vote_share_mean + 1e-10)
            all_cvs.extend(cvs)
        mean_cv = np.mean(all_cvs)

        return epa, mean_elim_prob, mean_cv

    def analyze_single_param(self, param_name: str,
                             param_values: List[float],
                             seasons: List[int] = [1, 2, 3, 4, 5]) -> pd.DataFrame:
        """
        分析单个超参数的影响

        Args:
            param_name: 参数名称
            param_values: 要测试的参数值列表
            seasons: 用于测试的赛季

        Returns:
            DataFrame with results
        """
        records = []

        for value in param_values:
            print(f"  测试 {param_name} = {value}")

            # 复制基础参数并修改目标参数
            params = self.base_params.copy()
            params[param_name] = value

            epa, elim_prob, cv = self.run_with_params(seasons, params)

            records.append({
                'param_name': param_name,
                'param_value': value,
                'EPA': epa,
                'elimination_probability': elim_prob,
                'mean_CV': cv
            })

        return pd.DataFrame(records)

    def full_sensitivity_analysis(self,
                                   seasons: List[int] = [1, 2, 3, 4, 5]
                                   ) -> Dict[str, pd.DataFrame]:
        """
        完整的敏感性分析

        测试所有关键超参数
        """
        results = {}

        # 1. 温度参数
        print("\n分析 temperature...")
        results['temperature'] = self.analyze_single_param(
            'temperature',
            [3, 5, 8, 10, 15, 20, 30],
            seasons
        )

        # 2. 先验强度
        print("\n分析 prior_alpha...")
        results['prior_alpha'] = self.analyze_single_param(
            'prior_alpha',
            [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
            seasons
        )

        # 3. 先验关联
        print("\n分析 prior_beta...")
        results['prior_beta'] = self.analyze_single_param(
            'prior_beta',
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
            seasons
        )

        # 4. 提议步长
        print("\n分析 initial_scale...")
        results['initial_scale'] = self.analyze_single_param(
            'initial_scale',
            [0.02, 0.04, 0.06, 0.08, 0.10, 0.15],
            seasons
        )

        return results

    def compute_stability_metrics(self,
                                   sensitivity_results: Dict[str, pd.DataFrame]
                                   ) -> pd.DataFrame:
        """
        计算稳定性指标

        对于每个超参数，计算EPA的变化范围和标准差
        """
        records = []

        for param_name, df in sensitivity_results.items():
            epa_values = df['EPA'].values
            cv_values = df['mean_CV'].values

            records.append({
                'parameter': param_name,
                'EPA_mean': np.mean(epa_values),
                'EPA_std': np.std(epa_values),
                'EPA_range': np.max(epa_values) - np.min(epa_values),
                'EPA_min': np.min(epa_values),
                'EPA_max': np.max(epa_values),
                'CV_mean': np.mean(cv_values),
                'CV_std': np.std(cv_values),
                'CV_range': np.max(cv_values) - np.min(cv_values)
            })

        return pd.DataFrame(records)


def plot_sensitivity(results: Dict[str, pd.DataFrame], output_path: str):
    """绘制敏感性分析图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib未安装，跳过可视化")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    param_names = ['temperature', 'prior_alpha', 'prior_beta', 'initial_scale']
    titles = ['Temperature (τ)', 'Prior Strength (α)',
              'Prior Correlation (β)', 'Initial Scale (λ₀)']

    for ax, param_name, title in zip(axes.flat, param_names, titles):
        if param_name not in results:
            continue

        df = results[param_name]

        # EPA曲线
        ax.plot(df['param_value'], df['EPA'], 'o-',
                color='steelblue', linewidth=2, markersize=8, label='EPA')

        # 添加默认值标记（V3默认值）
        default_values = {
            'temperature': 12.0,
            'prior_alpha': 1.5,
            'prior_beta': 0.3,
            'initial_scale': 0.06
        }

        ax.axvline(x=default_values[param_name], color='red',
                   linestyle='--', alpha=0.7, label='Default')

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('EPA (Elimination Prediction Accuracy)', fontsize=10)
        ax.set_title(f'Sensitivity to {title}', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"敏感性分析图已保存: {output_path}")


def plot_stability_summary(stability_df: pd.DataFrame, output_path: str):
    """绘制稳定性汇总图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    params = stability_df['parameter']
    epa_means = stability_df['EPA_mean']
    epa_ranges = stability_df['EPA_range']

    x = np.arange(len(params))

    # 条形图：EPA范围（稳定性指标）
    bars = ax.bar(x, epa_ranges, color='steelblue', alpha=0.7)

    # 添加数值标签
    for bar, val in zip(bars, epa_ranges):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(['Temperature\n(τ)', 'Prior Strength\n(α)',
                        'Prior Correlation\n(β)', 'Initial Scale\n(λ₀)'],
                       fontsize=10)
    ax.set_ylabel('EPA Range (Max - Min)', fontsize=11)
    ax.set_title('Model Stability: EPA Variation Across Hyperparameter Ranges',
                 fontsize=12)
    ax.set_ylim(0, max(epa_ranges) * 1.3)

    # 添加稳定性判断线
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.7,
               label='High Stability Threshold')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7,
               label='Medium Stability Threshold')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"稳定性汇总图已保存: {output_path}")


def main():
    import argparse

    # 设置随机种子确保结果可复现
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='超参数敏感性分析')
    parser.add_argument('--seasons', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='用于分析的赛季')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式（减少参数值数量）')

    args = parser.parse_args()

    # 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '2026_MCM-ICM_Problems',
                             '2026_MCM_Problem_C_Data.csv')

    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return

    print("="*60)
    print("Task 1: 超参数敏感性分析")
    print("="*60)
    print(f"分析赛季: {args.seasons}")

    analyzer = SensitivityAnalyzer(data_path)

    # 运行敏感性分析
    results = analyzer.full_sensitivity_analysis(args.seasons)

    # 计算稳定性指标
    stability = analyzer.compute_stability_metrics(results)

    # 输出结果
    print("\n" + "="*60)
    print("敏感性分析结果")
    print("="*60)

    for param_name, df in results.items():
        print(f"\n{param_name}:")
        print(df.to_string(index=False))

    print("\n" + "="*60)
    print("稳定性汇总")
    print("="*60)
    print(stability.to_string(index=False))

    # 判断稳定性
    print("\n" + "="*60)
    print("稳定性评估")
    print("="*60)

    for _, row in stability.iterrows():
        param = row['parameter']
        epa_range = row['EPA_range']

        if epa_range < 0.1:
            level = "高稳定性 ✓"
        elif epa_range < 0.2:
            level = "中等稳定性"
        else:
            level = "低稳定性 ⚠"

        print(f"{param:20s}: EPA变化范围 = {epa_range:.3f} → {level}")

    # 保存结果
    all_results = pd.concat(results.values(), ignore_index=True)
    all_results.to_csv(os.path.join(script_dir, 'sensitivity_analysis.csv'),
                       index=False, encoding='utf-8-sig')

    stability.to_csv(os.path.join(script_dir, 'stability_summary.csv'),
                     index=False, encoding='utf-8-sig')

    # 可视化
    plot_sensitivity(results, os.path.join(script_dir, 'sensitivity_curves.png'))
    plot_stability_summary(stability, os.path.join(script_dir, 'stability_summary.png'))

    print(f"\n结果已保存到 {script_dir}")


if __name__ == '__main__':
    main()
