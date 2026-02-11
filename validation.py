"""
validation.py
模型验证模块 - 基于似然比（LR）的半监督验证
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt

from feature_engine import EventProfileBuilder
from semi_supervised_detector import SemiSupervisedDetector


class CrossWindowValidator:
    """交叉窗口验证器（正负窗口联合评估）"""

    def __init__(self, feature_engine, detector_class=SemiSupervisedDetector):
        self.feature_engine = feature_engine
        self.detector_class = detector_class
        self.validation_results = []

    def _extract_feature_frame(self, window_data_dict):
        builder = EventProfileBuilder(self.feature_engine)
        feature_df = builder.fit(window_data_dict)
        return feature_df

    def leave_one_window_out(self, event_window_data, normal_window_data):
        """
        对事件窗口和正常窗口分别进行LOO验证。

        事件窗口：每次留一事件窗口作为测试，剩余事件+全部正常训练。
        正常窗口：每次留一正常窗口作为测试，全部事件+剩余正常训练。
        """
        if not event_window_data or not normal_window_data:
            raise ValueError("event_window_data and normal_window_data are both required")

        results = []

        event_names = list(event_window_data.keys())
        normal_names = list(normal_window_data.keys())

        # 事件窗口LOO
        for test_name in event_names:
            train_event = {k: v for k, v in event_window_data.items() if k != test_name}
            train_normal = normal_window_data

            if len(train_event) < 2 or len(train_normal) < 2:
                continue

            detector, feature_names = self._fit_detector(train_event, train_normal)
            feature_vector = self._extract_single_window_vector(event_window_data[test_name], feature_names)
            assess = detector.assess_vector(feature_vector)
            lr = assess['lr_score']

            results.append({
                'test_window': test_name,
                'window_type': 'event',
                'true_label': 1,
                'predicted_label': detector.predict_label(lr),
                'lr_score': lr,
                'risk_index': assess['risk_index'],
                'log_event': assess['log_event'],
                'log_normal': assess['log_normal'],
            })

        # 正常窗口LOO
        for test_name in normal_names:
            train_event = event_window_data
            train_normal = {k: v for k, v in normal_window_data.items() if k != test_name}

            if len(train_event) < 2 or len(train_normal) < 2:
                continue

            detector, feature_names = self._fit_detector(train_event, train_normal)
            feature_vector = self._extract_single_window_vector(normal_window_data[test_name], feature_names)
            assess = detector.assess_vector(feature_vector)
            lr = assess['lr_score']

            results.append({
                'test_window': test_name,
                'window_type': 'normal',
                'true_label': 0,
                'predicted_label': detector.predict_label(lr),
                'lr_score': lr,
                'risk_index': assess['risk_index'],
                'log_event': assess['log_event'],
                'log_normal': assess['log_normal'],
            })

        self.validation_results = results
        return results

    def _fit_detector(self, train_event, train_normal):
        event_df = self._extract_feature_frame(train_event)
        normal_df = self._extract_feature_frame(train_normal)

        feature_names = sorted(set(event_df.columns).union(normal_df.columns))
        event_df = event_df.reindex(columns=feature_names, fill_value=0)
        normal_df = normal_df.reindex(columns=feature_names, fill_value=0)

        detector = self.detector_class()
        detector.fit(event_df.values, normal_df.values, feature_names=feature_names)
        return detector, feature_names

    def _extract_single_window_vector(self, window_series_dict, feature_names):
        feature_dict = self.feature_engine.extract_all_features(window_series_dict)
        feature_dict.update(self.feature_engine.calculate_cross_features(window_series_dict))
        vector = np.array([feature_dict.get(name, 0) for name in feature_names], dtype=float)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    def calculate_validation_metrics(self, results):
        if not results:
            return {}

        y_true = np.array([r['true_label'] for r in results])
        y_pred = np.array([r['predicted_label'] for r in results])
        y_score = np.array([r['lr_score'] for r in results], dtype=float)
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=1e6, neginf=-1e6)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        if len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_score)
                pr_auc = average_precision_score(y_true, y_score)
            except ValueError:
                auc, pr_auc = 0.5, 0.5
        else:
            auc, pr_auc = 0.5, 0.5

        auc_ci = self.bootstrap_auc_ci(y_true, y_score, n_bootstrap=200)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'pr_auc': float(pr_auc),
            'auc_ci_low': float(auc_ci[0]),
            'auc_ci_high': float(auc_ci[1]),
            'correct_rate': float(np.mean(y_true == y_pred)),
            'avg_lr_score': float(np.mean(y_score)),
            'std_lr_score': float(np.std(y_score)),
        }


    def bootstrap_auc_ci(self, y_true, y_score, n_bootstrap=200, alpha=0.1):
        """Bootstrap AUC置信区间"""
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if len(np.unique(y_true)) < 2:
            return 0.5, 0.5

        rng = np.random.default_rng(42)
        auc_samples = []
        n = len(y_true)

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            try:
                auc_samples.append(roc_auc_score(y_true[idx], y_score[idx]))
            except ValueError:
                continue

        if not auc_samples:
            return 0.5, 0.5

        low = np.percentile(auc_samples, alpha / 2 * 100)
        high = np.percentile(auc_samples, (1 - alpha / 2) * 100)
        return low, high

    def rolling_time_backtest(self, event_window_data, normal_window_data):
        """时间滚动回测：逐步扩展训练集，评估后续窗口"""
        combined = []
        for name, data in event_window_data.items():
            combined.append((name, 1, data))
        for name, data in normal_window_data.items():
            combined.append((name, 0, data))

        # 按名称中的年份排序（取前4位数字）
        def _year(x):
            import re
            m = re.search(r'(19|20)\d{2}', x[0])
            return int(m.group()) if m else 9999

        combined.sort(key=_year)
        if len(combined) < 4:
            return []

        results = []
        for t in range(3, len(combined)):
            train = combined[:t]
            test = combined[t]

            train_event = {n: d for n, y, d in train if y == 1}
            train_normal = {n: d for n, y, d in train if y == 0}
            if len(train_event) < 2 or len(train_normal) < 2:
                continue

            detector, feature_names = self._fit_detector(train_event, train_normal)
            test_vector = self._extract_single_window_vector(test[2], feature_names)
            assess = detector.assess_vector(test_vector)
            lr = assess['lr_score']

            results.append({
                'test_window': test[0],
                'true_label': test[1],
                'predicted_label': detector.predict_label(lr),
                'lr_score': lr,
                'risk_index': assess['risk_index']
            })

        return results

    def plot_validation_results(self, results, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        windows = [r['test_window'] for r in results]
        lr_scores = [r['lr_score'] for r in results]
        risk_indices = [r['risk_index'] for r in results]
        labels = [r['true_label'] for r in results]

        # 1) LR条形图
        ax1 = axes[0, 0]
        colors = ['#E74C3C' if y == 1 else '#2ECC71' for y in labels]
        bars = ax1.bar(range(len(windows)), lr_scores, color=colors, alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.6, label='似然比=0 分界')
        ax1.set_xticks(range(len(windows)))
        ax1.set_xticklabels(windows, rotation=45, ha='right')
        ax1.set_ylabel('似然比分数')
        ax1.set_title('窗口似然比分数（红=事件，绿=正常）')
        ax1.grid(True, alpha=0.3)

        for bar, lr in zip(bars, lr_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{lr:.2f}',
                     ha='center', va='bottom' if lr >= 0 else 'top', fontsize=8)

        # 2) 风险指数散点图
        ax2 = axes[0, 1]
        x = np.arange(len(results))
        ax2.scatter(x, risk_indices, c=colors, s=80, edgecolor='black', alpha=0.8)
        ax2.axhline(50, color='orange', linestyle='--', alpha=0.6, label='风险中线')
        ax2.set_xticks(x)
        ax2.set_xticklabels(windows, rotation=45, ha='right')
        ax2.set_ylabel('风险指数')
        ax2.set_title('窗口风险指数分布')
        ax2.grid(True, alpha=0.3)

        # 3) 事件/正常窗口似然比分布箱线图
        ax3 = axes[1, 0]
        event_lr = [r['lr_score'] for r in results if r['true_label'] == 1]
        normal_lr = [r['lr_score'] for r in results if r['true_label'] == 0]
        bp = ax3.boxplot([event_lr, normal_lr], labels=['事件窗口', '正常窗口'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#FADBD8')
        bp['boxes'][1].set_facecolor('#D5F5E3')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.6)
        ax3.set_ylabel('似然比分数')
        ax3.set_title('事件/正常窗口似然比分布对比')
        ax3.grid(True, alpha=0.3)

        # 4) 对数似然对比
        ax4 = axes[1, 1]
        log_event = [r['log_event'] for r in results]
        log_normal = [r['log_normal'] for r in results]
        ax4.scatter(log_normal, log_event, c=colors, s=80, edgecolor='black', alpha=0.8)
        min_axis = min(min(log_normal), min(log_event))
        max_axis = max(max(log_normal), max(log_event))
        ax4.plot([min_axis, max_axis], [min_axis, max_axis], 'k--', alpha=0.5)
        ax4.set_xlabel('正常分布对数似然')
        ax4.set_ylabel('事件分布对数似然')
        ax4.set_title('事件/正常对数似然对比')
        ax4.grid(True, alpha=0.3)

        plt.suptitle('半监督异常检测验证结果', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        return fig

    def generate_validation_report(self, results, metrics, output_path=None):
        report_lines = [
            "=" * 80,
            "模型验证报告 - 半监督异常检测（似然比）",
            "=" * 80,
            "\n📊 验证指标汇总",
            "-" * 40,
            f"精确率 (Precision): {metrics.get('precision', 0):.3f}",
            f"召回率 (Recall): {metrics.get('recall', 0):.3f}",
            f"F1分数: {metrics.get('f1_score', 0):.3f}",
            f"ROC AUC: {metrics.get('auc', 0):.3f}",
            f"ROC AUC置信区间(90%): [{metrics.get('auc_ci_low', 0):.3f}, {metrics.get('auc_ci_high', 0):.3f}]",
            f"PR AUC: {metrics.get('pr_auc', 0):.3f}",
            f"正确识别率: {metrics.get('correct_rate', 0):.1%}",
            f"似然比分数均值: {metrics.get('avg_lr_score', 0):.3f}",
            f"似然比分数标准差: {metrics.get('std_lr_score', 0):.3f}",
            "\n📋 窗口级结果",
            "-" * 40,
        ]

        for r in results:
            label_name = '事件' if r['true_label'] == 1 else '正常'
            pred_name = '事件' if r['predicted_label'] == 1 else '正常'
            report_lines.append(
                f"{r['test_window']} ({label_name}): 似然比={r['lr_score']:.3f}, "
                f"Risk={r['risk_index']:.1f}, 预测={pred_name}"
            )

        if metrics.get('auc', 0) >= 0.85:
            quality = '优秀'
        elif metrics.get('auc', 0) >= 0.7:
            quality = '良好'
        elif metrics.get('auc', 0) >= 0.6:
            quality = '一般'
        else:
            quality = '需改进'

        report_lines.extend([
            "\n🔧 模型判别能力",
            "-" * 40,
            f"基于ROC AUC的综合评价: {quality}",
            "说明: 该评价基于事件窗口与正常窗口的真实对照分类能力。",
        ])

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"验证报告已保存到: {output_path}")

        return report_text
