"""
semi_supervised_detector.py
基于事件分布/正常分布的半监督异常检测器
"""

import numpy as np

from distribution_model import DistributionModel


class SemiSupervisedDetector:
    """似然比风险检测器"""

    def __init__(self, robust=True, regularization=1e-4, lr_scale=1.0):
        self.event_model = DistributionModel(robust=robust, regularization=regularization)
        self.normal_model = DistributionModel(robust=robust, regularization=regularization)
        self.feature_names = []
        self.lr_scale = lr_scale
        self.lr_center = 0.0
        self.lr_temperature = 1.0

    def fit(self, event_features, normal_features, feature_names=None):
        event_matrix = np.asarray(event_features, dtype=float)
        normal_matrix = np.asarray(normal_features, dtype=float)
        event_matrix = np.nan_to_num(event_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        normal_matrix = np.nan_to_num(normal_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if event_matrix.ndim != 2 or normal_matrix.ndim != 2:
            raise ValueError("event_features and normal_features must be 2D matrices")
        if event_matrix.shape[1] != normal_matrix.shape[1]:
            raise ValueError("event/normal features must share the same feature dimension")

        self.feature_names = feature_names or [f"f{i}" for i in range(event_matrix.shape[1])]

        self.event_model.fit(event_matrix, feature_names=self.feature_names)
        self.normal_model.fit(normal_matrix, feature_names=self.feature_names)

        # 基于训练样本自动做分数温度缩放，缓解风险指数饱和
        combined = np.vstack([event_matrix, normal_matrix])
        train_lr = self.score_batch(combined)
        self.lr_center = float(np.median(train_lr))
        lr_std = float(np.std(train_lr))
        self.lr_temperature = max(lr_std, 1e-3)
        return self

    def score(self, x):
        """似然比 LR = log P_event - log P_normal"""
        log_event = self.event_model.log_likelihood(x)
        log_normal = self.normal_model.log_likelihood(x)
        lr = log_event - log_normal
        return float(np.nan_to_num(lr, nan=0.0, posinf=1e6, neginf=-1e6))

    def score_batch(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self.score(row) for row in X])

    def risk_index(self, lr_score):
        safe_lr = float(np.nan_to_num(lr_score, nan=0.0, posinf=1e6, neginf=-1e6))
        normalized_lr = (safe_lr - self.lr_center) / self.lr_temperature
        return float(100 * self._sigmoid(self.lr_scale * normalized_lr))

    def predict_label(self, lr_score, threshold=None):
        """LR>0 判定更接近事件分布"""
        if threshold is None:
            threshold = self.lr_center
        return int(lr_score > threshold)

    def assess_vector(self, x):
        log_event = self.event_model.log_likelihood(x)
        log_normal = self.normal_model.log_likelihood(x)
        lr_score = float(np.nan_to_num(log_event - log_normal, nan=0.0, posinf=1e6, neginf=-1e6))

        event_contrib = self.event_model.diagonal_log_likelihood_contrib(x)
        normal_contrib = self.normal_model.diagonal_log_likelihood_contrib(x)

        feature_llr = {
            name: event_contrib[name] - normal_contrib[name]
            for name in self.feature_names
        }

        return {
            'log_event': float(log_event),
            'log_normal': float(log_normal),
            'lr_score': lr_score,
            'risk_index': self.risk_index(lr_score),
            'feature_llr': feature_llr,
        }


    def feature_sensitivity(self, x, epsilon=1e-3):
        """数值微分近似的特征敏感性：d(风险指数)/d(feature)"""
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        base_lr = self.score(x)
        base_risk = self.risk_index(base_lr)

        sensitivities = {}
        for i, name in enumerate(self.feature_names):
            step = epsilon * max(1.0, abs(x[i]))
            x_perturb = x.copy()
            x_perturb[i] += step
            risk_perturb = self.risk_index(self.score(x_perturb))
            sensitivities[name] = (risk_perturb - base_risk) / step

        return sensitivities

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -60, 60)
        return 1 / (1 + np.exp(-x))