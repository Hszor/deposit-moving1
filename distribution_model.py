"""
distribution_model.py
半监督检测的分布建模模块
"""

import numpy as np
from sklearn.covariance import MinCovDet


class DistributionModel:
    """多变量高斯分布模型（支持稳健协方差估计）"""

    def __init__(self, robust=True, regularization=1e-4):
        self.robust = robust
        self.regularization = regularization
        self.mean = None
        self.cov = None
        self.inv_cov = None
        self.log_det_cov = None
        self.feature_names = []

    def fit(self, feature_matrix, feature_names=None):
        """
        Parameters
        ----------
        feature_matrix : array-like, shape (n_samples, n_features)
        feature_names : list[str]
        """
        X = np.asarray(feature_matrix, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if X.ndim != 2:
            raise ValueError("feature_matrix must be 2D")
        if X.shape[0] < 2:
            raise ValueError("at least 2 samples are required to estimate covariance")

        n_samples, n_features = X.shape
        self.feature_names = feature_names or [f"f{i}" for i in range(n_features)]

        self.mean = np.median(X, axis=0) if self.robust else np.mean(X, axis=0)

        cov_estimated = None
        if self.robust and n_samples >= max(4, n_features + 1):
            try:
                mcd = MinCovDet().fit(X)
                cov_estimated = mcd.covariance_
                self.mean = mcd.location_
            except (ValueError, np.linalg.LinAlgError):
                cov_estimated = None

        if cov_estimated is None:
            cov_estimated = np.cov(X, rowvar=False)

        if cov_estimated.ndim == 0:
            cov_estimated = np.array([[float(cov_estimated)]])

        cov_estimated = np.nan_to_num(cov_estimated, nan=0.0, posinf=0.0, neginf=0.0)
        cov_estimated = cov_estimated + np.eye(cov_estimated.shape[0]) * self.regularization

        self.cov = cov_estimated
        self.inv_cov = np.linalg.pinv(self.cov)

        sign, log_det = np.linalg.slogdet(self.cov)
        if sign <= 0:
            self.cov = self.cov + np.eye(self.cov.shape[0]) * (self.regularization * 10)
            self.inv_cov = np.linalg.pinv(self.cov)
            _, log_det = np.linalg.slogdet(self.cov)

        if not np.isfinite(log_det):
            log_det = 0.0
        self.log_det_cov = float(log_det)
        return self

    def mahalanobis_distance(self, x):
        """返回马氏距离"""
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        diff = x - self.mean
        dist2 = float(diff.T @ self.inv_cov @ diff)
        return np.sqrt(max(dist2, 0.0))

    def log_likelihood(self, x):
        """返回不含常数项的对数似然"""
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        diff = x - self.mean
        quad = float(diff.T @ self.inv_cov @ diff)
        n_features = len(diff)
        ll = -0.5 * (quad + self.log_det_cov + n_features * np.log(2 * np.pi))
        return float(np.nan_to_num(ll, nan=-1e6, posinf=1e6, neginf=-1e6))

    def diagonal_log_likelihood_contrib(self, x):
        """基于对角协方差近似的特征级对数似然贡献（用于解释）"""
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        variances = np.diag(self.cov)
        safe_var = np.where(variances <= 1e-10, 1e-10, variances)

        contrib = {}
        for i, feature_name in enumerate(self.feature_names):
            diff2 = (x[i] - self.mean[i]) ** 2
            contrib[feature_name] = -0.5 * (diff2 / safe_var[i] + np.log(safe_var[i]))
        return contrib
