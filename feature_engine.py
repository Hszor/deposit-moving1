"""
feature_engine.py
高级特征工程引擎 - 基于窗口的结构特征提取
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.covariance import MinCovDet
import warnings

warnings.filterwarnings('ignore')


class WindowFeatureEngine:
    """
    窗口特征工程引擎
    提取水平、动态、形态三层特征
    """

    def __init__(self):
        # 特征配置
        self.feature_config = {
            'level_features': ['mean', 'std', 'min', 'max', 'p25', 'p50', 'p75'],
            'structure_features': ['diff_mean', 'diff_std', 'diff_max',
                                   'diff2_mean', 'diff2_max',
                                   'peak_value', 'peak_position', 'time_to_peak',
                                   'slope', 'trend_strength'],
            'shape_features': ['skewness', 'kurtosis', 'autocorr_1', 'autocorr_2',
                               'max_drawdown', 'up_ratio', 'vol_clustering']
        }

    def extract_level_features(self, series):
        """水平特征：静态统计量"""
        features = {}

        if len(series) < 2:
            return {f: 0 for f in self.feature_config['level_features']}

        # 基本统计量
        features['mean'] = np.mean(series)
        features['std'] = np.std(series) if len(series) > 1 else 0
        features['min'] = np.min(series)
        features['max'] = np.max(series)

        # 分位数
        if len(series) >= 4:
            features['p25'] = np.percentile(series, 25)
            features['p50'] = np.percentile(series, 50)
            features['p75'] = np.percentile(series, 75)
        else:
            features['p25'] = features['p50'] = features['p75'] = np.median(series)

        return features

    def extract_structure_features(self, series):
        """结构特征：动态变化模式"""
        features = {}

        if len(series) < 3:
            return {f: 0 for f in self.feature_config['structure_features']}

        # 一阶差分（速度）
        diff = np.diff(series)
        features['diff_mean'] = np.mean(diff)
        features['diff_std'] = np.std(diff) if len(diff) > 1 else 0
        features['diff_max'] = np.max(np.abs(diff))

        # 二阶差分（加速度）
        if len(series) >= 3:
            diff2 = np.diff(diff)
            features['diff2_mean'] = np.mean(diff2) if len(diff2) > 0 else 0
            features['diff2_max'] = np.max(np.abs(diff2)) if len(diff2) > 0 else 0

        # 峰值特征
        if len(series) >= 3:
            peaks, _ = signal.find_peaks(series, prominence=np.std(series) * 0.5)
            if len(peaks) > 0:
                features['peak_value'] = np.max(series[peaks])
                features['peak_position'] = peaks[np.argmax(series[peaks])] / len(series)
            else:
                features['peak_value'] = np.max(series)
                features['peak_position'] = np.argmax(series) / len(series)

        # 时间到峰值（启动速度）
        if 'peak_position' in features:
            features['time_to_peak'] = features['peak_position']

        # 趋势强度（线性回归斜率）
        if len(series) >= 2:
            x = np.arange(len(series))
            slope, intercept = np.polyfit(x, series, 1)
            features['slope'] = slope

            # 趋势强度：R²
            y_pred = slope * x + intercept
            ss_res = np.sum((series - y_pred) ** 2)
            ss_tot = np.sum((series - np.mean(series)) ** 2)
            features['trend_strength'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return features

    def extract_shape_features(self, series):
        """形态特征：分布形状"""
        features = {}

        if len(series) < 4:
            return {f: 0 for f in self.feature_config['shape_features']}

        # 偏度（不对称性）
        features['skewness'] = stats.skew(series) if len(series) > 2 else 0

        # 峰度（尾部厚度）
        features['kurtosis'] = stats.kurtosis(series) if len(series) > 3 else 0

        # 自相关性（记忆效应）
        if len(series) >= 3:
            autocorr = np.correlate(series - np.mean(series),
                                    series - np.mean(series), mode='full')
            autocorr = autocorr[autocorr.size // 2:] / autocorr[autocorr.size // 2]
            features['autocorr_1'] = autocorr[1] if len(autocorr) > 1 else 0
            features['autocorr_2'] = autocorr[2] if len(autocorr) > 2 else 0

        # 最大回撤
        if len(series) >= 2:
            cummax = np.maximum.accumulate(series)
            safe_cummax = np.where(np.abs(cummax) < 1e-10, np.nan, cummax)
            drawdown = (series - safe_cummax) / safe_cummax
            drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
            features['max_drawdown'] = np.min(drawdown) if len(drawdown) > 0 else 0

        # 上涨天数占比
        returns = np.diff(series) if len(series) > 1 else [0]
        features['up_ratio'] = np.sum(np.array(returns) > 0) / len(returns) if len(returns) > 0 else 0

        # 波动聚集性（GARCH思想简化版）
        if len(series) >= 5:
            rolling_std = pd.Series(series).rolling(3, min_periods=2).std().dropna()
            if len(rolling_std) >= 2:
                vol_corr = np.corrcoef(rolling_std.values[:-1],
                                       rolling_std.values[1:])[0, 1]
                features['vol_clustering'] = 0 if np.isnan(vol_corr) else vol_corr
            else:
                features['vol_clustering'] = 0

        return features

    def extract_all_features(self, series_dict):
        """
        提取所有特征

        Parameters:
        -----------
        series_dict : dict
            包含多个指标的字典，如 {'growth_gap': series1, 'maturity_rate': series2}
        """
        all_features = {}

        for indicator, series in series_dict.items():
            # 水平特征
            level = self.extract_level_features(series)
            for k, v in level.items():
                all_features[f'{indicator}_level_{k}'] = v

            # 结构特征
            structure = self.extract_structure_features(series)
            for k, v in structure.items():
                all_features[f'{indicator}_structure_{k}'] = v

            # 形态特征
            shape = self.extract_shape_features(series)
            for k, v in shape.items():
                all_features[f'{indicator}_shape_{k}'] = v

        return all_features

    def calculate_cross_features(self, series_dict):
        """
        计算跨指标特征（指标间的相关性和领先滞后关系）
        """
        features = {}
        indicators = list(series_dict.keys())

        if len(indicators) < 2:
            return features

        # 计算两两相关性
        for i in range(len(indicators)):
            for j in range(i + 1, len(indicators)):
                name_i = indicators[i]
                name_j = indicators[j]

                if len(series_dict[name_i]) == len(series_dict[name_j]) and len(series_dict[name_i]) > 1:
                    corr = np.corrcoef(series_dict[name_i], series_dict[name_j])[0, 1]
                    features[f'corr_{name_i}_{name_j}'] = corr if not np.isnan(corr) else 0

                    # 领先滞后相关性（i领先j一个周期）
                    if len(series_dict[name_i]) > 2:
                        corr_lead = np.corrcoef(series_dict[name_i][:-1],
                                                series_dict[name_j][1:])[0, 1]
                        features[f'lead_corr_{name_i}_{name_j}'] = corr_lead if not np.isnan(corr_lead) else 0

        return features


class EventProfileBuilder:
    """
    事件原型构建器
    基于已知窗口构建特征原型
    """

    def __init__(self, feature_engine):
        self.feature_engine = feature_engine
        self.event_features = []  # 所有事件窗口的特征
        self.event_names = []  # 事件名称
        self.feature_names = []  # 特征名称
        self.profile = {}  # 原型统计量

    def fit(self, window_data_dict, window_names=None):
        """
        从已知窗口构建事件原型

        Parameters:
        -----------
        window_data_dict : dict
            键：窗口名称，值：字典，包含各个指标的序列
        window_names : list, optional
            窗口名称列表，用于标记正样本
        """
        self.event_features = []
        self.event_names = []

        for name, data_dict in window_data_dict.items():
            # 提取特征
            features = self.feature_engine.extract_all_features(data_dict)

            # 添加跨指标特征
            cross_features = self.feature_engine.calculate_cross_features(data_dict)
            features.update(cross_features)

            self.event_features.append(features)
            self.event_names.append(name)

        # 转换为DataFrame
        self.feature_df = pd.DataFrame(self.event_features)
        self.feature_names = self.feature_df.columns.tolist()

        # 计算原型统计量
        self._build_profile()

        return self.feature_df

    def _build_profile(self):
        """构建特征原型"""
        if self.feature_df.empty:
            return

        # 使用稳健估计（抗异常值）
        self.profile = {
            'mean': self.feature_df.median(axis=0).to_dict(),  # 使用中位数而非均值
            'std': self.feature_df.apply(lambda x: np.median(np.abs(x - np.median(x))), axis=0).to_dict(),  # 使用中位数绝对偏差 # 使用中位数绝对偏差
            'min': self.feature_df.min(axis=0).to_dict(),
            'max': self.feature_df.max(axis=0).to_dict(),
            'cov_matrix': None  # 不直接存储协方差矩阵
        }

        # 使用最小协方差行列式估计协方差（更稳健）
        try:
            if len(self.event_features) > 5:
                X = self.feature_df.values
                robust_cov = MinCovDet().fit(X)
                self.profile['cov_matrix'] = robust_cov.covariance_
                self.profile['precision_matrix'] = robust_cov.precision_
        except (ValueError, np.linalg.LinAlgError):
            # 如果稳健估计失败，使用样本协方差
            if len(self.event_features) > 1:
                self.profile['cov_matrix'] = self.feature_df.cov().values

        return self.profile

    def get_feature_importance(self):
        """
        计算特征重要性（基于事件窗口的变异系数）
        变异系数越小，特征越稳定，重要性越高
        """
        if self.feature_df.empty:
            return {}

        # 计算变异系数（标准差/均值）
        cv = (self.feature_df.std() / self.feature_df.mean().abs()).replace([np.inf, -np.inf], 0).fillna(0)

        # 转换为重要性分数（变异系数越小，重要性越高）
        importance = 1 / (cv + 1e-10)
        importance = importance / importance.sum()  # 归一化

        return importance.to_dict()

    def get_prototype_vector(self, method='median'):
        """获取原型特征向量"""
        if method == 'median':
            return self.feature_df.median(axis=0).values
        elif method == 'mean':
            return self.feature_df.mean(axis=0).values
        else:
            raise ValueError(f"Unknown method: {method}")


class SimilarityScorer:
    """
    相似度评分器
    三层评分体系：水平、结构、形态
    """

    def __init__(self, event_profile, feature_weights=None):
        self.profile = event_profile
        self.feature_weights = feature_weights or self._calculate_default_weights()

    def _calculate_default_weights(self):
        """计算默认特征权重"""
        weights = {}

        # 根据不同特征类型分配权重
        for feature in self.profile['mean'].keys():
            if '_level_' in feature:
                weights[feature] = 0.3  # 水平特征
            elif '_structure_' in feature:
                weights[feature] = 0.5  # 结构特征（最重要）
            elif '_shape_' in feature:
                weights[feature] = 0.2  # 形态特征
            else:
                weights[feature] = 0.3  # 其他特征（如相关性）

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def calculate_robust_z_score(self, feature_vector, feature_names):
        """
        计算稳健Z分数（使用中位数和MAD）
        避免异常值影响
        """
        z_scores = {}

        for i, feature_name in enumerate(feature_names):
            if feature_name in self.profile['mean']:
                value = feature_vector[i]
                median = self.profile['mean'][feature_name]
                mad = self.profile['std'][feature_name]

                # 使用稳健Z分数
                if mad > 0:
                    z = abs(value - median) / (1.4826 * mad)  # 1.4826是正态分布下MAD到std的转换
                else:
                    z = abs(value - median) if median != 0 else 0

                z_scores[feature_name] = z

        return z_scores

    def score_level_features(self, z_scores):
        """水平特征评分"""
        level_features = [f for f in z_scores.keys() if '_level_' in f]
        if not level_features:
            return 0

        # 加权平均Z分数
        weighted_sum = 0
        weight_sum = 0

        for feature in level_features:
            weight = self.feature_weights.get(feature, 0)
            weighted_sum += z_scores[feature] * weight
            weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else 0

    def score_structure_features(self, z_scores):
        """结构特征评分"""
        structure_features = [f for f in z_scores.keys() if '_structure_' in f]
        if not structure_features:
            return 0

        # 关键结构特征额外加权
        key_structure_features = ['_slope', '_diff_', '_peak_', '_trend_']

        weighted_sum = 0
        weight_sum = 0

        for feature in structure_features:
            base_weight = self.feature_weights.get(feature, 0)

            # 关键结构特征加倍权重
            multiplier = 1
            for key in key_structure_features:
                if key in feature:
                    multiplier = 2
                    break

            weight = base_weight * multiplier
            weighted_sum += z_scores[feature] * weight
            weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else 0

    def score_shape_features(self, z_scores):
        """形态特征评分"""
        shape_features = [f for f in z_scores.keys() if '_shape_' in f]
        if not shape_features:
            return 0

        weighted_sum = 0
        weight_sum = 0

        for feature in shape_features:
            weight = self.feature_weights.get(feature, 0)
            weighted_sum += z_scores[feature] * weight
            weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else 0

    def score_cross_features(self, z_scores):
        """跨指标特征评分"""
        cross_features = [f for f in z_scores.keys() if f.startswith('corr_') or f.startswith('lead_')]
        if not cross_features:
            return 0

        weighted_sum = 0
        weight_sum = 0

        for feature in cross_features:
            weight = self.feature_weights.get(feature, 0)
            weighted_sum += z_scores[feature] * weight
            weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else 0

    def calculate_total_score(self, feature_vector, feature_names):
        """
        计算总分：三层加权
        """
        # 计算Z分数
        z_scores = self.calculate_robust_z_score(feature_vector, feature_names)

        if not z_scores:
            return 0

        # 计算各层分数
        level_score = self.score_level_features(z_scores)
        structure_score = self.score_structure_features(z_scores)
        shape_score = self.score_shape_features(z_scores)
        cross_score = self.score_cross_features(z_scores)

        # 加权总分（结构特征权重最高）
        weights = {
            'level': 0.3,
            'structure': 0.4,
            'shape': 0.2,
            'cross': 0.1
        }

        total_z = (level_score * weights['level'] +
                   structure_score * weights['structure'] +
                   shape_score * weights['shape'] +
                   cross_score * weights['cross'])

        # 转换为风险指数（0-100）
        # 使用sigmoid函数映射
        risk_index = 100 * self._sigmoid(total_z)

        # 分项分数（用于解释性）
        breakdown = {
            'level_score': level_score,
            'structure_score': structure_score,
            'shape_score': shape_score,
            'cross_score': cross_score,
            'total_z': total_z,
            'risk_index': risk_index
        }

        return breakdown

    def _sigmoid(self, x):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-x))

    def interpret_risk_index(self, risk_index):
        """解释风险指数"""
        if risk_index < 30:
            return "正常区", "风险特征不明显"
        elif risk_index < 50:
            return "轻度异常", "部分特征偏离"
        elif risk_index < 70:
            return "结构异动", "重要结构特征相似"
        elif risk_index < 85:
            return "高度相似", "与历史事件高度相似"
        else:
            return "极高风险", "与历史事件非常相似，需警惕"