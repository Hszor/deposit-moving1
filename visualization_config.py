"""
visualization_config.py
高级可视化配置
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd


def configure_visualization_style():
    """
    配置高级可视化样式
    """
    # 设置专业样式
    plt.style.use('seaborn-v0_8-darkgrid')

    # 使用viridis色系（专业感）
    plt.rcParams['image.cmap'] = 'viridis'

    # 配置中文和数学字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 图形尺寸和DPI
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

    # 字体大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 9

    # 线条和标记
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['grid.alpha'] = 0.3

    # 颜色配置
    COLORS = {
        'primary': '#2C3E50',  # 深蓝灰
        'secondary': '#3498DB',  # 亮蓝
        'accent': '#E74C3C',  # 红色
        'success': '#2ECC71',  # 绿色
        'warning': '#F39C12',  # 橙色
        'info': '#9B59B6',  # 紫色

        # 风险等级颜色
        'risk_low': '#27AE60',  # 深绿
        'risk_medium': '#F1C40F',  # 黄色
        'risk_high': '#E67E22',  # 橙色
        'risk_extreme': '#E74C3C',  # 红色

        # 序列颜色
        'historical': '#95A5A6',  # 灰色（历史）
        'forecast': '#3498DB',  # 蓝色（预测）
        'event': '#E74C3C',  # 红色（事件）
        'normal': '#2ECC71'  # 绿色（正常）
    }

    # 专业调色板
    PALETTES = {
        'risk': ['#27AE60', '#F1C40F', '#E67E22', '#E74C3C'],  # 风险等级
        'sequential': 'viridis',  # 顺序数据
        'diverging': 'RdBu_r',  # 发散数据
        'qualitative': 'Set3',  # 分类数据
    }

    return COLORS, PALETTES


def create_structural_comparison_plot(historical_data, forecast_data,
                                      event_windows, save_path=None):
    """
    创建结构对比图（高级版本）
    """
    COLORS, _ = configure_visualization_style()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 1. 增长缺口对比
    ax1 = axes[0]

    # 绘制历史数据（灰色）
    if 'growth_gap' in historical_data.columns:
        ax1.plot(historical_data['date'], historical_data['growth_gap'],
                 color=COLORS['historical'], alpha=0.6, linewidth=1,
                 label='历史数据')

    # 标记事件窗口
    for name, (start, end) in event_windows.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        ax1.axvspan(start_dt, end_dt, alpha=0.1, color=COLORS['event'])

        # 添加标签（仅第一个窗口）
        if name == list(event_windows.keys())[0]:
            ax1.text((start_dt + (end_dt - start_dt) / 2),
                     ax1.get_ylim()[1] * 0.9, '事件窗口',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor=COLORS['event'], alpha=0.5))

    # 绘制预测数据
    if forecast_data and 'growth_gap' in forecast_data:
        # 创建预测时间序列
        forecast_dates = pd.date_range('2026-01-01', periods=len(forecast_data['growth_gap']),
                                       freq='QE')
        ax1.plot(forecast_dates, forecast_data['growth_gap'],
                 color=COLORS['forecast'], linewidth=2.5, marker='o',
                 markersize=8, label='2026年预测')

        # 添加置信区间（简化）
        forecast_std = np.std(forecast_data['growth_gap'])
        ax1.fill_between(forecast_dates,
                         forecast_data['growth_gap'] - forecast_std,
                         forecast_data['growth_gap'] + forecast_std,
                         color=COLORS['forecast'], alpha=0.2)

    ax1.set_ylabel('增长缺口 (%)', fontweight='bold')
    ax1.set_title('增长缺口结构对比', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. 存款到期率对比
    ax2 = axes[1]

    if 'maturity_rate' in historical_data.columns:
        ax2.plot(historical_data['date'], historical_data['maturity_rate'] * 100,
                 color=COLORS['historical'], alpha=0.6, linewidth=1)

    # 标记事件窗口
    for start, end in event_windows.values():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        ax2.axvspan(start_dt, end_dt, alpha=0.1, color=COLORS['event'])

    # 绘制预测数据
    if forecast_data and 'maturity_rate' in forecast_data:
        forecast_dates = pd.date_range('2026-01-01', periods=len(forecast_data['maturity_rate']),
                                       freq='QE')
        ax2.plot(forecast_dates, forecast_data['maturity_rate'] * 100,
                 color=COLORS['forecast'], linewidth=2.5, marker='s',
                 markersize=8)

    ax2.set_ylabel('存款到期率 (%)', fontweight='bold')
    ax2.set_title('存款到期率结构对比', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)

    # 3. 结构相似度评分
    ax3 = axes[2]

    # 这里可以添加结构相似度的时间序列
    # 示例：计算滚动相似度
    if 'growth_gap' in historical_data.columns and len(historical_data) > 20:
        # 计算滚动窗口的结构特征相似度（简化示例）
        window_size = 8
        similarity_scores = []
        dates = []

        for i in range(len(historical_data) - window_size + 1):
            window_data = historical_data.iloc[i:i + window_size]

            # 这里应该使用真正的结构相似度计算
            # 简化：使用增长缺口的波动性作为代理
            vol = window_data['growth_gap'].std()
            similarity_scores.append(vol)
            dates.append(window_data['date'].iloc[window_size // 2])

        ax3.plot(dates, similarity_scores, color=COLORS['secondary'],
                 linewidth=1.5, alpha=0.7, label='结构波动性')

        # 标记高波动期（可能的结构变化）
        threshold = np.percentile(similarity_scores, 75)
        high_vol_idx = np.where(np.array(similarity_scores) > threshold)[0]

        if len(high_vol_idx) > 0:
            for idx in high_vol_idx:
                ax3.axvspan(dates[idx], dates[min(idx + 1, len(dates) - 1)],
                            alpha=0.2, color=COLORS['warning'])

    ax3.set_xlabel('日期', fontweight='bold')
    ax3.set_ylabel('结构波动性', fontweight='bold')
    ax3.set_title('结构特征时间演变', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 设置x轴日期格式
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('存款搬家结构特征对比分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
