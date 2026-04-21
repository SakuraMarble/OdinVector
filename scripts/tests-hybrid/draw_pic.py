import matplotlib.pyplot as plt
import numpy as np

# ==================== 全局样式设置 ====================
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


# ==================== 数据整理 ====================
# 1. 候选队列长度 L 实验
L_values = [100, 300, 500]
L_qps = [613, 624, 417]
L_recall = [94.68, 94.72, 94.74]

# 2. 切换阈值 θ_switch (hit_rate) 实验
threshold_values = [0.00, 0.01, 0.02, 0.03, 1.00]
threshold_qps = [577, 370, 624, 261, 246]
threshold_recall = [9.98, 94.62, 94.72, 94.73, 94.75]

# 3. 粗排倍数 ρ_mult 实验
mult_values = [10, 20, 40]
mult_qps = [310, 411, 310]
mult_recall = [81.75, 94.72, 98.83]

# 4. 线程数扩展性实验 (注意：Recall 均为 94.72，更适合画 QPS vs 线程数)
thread_values = [12, 45, 64]
thread_qps = [172, 624, 387]
thread_recall = [94.72, 94.72, 94.72]


# ==================== 绘图函数 ====================
def plot_qps_recall(x, qps, recall, xlabel, title, filename,
                    annotate_points=True, x_is_log=False):
    """绘制 QPS 与 Recall 的双纵轴曲线图"""
    fig, ax1 = plt.subplots()

    # QPS (左轴)
    color_qps = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('QPS (queries/sec)', color=color_qps)
    ax1.plot(x, qps, marker='o', linestyle='-', color=color_qps,
             linewidth=2, markersize=8, label='QPS')
    ax1.tick_params(axis='y', labelcolor=color_qps)

    # Recall (右轴)
    ax2 = ax1.twinx()
    color_recall = 'tab:red'
    ax2.set_ylabel('Recall@k (%)', color=color_recall)
    ax2.plot(x, recall, marker='s', linestyle='--', color=color_recall,
             linewidth=2, markersize=8, label='Recall')
    ax2.tick_params(axis='y', labelcolor=color_recall)

    # 可选：在点上标注数值
    if annotate_points:
        for i, (xi, yi) in enumerate(zip(x, qps)):
            ax1.annotate(f'{yi}', (xi, yi), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9, color=color_qps)
        for i, (xi, yi) in enumerate(zip(x, recall)):
            ax2.annotate(f'{yi:.2f}%', (xi, yi), textcoords="offset points",
                         xytext=(0, -15), ha='center', fontsize=9, color=color_recall)

    # 标题与网格
    plt.title(title)
    ax1.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"图片已保存: {filename}")


def plot_qps_vs_threads(threads, qps, title, filename):
    """单独绘制线程数 - QPS 柱状图 (Recall 为常数)"""
    fig, ax = plt.subplots()
    bars = ax.bar([str(t) for t in threads], qps, color='steelblue', edgecolor='black')
    ax.set_xlabel('Threads')
    ax.set_ylabel('QPS (queries/sec)')
    ax.set_title(title)

    # 在柱顶标注数值
    for bar, val in zip(bars, qps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val}', ha='center', va='bottom', fontsize=11)

    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"图片已保存: {filename}")


# ==================== 生成各子图 ====================
if __name__ == "__main__":
    # 图1: 候选队列长度 L 的影响
    plot_qps_recall(
        x=L_values, qps=L_qps, recall=L_recall,
        xlabel='Candidate Queue Length L',
        title='Impact of L on QPS and Recall',
        filename='fig_L_qps_recall.png'
    )

    # 图2: 切换阈值 θ_switch 的影响 (注意 x=0 时对数坐标不适合，直接线性)
    plot_qps_recall(
        x=threshold_values, qps=threshold_qps, recall=threshold_recall,
        xlabel='Switch Threshold θ_switch',
        title='Impact of θ_switch on QPS and Recall',
        filename='fig_threshold_qps_recall.png'
    )

    # 图3: 粗排倍数 ρ_mult 的影响
    plot_qps_recall(
        x=mult_values, qps=mult_qps, recall=mult_recall,
        xlabel='Rerank Multiplier ρ_mult',
        title='Impact of ρ_mult on QPS and Recall',
        filename='fig_mult_qps_recall.png'
    )

    # 图4: 线程数扩展性 (Recall 为常数，改用柱状图更合适)
    plot_qps_vs_threads(
        threads=thread_values, qps=thread_qps,
        title='Scalability with Thread Count (Recall constant @ 94.72%)',
        filename='fig_threads_qps.png'
    )

    print("所有图像生成完毕。")