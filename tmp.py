import matplotlib.pyplot as plt
import numpy as np

def main():
    # ================= 数据准备 (保持不变) =================
    experiments = [
        {"id": "1", "name": "FreshDiskANN-Modify-build (R32_L32_B18_M16)", "meta": "Build:25G/36h | SearchMem:40G",
         "recall": [32.71, 44.87, 57.83, 70.10, 73.75, 83.53, 90.39],
         "qps":    [706.97, 545.55, 394.92, 255.75, 235.71, 138.90, 76.10]},
        {"id": "2", "name": "DiskANN-Origin-build (R32_L32_B18_M16)", "meta": "Build:80G/36h  | SearchMem:18.6G",
         "recall": [44.50, 61.35, 73.44, 83.30, 87.73],
         "qps":    [651.96, 375.69, 221.33, 121.13, 83.05]},
        {"id": "3", "name": "DiskANN-Origin-build (R64_L100_B60_M64)", "meta": "Build:125G/62h | SearchMem:62G",
         "recall": [78.33, 89.62, 95.06, 97.97, 98.85],
         "qps":    [708.32, 417.08, 261.85, 131.04, 85.38]},
        {"id": "4", "name": "PG-HNSW build (M=8 EFC=100)", "meta": "Build:653G/22h | SearchMem:High",
         "recall": [x * 100 for x in [0.4013, 0.5227, 0.6370, 0.7375, 0.8175, 0.8785, 0.9174]],
         "qps":    [2902.25, 2339.20, 1630.39, 1006.65, 581.36, 316.48, 171.53]},
        {"id": "5", "name": "ODinANN-Modify-Insert (0.99B R32_L100_B33+10M)", "meta": "Peakinsert 67GB/18h | SearchMem:70G",
         "recall": [48.52, 62.46, 74.92, 84.79, 87.28, 93.20, 96.68],
         "qps":    [853.10, 679.57, 479.31, 281.48, 236.30, 140.72, 74.22]},
        {"id": "6", "name": "ODinANN-Origin-Insert (0.99B R64_L100_B60+10M)", "meta": "Peakinsert 200G/30h | SearchMem:95G",
         "recall": [63.69, 77.06, 86.78, 93.32, 94.85, 97.85, 99.24],
         "qps":    [897.63, 675.44, 477.10, 306.89, 265.39, 142.10, 73.43]},
        {"id": "7", "name": "FreshDiskANN-Origin-Insert (0.99B R32_L100_B33+10M)", "meta": "Peakinsert:140G/19.5h | SearchMem:73G",
         "recall": [48.47, 62.57, 74.87, 84.75, 87.26, 93.17, 96.70],
         "qps":    [579.83, 467.23, 367.81, 237.29, 201.11, 116.13, 65.25]},
        {"id": "8", "name": "ODinANN-Origin-Insert (0.99B R32_L100_B33+10M)", "meta": "Peakinsert:125G/21h | SearchMem:73G",
         "recall": [48.23, 62.36, 74.83, 84.68, 87.21, 93.15, 96.67],
         "qps":    [780.28, 573.70, 430.81, 282.93, 236.59, 132.94, 73.43]}
    ]

    # ================= 绘图设置 (颜值优化版) =================
    plt.figure(figsize=(20, 14))
    
    # 核心改进：定义一套“高级色池”
    # 我们先取 tab20 的颜色，这是公认好看的分类色
    tab20_colors = list(plt.cm.tab20.colors)
    # 把原本那种浅色（成对出现的第2个）挪到后面，先用深色，保证对比度
    pretty_colors = tab20_colors[::2] + tab20_colors[1::2]
    # 手动在第11个位置（Exp 11）插入你想要的黄色/金色，或者确保它存在
    if len(pretty_colors) > 10:
        pretty_colors[10] = '#FFD700' # 纯金黄

    # 形状池
    marker_pool = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'P']

    # 循环绘制
    for i, exp in enumerate(experiments):
        # 自动取色和取形状：利用 % len 确保永远不会溢出
        current_color = pretty_colors[i % len(pretty_colors)]
        current_marker = marker_pool[i % len(marker_pool)]
        
        plt.plot(exp['recall'], exp['qps'], 
                 marker=current_marker, 
                 color=current_color, 
                 linewidth=2.5, 
                 markersize=10, 
                 alpha=0.9, # 稍微加点透明度更显高级
                 label=f"Exp {exp['id']}")

        # --- 文本标注 (背景框颜色跟随线条) ---
        idx_to_label = -2 if len(exp['recall']) > 2 else -1
        if float(exp['qps'][0]) > 1000: idx_to_label = -1 
        
        x_text = exp['recall'][idx_to_label]
        y_text = min(exp['qps'][idx_to_label], 980) 
        
        plt.text(x_text, y_text + 15, f"Exp {exp['id']}", 
                 fontsize=11, fontweight='bold', color='black',
                 # 背景框颜色调淡一点，不遮挡曲线
                 bbox=dict(facecolor=current_color, alpha=0.2, edgecolor='none', pad=1))

    # ================= 细节打磨 =================
    plt.title('Recall vs QPS Performance Benchmark', fontsize=24, fontweight='bold', pad=30, color='#333333')
    plt.xlabel('Recall@10 (%)', fontsize=16, labelpad=10)
    plt.ylabel('Queries Per Second (QPS)', fontsize=16, labelpad=10)

    plt.ylim(0, 1000)
    plt.xlim(30, 100)
    plt.yticks(np.arange(0, 1001, 50), fontsize=12)
    plt.xticks(np.arange(30, 101, 5), fontsize=12)

    # 网格线调淡，让图表看起来更干净
    plt.grid(True, which='major', linestyle='-', alpha=0.4, color='#999999')
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)

    # 图例排版
    legend_handles, _ = plt.gca().get_legend_handles_labels()
    custom_labels = [f"Exp {e['id']}: {e['name']}\n      {e['meta']}" for e in experiments]

    plt.legend(legend_handles, custom_labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, -0.1), 
               ncol=3, 
               fontsize=10, 
               frameon=True, 
               shadow=False, # 去掉阴影更扁平化
               borderpad=1.2,
               edgecolor='#CCCCCC')

    plt.subplots_adjust(bottom=0.25, top=0.92, left=0.08, right=0.95)

    output_file = "benchmark_pretty_final.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ 漂亮的高清图已生成: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()