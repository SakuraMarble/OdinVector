#!/bin/bash

# 激活 psrecord-env 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate psrecord-env

# ================= 1. 路径与参数配置 =================

# --- 输入/工具路径 (绝对路径) ---
# 混合搜索工具的可执行文件路径
SEARCH_TOOL="/home/mabojing/clean-workbase/OdinVectorDB/build/tests/hybrid_tag_search"
# 磁盘索引前缀路径（不含 _disk.index 后缀）
INDEX_PREFIX="/home/mabojing/clean-workbase/sift_R64_L100_M1_B006/sift1m"   # 已是分片构建的版了
# 查询向量文件
QUERY_FILE="/home/mabojing/datasets/sift1m/query_gt_pair_query10k_label100/query_vector.bin"
# Groundtruth文件（若无则填 null）
TRUTHSET="/home/mabojing/datasets/sift1m/query_gt_pair_query10k_label100/groundtruth.bin"
# 倒排索引文件路径
IVF_INDEX="/home/mabojing/index_build_1m_R64_L100_B1_M1/ivf.index"
# 原始向量文件路径
RAW_VECTOR="/home/mabojing/datasets/sift1m/sift1m_raw_vectors.bin"
# 查询标签表达式文件
QUERY_LABEL_FILE="/home/mabojing/datasets/sift1m/query_gt_pair_query10k_label100/query_label.txt"

# --- 输出配置 ---
# 所有生成文件存放的文件夹
WORK_DIR="/home/mabojing/clean-workbase/search_DEBUG_new_stage"

# --- 搜索参数 ---
DATA_TYPE="float"           # 向量类型：float/int8/uint8
NUM_THREADS=1              # OpenMP 线程数
BEAMWIDTH=4                 # Beam Search 宽度
K=10                        # 返回 Top-K 结果
SIMILARITY="l2"             # 距离度量：l2/cosine
MEM_L=10                     # 内存索引 L 参数（0 = 不使用内存索引）
HIT_RATE_THRESHOLD=1      # 策略切换阈值
IVF_TOPL_MULTIPLIER=10000       # IVF 粗排扩展倍数
SEARCH_LISTS="300"   # 搜索列表大小列表（可多个，需 ≥ K）

# --- 日志与监控文件 ---
APP_LOG="${WORK_DIR}/search_process.log"       # 程序标准输出日志
MEM_LOG="${WORK_DIR}/resource_monitor.log"     # 内存/CPU数据记录
MEM_PLOT="${WORK_DIR}/resource_monitor.png"    # 内存/CPU变化曲线图

# ================= 2. 环境准备 =================

# 创建输出目录 (如果不存在)
if [ ! -d "$WORK_DIR" ]; then
    echo "创建工作目录: $WORK_DIR"
    mkdir -p "$WORK_DIR"
fi

# 检查必要文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ 错误: 文件不存在 - $1"
        exit 1
    fi
}

echo "=================================================="
echo "   检查输入文件"
echo "=================================================="
check_file "$SEARCH_TOOL"
check_file "${INDEX_PREFIX}_disk.index"
check_file "$QUERY_FILE"
check_file "$IVF_INDEX"
check_file "$RAW_VECTOR"
check_file "$QUERY_LABEL_FILE"
if [ "$TRUTHSET" != "null" ] && [ ! -f "$TRUTHSET" ]; then
    echo "⚠️ 警告: Groundtruth 文件不存在，将跳过召回率计算 - $TRUTHSET"
fi

# 检查 psrecord 是否安装
if ! command -v psrecord &> /dev/null; then
    echo "警告: 未检测到 psrecord，监控可能无法启动。"
    echo "建议运行: pip install psrecord matplotlib"
fi

# 屏蔽 Python 绘图时的警告
export PYTHONWARNINGS="ignore"

# ================= 3. 启动混合搜索 =================

echo "=================================================="
echo "   开始混合标签向量搜索"
echo "=================================================="
echo "工具路径: $SEARCH_TOOL"
echo "索引前缀: $INDEX_PREFIX"
echo "查询文件: $QUERY_FILE"
echo "标签文件: $QUERY_LABEL_FILE"
echo "输出目录: $WORK_DIR"
echo "--------------------------------------------------"
echo "搜索参数:"
echo "  - 数据类型: $DATA_TYPE"
echo "  - 线程数: $NUM_THREADS"
echo "  - Beamwidth: $BEAMWIDTH"
echo "  - Top-K: $K"
echo "  - 相似度: $SIMILARITY"
echo "  - 阈值: $HIT_RATE_THRESHOLD"
echo "  - IVF倍数: $IVF_TOPL_MULTIPLIER"
echo "  - 搜索列表: $SEARCH_LISTS"
echo "--------------------------------------------------"

# 切换到工作目录
cd "$WORK_DIR" || { echo "无法进入目录 $WORK_DIR"; exit 1; }

# 执行混合搜索命令
# 参数顺序参考 hybrid_tag_search 使用文档
nohup "$SEARCH_TOOL" \
    "$DATA_TYPE" \
    "$INDEX_PREFIX" \
    "$NUM_THREADS" \
    "$BEAMWIDTH" \
    "$QUERY_FILE" \
    "$TRUTHSET" \
    "$K" \
    "$SIMILARITY" \
    "$MEM_L" \
    "$IVF_INDEX" \
    "$RAW_VECTOR" \
    "$QUERY_LABEL_FILE" \
    "$HIT_RATE_THRESHOLD" \
    "$IVF_TOPL_MULTIPLIER" \
    $SEARCH_LISTS > "$APP_LOG" 2>&1 &

# 获取搜索进程 PID
SEARCH_PID=$!

if [ -z "$SEARCH_PID" ]; then
    echo "❌ 错误：搜索进程启动失败！"
    exit 1
fi

echo "✅ 搜索进程已启动，PID: $SEARCH_PID"

# ================= 4. 启动资源监控 (psrecord) =================

echo "   正在启动 psrecord 监控..."

# --interval 0.5: 每0.5秒采样一次
# --include-children: 包含子线程/子进程的资源消耗
nohup psrecord $SEARCH_PID \
    --log "$MEM_LOG" \
    --plot "$MEM_PLOT" \
    --interval 0.1 \
    --include-children > /dev/null 2>&1 &

MONITOR_PID=$!

echo "✅ 监控进程已启动，PID: $MONITOR_PID"
echo "--------------------------------------------------"
echo "📋 操作指南:"
echo "1. 查看实时搜索日志:  tail -f $APP_LOG"
echo "2. 查看资源数据记录:  tail -f $MEM_LOG"
echo "3. 查看混合策略统计:  grep 'IVF/Graph' $APP_LOG"
echo "4. 任务完成后，请查看: $MEM_PLOT (内存消耗图)"
echo "=================================================="

# ================= 5. 等待完成提示 =================

echo ""
echo "⏳ 等待搜索完成... (可以使用 Ctrl+C 中断监控，但搜索进程仍在后台运行)"
echo "   查看进程状态: ps -p $SEARCH_PID"
echo "   强制终止搜索: kill $SEARCH_PID"
echo ""

# # 可选：等待搜索进程结束并显示结果
# wait $SEARCH_PID
# if [ $? -eq 0 ]; then
#     echo "=================================================="
#     echo "✅ 搜索任务成功完成！"
#     echo "=================================================="
#     echo "📊 结果摘要:"
#     echo "   - 搜索日志: $APP_LOG"
#     echo "   - 内存曲线: $MEM_PLOT"
#     echo ""
#     echo "   查看性能统计:"
#     echo "   tail -n 20 $APP_LOG | grep -E 'L\s+Beamwidth|IVF/Graph'"
#     echo "=================================================="
# else
#     echo "❌ 搜索进程异常退出，请检查日志: $APP_LOG"
# fi