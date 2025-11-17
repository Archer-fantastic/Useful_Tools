import os
import random
import shutil
import argparse
from pathlib import Path

def split_dataset_recursive(raw_root_dir, test_root_dir, split_ratio=0.1, operation='copy'):
    # 1. 检查原始数据集根目录是否存在
    raw_root = Path(raw_root_dir)
    if not raw_root.exists():
        print(f"错误：原始数据集路径 {raw_root_dir} 不存在，请检查路径！")
        return

    # 2. 验证操作模式是否合法
    if operation not in ['copy', 'move']:
        print(f"错误：操作模式 {operation} 不支持，请使用 'copy' 或 'move'！")
        return

    # 3. 确保测试集根目录存在
    test_root = Path(test_root_dir)
    test_root.mkdir(parents=True, exist_ok=True)

    # 4. 递归遍历原始数据集中的所有文件夹
    for raw_dir, _, file_names in os.walk(raw_root_dir):
        # 过滤掉空文件夹（无文件则跳过）
        if not file_names:
            continue

        # 5. 计算当前文件夹相对于原始根目录的相对路径（复刻结构）
        relative_path = os.path.relpath(raw_dir, raw_root_dir)
        # 构建测试集中对应的文件夹路径
        test_dir = test_root / relative_path
        test_dir.mkdir(parents=True, exist_ok=True)

        # 6. 按比例筛选当前文件夹下的测试集文件
        total_files = len(file_names)
        # 计算应挑选的测试集文件数量（至少1个，若文件数≥1且比例导致数量为0）
        test_file_count = max(1, int(total_files * split_ratio)) if total_files > 0 else 0
        # 防止比例过高导致测试集数量超过总文件数
        test_file_count = min(test_file_count, total_files)

        # 7. 随机挑选文件（固定种子确保结果可复现，删除则每次随机）
        random.seed(42)
        test_files = random.sample(file_names, test_file_count)

        # 8. 根据选择的操作模式处理文件（复制或移动）
        for file_name in test_files:
            src_file = Path(raw_dir) / file_name
            dst_file = test_dir / file_name
            if operation == 'copy':
                shutil.copy2(src_file, dst_file)  # 复制文件并保留元数据
            else:  # move模式
                shutil.move(src_file, dst_file)   # 移动文件（原始路径中移除）

        # 打印当前文件夹的处理结果
        print(f"文件夹 {relative_path} 处理完成：")
        print(f"  - 总文件数：{total_files}")
        print(f"  - 测试集文件数：{test_file_count}")
        print(f"  - 操作模式：{operation}")
        print(f"  - 测试集路径：{test_dir}\n")

    print(f"递归划分完成！测试集根目录：{test_root_dir}")
    print(f"操作模式：{operation}（{'原始文件未改动' if operation == 'copy' else '原始文件已移除测试集数据'}）")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="递归划分分类数据集（保留完整目录结构）")
    parser.add_argument("--raw_dir", default=r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\分类模型-亮痕-暗痕-白点-漏金属-裂纹\白点\20251112_白点误判', help="原始数据集根目录")
    parser.add_argument("--test_dir", default=r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\分类模型-测试集\白点\20251112_白点误判', help="测试集输出根目录")
    parser.add_argument("--ratio", type=float, default=0.1, help="测试集划分比例（默认0.1）")
    parser.add_argument("--operation", choices=['copy', 'move'], default='move', help="文件操作模式（copy：复制，move：移动，默认copy）")
    
    args = parser.parse_args()

    # 调用递归划分函数
    split_dataset_recursive(
        raw_root_dir=args.raw_dir,
        test_root_dir=args.test_dir,
        split_ratio=args.ratio,
        operation=args.operation
    )