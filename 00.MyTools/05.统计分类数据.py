import os
from collections import defaultdict

def count_images_in_dir(dir_path):
    """统计单个目录下的图片数量（支持常见图片格式）"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    count = 0
    for file in os.listdir(dir_path):
        if file.lower().endswith(image_extensions):
            count += 1
    return count

def count_images_recursive(dir_path):
    """递归统计目录下所有图片（包括所有子文件夹）"""
    total = 0
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                total += 1
    return total

def mode1_statistics(root_dir, show_subfolders=True):
    """
    模式一：顶层目录为类别（cls1, cls2...），支持多层子文件夹统计
    统计每个类别的总样本数及所有层级子文件夹的样本数
    show_subfolders: 是否显示子文件夹的详细信息（True/False）
    """
    print("\n===== 模式一统计结果 =====")
    total_stats = defaultdict(dict)  # {类别: {子文件夹路径: 数量, "总数量": 合计}}
    
    # 遍历顶层类别目录
    for cls_name in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue  # 跳过非目录文件
        
        subfolder_counts = {}
        total = 0
        
        # 递归遍历所有层级子文件夹
        for root, _, _ in os.walk(cls_path):
            rel_path = os.path.relpath(root, cls_path)
            cnt = count_images_in_dir(root)
            if cnt > 0:
                subfolder_counts[rel_path] = cnt
                total += cnt
        
        total_stats[cls_name] = {**subfolder_counts, "总数量": total}
    
    # 打印统计结果
    for cls, stats in total_stats.items():
        if show_subfolders and len(stats) > 1:
            print(f"\n类别: {cls}")
            print("-" * 50)
            for subfolder, cnt in stats.items():
                if subfolder != "总数量":
                    print(f"  子文件夹 {subfolder}: {cnt} 张")
            print("-" * 50)
        print(f"  {cls}总样本数: {stats['总数量']} 张")
    
    all_total = sum(stats["总数量"] for stats in total_stats.values())
    print("\n" + "=" * 50)
    print(f"所有类别总样本数: {all_total} 张")
    return total_stats

def mode2_statistics(root_dir):
    """
    模式二：文件夹名称即为类别名称，直接统计每个类别的样本总量
    """
    print("\n===== 模式二统计结果 =====")
    cls_counts = defaultdict(int)
    
    for cls_name in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls_name)
        if os.path.isdir(cls_path):
            total = count_images_recursive(cls_path)
            cls_counts[cls_name] = total
    
    for cls, cnt in cls_counts.items():
        print(f"类别 {cls}: {cnt} 张")
    
    total = sum(cls_counts.values())
    print("\n" + "=" * 30)
    print(f"所有类别总样本数: {total} 张")
    return cls_counts

def mode3_statistics(root_dir):
    """
    模式三：仅统计顶层类别（cls1、cls2等）的总样本量，递归包含所有子文件夹
    输出格式：cls1：样本量xxx
    """
    print("\n===== 模式三统计结果 =====")
    cls_counts = defaultdict(int)
    
    # 遍历顶层所有类别文件夹
    for cls_name in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls_name)
        if os.path.isdir(cls_path):
            # 递归统计该类别下所有图片（包括所有子文件夹）
            total = count_images_recursive(cls_path)
            cls_counts[cls_name] = total
    
    # 按类别名称排序输出
    for cls in sorted(cls_counts.keys()):
        print(f"{cls}：样本量{cls_counts[cls]}")
    
    # 总样本数
    total = sum(cls_counts.values())
    print("\n" + "=" * 30)
    print(f"所有类别总样本量: {total}")
    return cls_counts

def main():
    # 设置参数
    # root_dir = r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\分类模型-划痕-白点'
    # root_dir = r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\分类模型_数据增强'
    # root_dir = r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\00.分类模型 - 初始数据'
    # root_dir = r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\分类模型-亮痕-暗痕-白点-漏金属-裂纹'
    root_dir = r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\检测模型\原始数据集' 
    mode = 3  # 1: 模式一，2: 模式二，3: 模式三
    show_subfolders = True  # 仅模式一有效
    
    if not os.path.exists(root_dir):
        print(f"错误：目录 {root_dir} 不存在！")
        return
    
    if mode == 1:
        mode1_statistics(root_dir, show_subfolders=show_subfolders)
    elif mode == 2:
        mode2_statistics(root_dir)
    elif mode == 3:
        mode3_statistics(root_dir)
    else:
        print("错误：无效的模式编号，仅支持 1、2 或 3！")


if __name__ == "__main__":
    main()