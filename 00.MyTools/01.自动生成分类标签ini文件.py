import os
import argparse

def generate_labels_ini(target_dir):
    # 检查目标文件夹是否存在
    if not os.path.exists(target_dir):
        print(f"错误：目标文件夹不存在 - {target_dir}")
        return
    if not os.path.isdir(target_dir):
        print(f"错误：指定路径不是文件夹 - {target_dir}")
        return

    # 获取目标文件夹下的所有子文件夹名称（排除文件和隐藏文件夹）
    classes = []
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            classes.append(item)

    if not classes:
        print(f"警告：目标文件夹下没有找到子文件夹 - {target_dir}")
        return

    # 按字母顺序排序
    classes.sort()

    # 在目标文件夹下生成labels.ini
    output_path = os.path.join(target_dir, 'labels.ini')
    with open(output_path, 'w', encoding='utf-8') as f:
        for cls in classes:
            f.write(f"{cls}\t{cls}\n")  # 类别名 + Tab + 类别名

    print(f"成功生成labels.ini文件：{output_path}")
    print(f"包含 {len(classes)} 个类别：")
    for cls in classes:
        print(f"- {cls}")

if __name__ == "__main__":
    # 解析命令行参数，支持指定目标文件夹
    parser = argparse.ArgumentParser(description='生成labels.ini文件，读取指定文件夹下的子文件夹作为类别')
    parser.add_argument('--folder', default=r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\分类模型-lwl' ,help='目标文件夹路径（例如：./data 或 /home/user/dataset）')
    args = parser.parse_args()

    # 处理路径（支持相对路径和绝对路径）
    target_directory = os.path.abspath(args.folder)
    generate_labels_ini(target_directory)