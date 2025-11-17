import numpy as np

# 替换为你的特征库路径
feature_db_path = r"D:\Min\Projects\VSCodeProjects\dataset\feature_cls_data\feature_database\resnet18_ce_20251029_155834_feature_db.npy"

# 加载特征库
feature_db = np.load(feature_db_path, allow_pickle=True).item()

# 打印类别数量
print(f"特征库包含 {len(feature_db)} 个类别：{list(feature_db.keys())}")

# 打印第一个类别的平均特征向量（前10个元素）
first_class = list(feature_db.keys())[0]
print(f"\n{first_class} 的平均特征向量（前10维）：")
print(feature_db[first_class][:10])