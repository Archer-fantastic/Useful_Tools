import os
import json

def convert_labelme_labels(json_dir, old_label="A", new_label="B", save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 递归遍历所有子目录
    for root, dirs, files in os.walk(json_dir):
        for filename in files:
            if not filename.endswith(".json"):
                continue

            json_path = os.path.join(root, filename)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            changed = False

            # 修改标签
            for shape in data.get("shapes", []):
                if shape.get("label") == old_label:
                    shape["label"] = new_label
                    changed = True

            rel_path = os.path.relpath(root, json_dir)

            # 保存路径设置（保持目录结构）
            if save_dir:
                target_dir = os.path.join(save_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                save_path = os.path.join(target_dir, filename)
            else:
                save_path = json_path

            if changed:
                print(f"[修改] {json_path}: {old_label} → {new_label}")
            else:
                print(f"[跳过] {json_path}: 无 {old_label}")

            # 保存 JSON
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    print("\n处理完成！")


if __name__ == "__main__":
    convert_labelme_labels(
        json_dir=r"Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\检测模型\原始数据集\白点_数据增强",
        old_label="脱碳",
        new_label="白点",
        save_dir=None  # 覆盖保存
    )
