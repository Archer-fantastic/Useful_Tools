import os
import glob
import shutil
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort  # 用于ONNX模型
import tensorrt as trt  # 用于TensorRT模型
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化CUDA上下文
from datetime import datetime  # 新增：用于生成时间戳


def load_class_names(labels_ini_path):
    """从labels.ini/classes.txt加载类别名称"""
    if not os.path.exists(labels_ini_path):
        return None
    class_names = []
    with open(labels_ini_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                class_names.append(line.split('\t')[0] if '\t' in line else line)
    return class_names if class_names else None


def preprocess_image(image_path):
    """图像预处理（与训练时保持一致）"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32)
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    for i in range(3):
        img_np[i] = (img_np[i] / 255.0 - NORMALIZE_MEAN[i]) / NORMALIZE_STD[i]
    return img_np[np.newaxis, ...]


class ONNXModel:
    """ONNX模型加载与推理"""
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, img_tensor):
        outputs = self.session.run([self.output_name], {self.input_name: img_tensor})
        return outputs[0]


class TRTModel:
    """TensorRT模型加载与推理"""
    def __init__(self, model_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(model_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def predict(self, img_tensor):
        np.copyto(self.inputs[0]['host'], img_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(1, -1)


def load_model(model_path):
    """根据文件后缀自动选择模型加载方式"""
    ext = os.path.splitext(model_path)[1].lower()
    if ext == '.onnx':
        print(f"加载ONNX模型: {model_path}")
        return ONNXModel(model_path)
    elif ext in ['.trt', '.engine']:
        print(f"加载TensorRT模型: {model_path}")
        return TRTModel(model_path)
    else:
        raise ValueError(f"不支持的模型格式: {ext}，仅支持 .onnx, .trt, .engine")


def predict_single_image(model, image_path):
    """预测单张图像（支持ONNX/TRT）"""
    try:
        img_tensor = preprocess_image(image_path)
        outputs = model.predict(img_tensor)
        probabilities = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
        pred_idx = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0, pred_idx] * 100
        return {
            "image_path": image_path,
            "pred_class": CLASS_NAMES[pred_idx],
            "confidence": confidence
        }
    except Exception as e:
        print(f"预测失败 {image_path}：{str(e)}")
        return None


def collect_image_paths(input_path, recursive=False):
    """收集所有图像路径"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    image_paths = []
    
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if any(ext in ext_list for ext_list in image_extensions):
            image_paths.append(input_path)
        else:
            print(f"警告：{input_path} 不是支持的图像格式")
        return image_paths
    
    elif os.path.isdir(input_path):
        for ext in image_extensions:
            pattern = os.path.join(input_path, '**', ext) if recursive else os.path.join(input_path, ext)
            image_paths.extend(glob.glob(pattern, recursive=recursive))
        return sorted(image_paths)
    
    else:
        print(f"错误：输入路径不存在 - {input_path}")
        return []


def handle_image_file(image_path, pred_class, save_root, action):
    """处理预测后的图像文件（复制/移动/不保存）"""
    if action == 'none':
        return
    
    img_filename = os.path.basename(image_path)
    # 核心修改：在保存目录中加入时间戳子文件夹
    save_dir = os.path.join(save_root, pred_class)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, img_filename)
    
    # 避免文件名重复
    count = 1
    while os.path.exists(save_path):
        name, ext = os.path.splitext(img_filename)
        save_path = os.path.join(save_dir, f"{name}_{count}{ext}")
        count += 1
    
    # 执行复制/移动操作
    try:
        if action == 'copy':
            shutil.copy2(image_path, save_path)
            print(f"已复制: {image_path} -> {save_path}")
        elif action == 'move':
            shutil.move(image_path, save_path)
            print(f"已移动: {image_path} -> {save_path}")
    except Exception as e:
        print(f"文件操作失败 {image_path}：{str(e)}")


# --------------------------
# 配置参数（根据需求修改！）
# --------------------------
# 1. 模型与输入路径
MODEL_PATH = r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res\resnet18_20251030_150116\best_model.onnx"
INPUT_PATH = r"D:\Min\Projects\VSCodeProjects\dataset\cls_test_data\data"
RECURSIVE = True  # 是否递归遍历文件夹
LABELS_PATH = r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res\resnet18_20251030_150116\classes.txt"

# 2. 图像预处理参数（需与训练一致）
IMAGE_SIZE = (320, 320)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# 3. 结果保存与文件操作配置
SAVE_ROOT = r"D:\Min\Projects\VSCodeProjects\dataset\cls_test_data\pred_result"  # 保存根目录
FILE_ACTION = "copy"  # 文件操作：copy（默认）/ move / none（不保存）

CLASS_NAMES = None  # 从labels/classes文件加载


def main():
    # 生成时间戳（格式：年-月-日_时-分-秒，避免特殊字符）
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 1. 加载类别名称
    global CLASS_NAMES
    labels_path = LABELS_PATH
    if not labels_path:
        labels_path = os.path.join(os.path.dirname(MODEL_PATH), "labels.ini")
    
    CLASS_NAMES = load_class_names(labels_path)
    if not CLASS_NAMES:
        print(f"错误：未找到类别文件，请检查路径是否正确：{labels_path}")
        return

    # 2. 加载模型
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        return

    # 3. 收集图像路径
    print(f"\n处理输入：{INPUT_PATH}（递归模式：{RECURSIVE}）")
    image_paths = collect_image_paths(INPUT_PATH, RECURSIVE)
    if not image_paths:
        print("未找到图像文件")
        return
    print(f"共找到 {len(image_paths)} 张图像")

    # 4. 预测并处理文件（核心修改：在保存根目录后添加时间戳文件夹）
    # 最终保存路径格式：SAVE_ROOT/时间戳/预测类别/图像文件
    save_root_with_timestamp = os.path.join(SAVE_ROOT, timestamp)
    print(f"\n文件操作模式：{FILE_ACTION}，带时间戳的保存根目录：{save_root_with_timestamp if FILE_ACTION != 'none' else '无'}")
    print("\n预测结果：")
    print("-" * 120)
    print(f"{'图像路径':<70} | {'预测类别':<20} | 置信度(%) | 操作状态")
    print("-" * 120)
    
    for img_path in image_paths:
        result = predict_single_image(model, img_path)
        if result:
            pred_class = result["pred_class"]
            confidence = result["confidence"]
            # 执行文件操作时使用带时间戳的保存根目录
            if FILE_ACTION in ['copy', 'move']:
                handle_image_file(img_path, pred_class, save_root_with_timestamp, FILE_ACTION)
                status = "成功"
            else:
                status = "无"
            print(f"{result['image_path']:<70} | {pred_class:<20} | {confidence:.2f} | {status}")
        else:
            print(f"{img_path:<70} | {'预测失败':<20} | {'-':<10} | 无")
    print("-" * 120)
    print(f"\n预测完成！结果保存至：{save_root_with_timestamp if FILE_ACTION != 'none' else '无'}")


if __name__ == "__main__":
    main()