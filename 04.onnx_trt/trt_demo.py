"""
TensorRT 图像分类推理 Demo (ONNX -> TRT -> 推理)
支持动态 batch、FP16 加速、自动构建引擎
"""

import os
import argparse
import time
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# -------------------------------
# 全局配置
# -------------------------------

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


# -------------------------------
# 图像预处理
# -------------------------------

def preprocess_image(image_path, input_size=(320, 320)):
    """预处理图像：调整大小、归一化、转为 NCHW 格式"""
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(input_size, Image.BILINEAR)

        # 转为数组并归一化
        img_np = np.array(image).astype(np.float32) / 255.0
        img_np = (img_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)   # CHW -> NCHW

        return img_np.copy()
    except Exception as e:
        print(f"❌ 图像预处理失败: {e}")
        return None


# -------------------------------
# 构建 TensorRT 引擎
# -------------------------------

def build_engine(onnx_file, engine_file, precision="fp16", 
                 min_batch=1, opt_batch=1, max_batch=8, img_size=(320, 320)):
    """从 ONNX 构建 TRT 引擎"""
    if os.path.exists(engine_file):
        print(f"🟢 TRT 引擎已存在，跳过构建: {engine_file}")
        return True

    if not os.path.exists(onnx_file):
        print(f"❌ ONNX 文件不存在: {onnx_file}")
        return False

    print(f"🛠️ 正在构建 TensorRT 引擎... ({precision})")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"解析错误: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # 动态 shape 设置
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    min_shape = (min_batch, 3, img_size[1], img_size[0])
    opt_shape = (opt_batch, 3, img_size[1], img_size[0])
    max_shape = (max_batch, 3, img_size[1], img_size[0])
    profile.set_shape(input_name, min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape)
    config.add_optimization_profile(profile)

    # 精度设置
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ 已启用 FP16 精度")

    # 序列化引擎
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("❌ 引擎构建失败")
        return False

    with open(engine_file, "wb") as f:
        f.write(engine_bytes)
    print(f"✅ TRT 引擎已保存至: {engine_file}")
    return True


# -------------------------------
# 加载引擎 & 绑定索引
# -------------------------------

def load_engine(engine_file):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


# -------------------------------
# 后处理：Softmax + 标签映射
# -------------------------------

def postprocess(output, labels=None):
    probs = np.exp(output) / np.sum(np.exp(output), axis=1)
    idx = np.argmax(probs, axis=1)[0]
    prob = probs[0, idx]

    cls_name = f"Class {idx}"
    if labels and idx < len(labels):
        cls_name = labels[idx]

    print("\n🎯 推理结果:")
    print(f"  类别: {cls_name} (ID={idx})")
    print(f"  置信度: {prob:.4f} ({prob*100:.2f}%)")
    return {"class": cls_name, "index": int(idx), "confidence": float(prob)}


# -------------------------------
# 加载标签
# -------------------------------

def load_labels(label_file):
    if not label_file or not os.path.exists(label_file):
        return None
    with open(label_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]



# -------------------------------
# 新版推理函数（适配 TensorRT 10）
# -------------------------------

def infer(engine, input_data):
    context = engine.create_execution_context()

    # 获取输入输出张量名称
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # 设置输入形状（动态 shape）
    context.set_input_shape(input_name, input_data.shape)

    # 分配 host & device 缓冲区
    h_input = cuda.pagelocked_empty(trt.volume(input_data.shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(
        trt.volume(context.get_tensor_shape(output_name)), dtype=np.float32
    )
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # 拷贝输入数据
    np.copyto(h_input, input_data.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # 设置张量地址
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # 执行推理
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

    # 拷贝结果回 CPU
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # 重塑输出
    output_shape = context.get_tensor_shape(output_name)
    return h_output.reshape(output_shape)


# -------------------------------
# 修改 main() 中打印信息部分
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res\resnet18_20251030_150116\best_model.onnx", help="输入 ONNX 模型路径")
    parser.add_argument("--engine", default=r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res\resnet18_20251030_150116\defect_detection.trt", help="输出 TRT 引擎路径")
    parser.add_argument("--image", default=r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\2025102913_备份\漏底涂\178_304-面积0.43-宽0.18-高1.4-第5611片--第1条-X_433.43mm-Y_7913.636米-45234836脉冲-07-29-22.7664特征法.bmp", help="输入测试图像路径")
    parser.add_argument("--labels", default=r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res\resnet18_20251030_150116\classes.txt", help="类别标签文件")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16", help="精度模式")
    parser.add_argument("--input-size", type=int, nargs=2, default=[320, 320], help="输入尺寸")
    args = parser.parse_args()

    print(f"🚀 开始运行 TensorRT 推理 demo (版本: {trt.__version__})")

    # Step 1: 构建引擎（无需修改，build_engine 不受影响）
    if not build_engine(args.onnx, args.engine, precision=args.precision, img_size=args.input_size):
        exit(1)

    # Step 2: 加载引擎
    engine = load_engine(args.engine)

    # ✅ 正确获取输入/输出张量数量和名字（TRT 10 写法）
    num_tensors = engine.num_io_tensors  # 新属性
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    print(f"📌 IO 张量总数: {num_tensors}")
    print(f"📌 输入张量名: {input_name} (类型: {engine.get_tensor_dtype(input_name)})")
    print(f"📌 输出张量名: {output_name} (类型: {engine.get_tensor_dtype(output_name)})")

    # Step 3: 预处理图像
    if not os.path.exists(args.image):
        print(f"⚠️ 测试图像未找到: {args.image}，尝试创建一张随机图像...")
        Image.fromarray(np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)).save("test.jpg")
        args.image = "test.jpg"

    input_data = preprocess_image(args.image, input_size=tuple(args.input_size))
    if input_data is None:
        print("❌ 预处理失败，退出")
        exit(1)

    # Step 4: 推理
    print("🔥 开始推理...")
    start = time.time()
    result = infer(engine, input_data)
    infer_time = (time.time() - start) * 1000
    print(f"⏱️ 推理耗时: {infer_time:.2f} ms")

    # Step 5: 后处理
    labels = load_labels(args.labels)
    postprocess(result, labels)
if __name__ == "__main__":
    main()