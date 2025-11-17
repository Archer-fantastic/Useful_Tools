import onnxruntime as ort
import numpy as np
from PIL import Image
import argparse
import re

def preprocess_image(image_path, input_size=(320, 320)):
    """图像预处理：调整大小、归一化等，确保输出为float32类型"""
    # 打开图像并转换为RGB
    image = Image.open(image_path).convert('RGB')
    
    # 调整图像大小
    image = image.resize(input_size)
    
    # 转换为numpy数组并明确指定为float32
    image_array = np.array(image, dtype=np.float32)
    
    # 转换为CHW格式 (HWC -> CHW)
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # 归一化（使用float32类型的均值和标准差）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_array = (image_array / 255.0 - mean) / std
    
    # 添加批次维度 (CHW -> BCHW)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def load_onnx_model(onnx_path):
    """加载ONNX模型并创建推理会话"""
    providers = ort.get_available_providers()
    print(f"使用执行 providers: {providers}")
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # 获取输入和输出信息
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    input_name = input_info.name
    input_shape = input_info.shape
    input_type = input_info.type
    output_name = output_info.name
    output_shape = output_info.shape
    
    print(f"模型输入名称: {input_name}, 输入形状: {input_shape}, 期望类型: {input_type}")
    print(f"模型输出名称: {output_name}, 输出形状: {output_shape}")
    
    return session, input_name, output_name

def infer_image(session, input_name, output_name, image_array):
    """使用ONNX模型进行推理"""
    print(f"实际输入数据类型: {image_array.dtype}")
    outputs = session.run([output_name], {input_name: image_array})
    return outputs[0]

def load_labels(labels_path):
    """
    加载标签文件，支持两种格式：
    格式一：每行包含两个相同的标签（用空格或制表符分隔）
    格式二：每行一个标签
    """
    labels = []
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()  # 去除首尾空白和换行符
                if not line:
                    continue  # 跳过空行
                
                # 尝试按格式一处理：分割为多个部分（支持空格、制表符等分隔）
                parts = re.split(r'\s+', line)  # 按任意空白字符分割
                if len(parts) >= 2 and parts[0] == parts[1]:
                    # 确认是格式一：取第一个部分作为标签
                    labels.append(parts[0])
                else:
                    # 格式二：直接取整行作为标签
                    labels.append(line)
        
        print(f"成功加载标签，共 {len(labels)} 个类别")
        return labels
        
    except Exception as e:
        print(f"加载标签文件出错: {str(e)}")
        return None

def postprocess_result(result, labels_path=None):
    """后处理推理结果，返回概率最高的类别"""
    # 计算softmax获取概率
    probabilities = np.exp(result) / np.sum(np.exp(result), axis=1, keepdims=True)
    
    # 获取最高概率的索引和值
    max_prob_index = np.argmax(probabilities, axis=1)[0]
    max_prob_value = probabilities[0, max_prob_index]
    
    # 获取类别名称
    class_name = f"类别 {max_prob_index}"
    if labels_path:
        labels = load_labels(labels_path)
        if labels and max_prob_index < len(labels):
            class_name = labels[max_prob_index]
    
    return {
        "class_index": int(max_prob_index),
        "class_name": class_name,
        "probability": float(max_prob_value)
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用ONNX模型对图像进行推理，支持两种标签格式')
    parser.add_argument('--model', default=r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res\resnet18_20251030_150116\best_model.onnx", help='ONNX模型文件路径')
    parser.add_argument('--image', default=r"D:\Min\2.项目资料\CYS250804阳极涂布机尾外观瑕疵CCD检测ATL\AI小图\暂时不检\暗划痕\338_1007-面积1.01-宽0.28-高3.32-第3736片--第1条-X_261.11mm-Y_3433.376米-19624047脉冲-01-16-35.2210特征法.bmp", help='输入图像文件路径')
    parser.add_argument('--labels', default=r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\labels.ini",help='类别标签文件路径（支持两种格式）')
    parser.add_argument('--input-size', type=int, nargs=2, default=[320, 320], 
                      help='模型输入图像尺寸，默认(320, 320)')
    
    args = parser.parse_args()
    
    try:
        # 1. 加载模型
        print(f"加载模型: {args.model}")
        session, input_name, output_name = load_onnx_model(args.model)
        
        # 2. 预处理图像
        print(f"预处理图像: {args.image}")
        image_array = preprocess_image(args.image, args.input_size)
        
        # 3. 执行推理
        print("开始推理...")
        result = infer_image(session, input_name, output_name, image_array)
        
        # 4. 后处理结果
        print("处理推理结果...")
        post_result = postprocess_result(result, args.labels)
        
        # 5. 输出结果
        print("\n推理结果:")
        print(f"最可能的类别: {post_result['class_name']} (索引: {post_result['class_index']})")
        print(f"置信度: {post_result['probability']:.4f} ({post_result['probability']*100:.2f}%)")
        
    except Exception as e:
        print(f"推理过程出错: {str(e)}")

if __name__ == "__main__":
    main()
