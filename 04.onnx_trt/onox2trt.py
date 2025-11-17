import tensorrt as trt
import argparse
import os

# 配置TRT日志
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def onnx_to_trt(onnx_path, trt_engine_path, precision="fp16", workspace_size=1 << 30,
                min_shape=(1, 3, 320, 320),  # 最小输入形状 (batch, channel, height, width)
                opt_shape=(1, 3, 320, 320),  # 最优输入形状（建议与实际推理时一致）
                max_shape=(8, 3, 320, 320)): # 最大输入形状（需覆盖可能的输入范围）
    """
    将带动态输入形状的ONNX模型转换为TensorRT引擎
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX模型文件不存在: {onnx_path}")
    
    try:
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # 解析ONNX模型
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("ONNX模型解析失败，错误信息：")
                for error in range(parser.num_errors):
                    print(f"  {parser.get_error(error)}")
                return False
        
        # 关键修复：为动态输入创建优化配置文件
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        
        # 1. 创建优化配置文件
        profile = builder.create_optimization_profile()
        # 2. 获取模型输入名称（默认取第一个输入）
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        # 3. 为输入绑定形状范围（min/opt/max）
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        # 4. 将配置文件添加到构建配置中
        config.add_optimization_profile(profile)
        
        # 配置精度模式
        if precision.lower() == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print(f"已启用FP16精度加速")
            else:
                print("当前GPU不支持FP16快速计算，自动回退到FP32")
        print(f"最终使用精度模式: {'FP16' if config.get_flag(trt.BuilderFlag.FP16) else 'FP32'}")
        
        # 构建引擎
        print("开始构建TRT引擎... (此过程可能耗时，请耐心等待)")
        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            print("TRT引擎构建失败")
            return False
        
        # 保存引擎
        with open(trt_engine_path, "wb") as f:
            f.write(serialized_engine)
        print(f"TRT引擎转换成功，已保存至: {trt_engine_path}")
        return True
        
    except Exception as e:
        print(f"转换过程出错: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="支持动态形状的ONNX转TensorRT引擎工具")
    parser.add_argument("--onnx", default=r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res\resnet18_20251030_150116\best_model.onnx", help="输入ONNX模型路径")
    parser.add_argument("--output", default=r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res\resnet18_20251030_150116\best_model.trt", help="输出TRT引擎路径（如 model.trt）")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"], 
                      help="推理精度模式，默认fp16")
    parser.add_argument("--workspace", type=int, default=1, 
                      help="工作空间大小（GB），默认1GB")
    # 动态形状参数（根据你的模型输入维度调整）
    parser.add_argument("--min-shape", type=int, nargs="+", default=[1, 3, 320, 320],
                      help="最小输入形状 (batch, channel, height, width)")
    parser.add_argument("--opt-shape", type=int, nargs="+", default=[1, 3, 320, 320],
                      help="最优输入形状")
    parser.add_argument("--max-shape", type=int, nargs="+", default=[8, 3, 320, 320],
                      help="最大输入形状")
    args = parser.parse_args()
    
    workspace_size = args.workspace * (1 << 30)
    
    onnx_to_trt(
        onnx_path=args.onnx,
        trt_engine_path=args.output,
        precision=args.precision,
        workspace_size=workspace_size,
        min_shape=tuple(args.min_shape),
        opt_shape=tuple(args.opt_shape),
        max_shape=tuple(args.max_shape)
    )


if __name__ == "__main__":
    main()
