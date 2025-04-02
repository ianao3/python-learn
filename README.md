# python-learn
| 业务需求 | 技术实现 | 库/工具| 对应要求        |
|------|------| --|-------------|
| 质谱信号实时处理  | 数字滤波+峰值检测	  | NumPy/SciPy  | numpy科学计算   |
| 细胞图像高通量分析  | 并行图像处理  |OpenCV+Dask  | 图像分析/神经网络模型 |
| 实验设备数据接口  | 串口/USB通信  |PySerial/libusb  | 高性能信号处理     |
| 算法部署到嵌入式设备  | Cython交叉编译  |Cython/ARM GCC  | shell脚本等    |

# 部署神经网络模型并使用TensorRT调优指南

TensorRT是NVIDIA提供的高性能深度学习推理优化器和运行时库，可以显著提高模型在NVIDIA GPU上的推理速度。以下是完整的部署流程和调优方法：

## 一、模型部署流程

### 1. 准备工作
- 安装必要组件：
  ```bash
  pip install tensorrt onnx onnxruntime-gpu pycuda
  ```
- 确认已安装正确版本的CUDA和cuDNN（与TensorRT版本匹配）

### 2. 模型转换

#### 从PyTorch到ONNX
```python
import torch
from your_model import YourModel

# 加载训练好的模型
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 创建示例输入
dummy_input = torch.randn(1, 3, 224, 224)  # 根据模型调整输入尺寸

# 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},  # 支持动态batch
    opset_version=11
)
```

#### 从TensorFlow到ONNX
```python
import tf2onnx
import tensorflow as tf

# 加载模型
model = tf.keras.models.load_model('saved_model')

# 转换为ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, output_path="model.onnx")
```

### 3. 使用TensorRT优化

#### 使用trtexec工具（命令行）
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

#### 使用Python API
```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

parser = trt.OnnxParser(network, logger)
with open("model.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # 启用FP16
config.max_workspace_size = 1 << 30  # 1GB

serialized_engine = builder.build_serialized_network(network, config)

with open("model.engine", "wb") as f:
    f.write(serialized_engine)
```

## 二、TensorRT调优技术

### 1. 精度优化
- **FP16模式**：通常能提供2-3倍加速，精度损失很小
  ```python
  config.set_flag(trt.BuilderFlag.FP16)
  ```
- **INT8量化**：更高性能但需要校准
  ```python
  config.set_flag(trt.BuilderFlag.INT8)
  config.int8_calibrator = YourCalibrator()  # 需要实现校准器
  ```

### 2. 性能优化
- **动态形状支持**：
  ```python
  profile = builder.create_optimization_profile()
  profile.set_shape("input", (1,3,224,224), (8,3,224,224), (32,3,224,224))
  config.add_optimization_profile(profile)
  ```
- **层融合**：自动优化，可通过查看引擎信息确认
  ```bash
  trtexec --onnx=model.onnx --exportLayerInfo=layer_info.json
  ```

### 3. 内存优化
- **工作空间大小**：
  ```python
  config.max_workspace_size = 1 << 30  # 1GB
  ```
- **内存池限制**：
  ```python
  runtime = trt.Runtime(logger)
  runtime.max_threads = 4  # 限制线程数
  ```

## 三、推理执行

### 使用TensorRT引擎推理
```python
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# 加载引擎
with open("model.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 创建执行上下文
context = engine.create_execution_context()

# 准备输入输出
input_binding = engine.get_binding_index("input")
output_binding = engine.get_binding_index("output")

# 分配内存
h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(input_binding)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(output_binding)), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

# 执行推理
def infer(input_data):
    np.copyto(h_input, input_data.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output.reshape(output_shape)
```

## 四、高级调优技巧

1. **自定义插件**：对于不支持的层，可以开发TensorRT插件
2. **多流处理**：使用多个CUDA流并行处理请求
3. **批处理策略**：调整最大批处理大小平衡延迟和吞吐量
4. **Profiling**：使用Nsight Systems分析性能瓶颈
   ```bash
   nsys profile -o report --force-overwrite true python your_script.py
   ```

5. **多GPU部署**：使用多实例GPU(MIG)或多进程服务

## 五、常见问题解决

1. **ONNX转换失败**：
   - 确保所有操作在ONNX opset中受支持
   - 简化模型结构，替换自定义层

2. **精度下降**：
   - 尝试禁用FP16/INT8，使用FP32基准
   - 检查校准数据集是否具有代表性

3. **性能未提升**：
   - 检查是否启用了所有优化标志
   - 使用trtexec的--dumpProfile查看各层执行时间

通过以上步骤，您可以将神经网络模型高效部署到NVIDIA GPU上，并利用TensorRT的各种优化技术获得最佳推理性能。