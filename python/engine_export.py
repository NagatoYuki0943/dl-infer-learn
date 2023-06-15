import torch
import onnx
import tensorrt as trt


# load onnx
ONNX_PATH = "./models/shufflenet_v2_x0_5.onnx"
ENGINE_PATH = "./models/shufflenet_v2_x0_5.trt"
onnx_model = onnx.load(ONNX_PATH)

# create builder and network
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

# parse onnx
parser = trt.OnnxParser(network, logger)

if not parser.parse(onnx_model.SerializeToString()):
    error_msgs = ''
    for error in range(parser.num_errors):
        error_msgs += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

# 配置
config = builder.create_builder_config()
# config.max_workspace_size = 1 << 20                  # old
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
# 默认是FP32的处理方法
mode = 'FP32'   # FP32 FP16 INT8(不是所有设备都支持)
if mode == "FP16":
    print("FP16!")
    config.set_flag(trt.BuilderFlag.FP16)
elif mode == "INT8":
    print("INT8!")
    config.set_flag(trt.BuilderFlag.INT8)

profile = builder.create_optimization_profile()
# IR 转换时，如果有多 Batch、多输入、动态 shape 的需求，都可以通过多次调用 set_shape 函数进行设置。
# set_shape 函数接受的传参分别是：输入节点名称，可接受的最小输入尺寸，最优的输入尺寸，可接受的最大输入尺寸。
# 一般要求这三个尺寸的大小关系为单调递增。
profile.set_shape(onnx_model.graph.input[0].name, [1, 3 ,224 ,224], [1, 3,224, 224], [1, 3 ,224 ,224])

config.add_optimization_profile(profile)

# create engine
device = torch.device('cuda:0')
with torch.cuda.device(device):
    # engine = builder.build_engine(network, config)    # old
    engine = builder.build_serialized_network(network, config)


# save engine
with open(ENGINE_PATH, mode='wb') as f:
    # f.write(bytearray(engine.serialize()))            # old
    f.write(bytearray(engine))
    print("============ONNX->TensorRT SUCCESS============")
