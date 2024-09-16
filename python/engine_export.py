"""TensorRT export https://developer.nvidia.com/tensorrt.

https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py#L653
"""

import onnx
import tensorrt as trt
from pathlib import Path


dynamic = True
half = False
workspace = 2  # TensorRT: workspace size (GB)
shape = [8, 3, 224, 224]  # max dynamic shape

if dynamic:
    f_onnx = Path("../models/shufflenet_v2_x0_5-dynamic_batch.onnx")
else:
    f_onnx = Path("../models/shufflenet_v2_x0_5.onnx")

assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
f_engine = f_onnx.with_suffix(".engine")  # TensorRT engine file

onnx_model = onnx.load(f_onnx)

# create builder and network
logger = trt.Logger(trt.Logger.INFO)  # INFO WARNING ERROR
# logger.min_severity = trt.Logger.Severity.VERBOSE
builder = trt.Builder(logger)
config = builder.create_builder_config()
# config.max_workspace_size = int(workspace * (1 << 30))  # Deprecation
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace * (1 << 30)))
explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(explicit_batch)

# parse onnx
parser = trt.OnnxParser(network, logger)
# if not parser.parse_from_file(str(f_onnx)): # 这样也可以
if not parser.parse(onnx_model.SerializeToString()):
    raise RuntimeError(f"failed to load ONNX file: {f_onnx}")

# get inputs shape
inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]
for inp in inputs:
    print(f'input: "{inp.name}" with shape: {inp.shape} type: {inp.dtype}')
for out in outputs:
    print(f'output: "{out.name}" with shape: {out.shape} type: {out.dtype}')

# dynamic shape
if dynamic:
    if shape[0] <= 1:
        print(f"WARNING ⚠️ 'dynamic=True' model requires max batch size'")
    profile = builder.create_optimization_profile()
    for inp in inputs:
        # IR 转换时，如果有多 Batch、多输入、动态 shape 的需求，都可以通过多次调用 set_shape 函数进行设置。
        # set_shape 函数接受的传参分别是：输入节点名称，可接受的最小输入尺寸，最优的输入尺寸，可接受的最大输入尺寸。
        # 一般要求这三个尺寸的大小关系为单调递增。
        profile.set_shape(
            inp.name, (1, *shape[1:]), (max(1, shape[0] // 2), *shape[1:]), shape
        )
    config.add_optimization_profile(profile)

# fp16 type
print(
    f"building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f_engine}"
)
if builder.platform_has_fast_fp16 and half:
    config.set_flag(trt.BuilderFlag.FP16)

# Write file
# with builder.build_engine(network, config) as engine, open(f_engine, "wb") as t:  # old
with builder.build_serialized_network(network, config) as engine, open(
    f_engine, "wb"
) as t:
    # Model
    # t.write(engine.serialize())  # old
    t.write(engine)
    print(f'convert "{f_onnx}" to "{f_engine}" Done!')
