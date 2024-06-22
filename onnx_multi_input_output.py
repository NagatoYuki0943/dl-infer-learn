import torch
from torch import nn
from torch import Tensor
from typing import Tuple
import onnx
import onnxslim
import onnxruntime as ort
import numpy as np


class Math(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """两个输入,两个输出

        Args:
            x (Tensor):
            y (Tensor):

        Returns:
            Tuple[Tensor, Tensor]:
        """
        return (x + y), (x - y)


a = torch.ones(1, 2)
b = torch.ones(1, 2)
cdist = Math()
c, d = cdist(a, b)
print(c)
# tensor([[2., 2.]])
print(d)
# tensor([[0., 0.]])


onnx_path = './models/Math.onnx'
torch.onnx.export(
    cdist,
    (a, b),     # 设置多个输入必须使用tuple
    onnx_path,
    opset_version=11,
    input_names =["a", "b"],
    output_names=["c", "d"],
)
model_ = onnx.load(onnx_path)
# 简化模型
model_simp = onnxslim.slim(model_)
onnx.save(model_simp, onnx_path)
print("export onnx success!")


## onnxruntime
net = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
# net = ort.InferenceSession(onnx_path1, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': 0}])

a = np.ones((1, 2), dtype=np.float32)
b = np.ones((1, 2), dtype=np.float32)

input_names  = net.get_inputs()
input_name1  = input_names[0].name
input_name2  = input_names[1].name
print(input_name1, input_name2)     # a b
output_names = net.get_outputs()
output_name1 = output_names[0].name
output_name2 = output_names[1].name
print(output_name1, output_name2)   # c d


# 设置多个输入(字典), 返回值总是一个list
out = net.run(None,{input_name1: a, input_name2: b})
print(type(out))    # <class 'list'>
print(out[0])
# [[2. 2.]]
print(out[1])
# [[0. 0.]]
