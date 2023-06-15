# ncnn shufflenetv2 infer

# convert onnx2ncnn

## 方式1

> 使用ncnn bin目录下的 `onnx2ncnn.exe` 转换

```sh
.\onnx2ncnn.exe shufflenet_v2_x0_5.onnx shufflenet_v2_x0_5.param shufflenet_v2_x0_5.bin
```

## 方式2

> 在线转换 https://convertmodel.com/

# download libraries

## opencv

> https://opencv.org

```yaml
# 环境变量
$opencv_path\build\x64\vc16\bin
```

## ncnn

> https://github.com/Tencent/ncnn/releases



```yaml
# 环境变量
$ncnn_path\bin
```

## vulkan

> https://vulkan.lunarg.com/

```yaml
# 环境变量
$VulkanSDK\Bin
```

# 修改 `CMakeLists.txt` 中的路径