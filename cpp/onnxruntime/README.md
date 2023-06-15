# onnxruntime shufflenetv2 infer

# onnxruntime官方例子

> https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx
>
> 推荐 `squeezenet/main.cpp`

# download libraries

## opencv

> https://opencv.org

```yaml
# 环境变量
$opencv_path\build\x64\vc16\bin
```

## onnxruntime

> onnxruntime下载地址 https://github.com/microsoft/onnxruntime/releases
>
> onnxruntime文档 https://onnxruntime.ai/docs/
>
> onnxruntime使用gpu要安装cuda和cudnn
>
> https://developer.nvidia.com/cuda-toolkit
>
> https://developer.nvidia.cn/zh-cn/cudnn



```yaml
# 环境变量
$onnxruntime_path\lib
```



# 修改 `CMakeLists.txt` 中的路径

# 错误

### 0xc000007b 0xC000007B

如果程序无法运行，将`onnxruntime\lib`下的`*.dll`文件复制到exe目录下可以解决