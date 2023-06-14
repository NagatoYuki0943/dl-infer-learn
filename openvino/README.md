# ncnn shufflenetv2 infer

# download libraries

# onnx2openvino

> 安装 openvino-dev
>
> `pip install openvino-dev`

```sh
# 转换模型
mo --input_model onnx_path --output_dir openvino_dir --compress_to_fp16 # 转换为fp16格式,可选
```



## opencv

> https://opencv.org

```yaml
# 环境变量
$opencv_path\build\x64\vc16\bin
```

## openvino

> https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html
>
> openvino文档



```yaml
# 环境变量
$openvino_path\runtime\bin\intel64\Debug
$openvino_path\runtime\bin\intel64\Release
$openvino_path\runtime\3rdparty\tbb\bin
```

# 修改 `CMakeLists.txt` 中的路径