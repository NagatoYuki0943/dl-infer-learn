#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

cv::Mat opencvSoftmax(cv::Mat& mat) {
    double minValue, maxValue;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(mat, &minValue, &maxValue, &minLoc, &maxLoc);
    cv::exp((mat - maxValue), mat);
    float sum = cv::sum(mat)[0];
    mat /= sum;
    return mat;
}

vector<float> vectorSoftmax(vector<float>& scores) {
    float maxValue = *max_element(scores.begin(), scores.end());

    // 减去最大值并求指数
    float temp;
    float sum = 0.0f;
    vector<float> results(scores.size());
    for (int i = 0; i < scores.size(); i++) {
        temp = exp(scores[i] - maxValue);
        sum += temp;
        results[i] = temp;
    }

    // 除以总和
    for (auto& result: results) {
        result /= sum;
    }
    return results;
}

/**
* https://github.com/Tencent/ncnn/blob/master/examples/squeezenet.cpp#LL60C41-L60C41
*/
static int print_topk(const vector<float>& cls_scores, const vector<string>& cls_names, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    vector<pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = make_pair(cls_scores[i], i);
    }

    partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
        greater<pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        string name = cls_names[index];
        fprintf(stderr, "%d = %f => %s\n", index, score, name.c_str());
    }

    return 0;
}

/**
* https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx
*/
int main() {
    cout << "onnxruntime version: " << Ort::GetVersionString() << endl; // 1.16.0

    string image_path = "../../../../../cat.jpg";
    string classes_name_path = "../../../../../imagenet_class_index.txt";

    string model_path = "../../../../../models/shufflenet_v2_x0_5-dynamic_batch.onnx";
    int threads = 4;
    string device = "cuda";
    int gpu_mem_limit = 4;

    cv::Mat image = cv::imread(image_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    /***************************** preprocess *****************************/
    // resize
    cv::resize(image, image, { 224, 224 });
    // 转换为float并归一化
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f, 0);
    // 标准化
    cv::Scalar mean_scalar = cv::Scalar(0.485f, 0.456f, 0.406f);
    cv::Scalar std_scalar = cv::Scalar(0.229f, 0.224f, 0.225f);
    cv::subtract(image, mean_scalar, image);
    cv::divide(image, std_scalar, image);

    // [H, W, C] -> [N, C, H, W]
    image = cv::dnn::blobFromImage(image);
    /***************************** preprocess *****************************/

    /**************************** onnxruntime *****************************/
    Ort::Env env{};                                         // 三个ort参数
    Ort::AllocatorWithDefaultOptions allocator{};
    Ort::RunOptions runOptions{};
    Ort::Session session = Ort::Session(nullptr);           // onnxruntime session
    size_t input_nums{};                                    // 模型输入值数量
    size_t output_nums{};                                   // 模型输出值数量
    vector<const char*> input_node_names;                   // 输入节点名
    vector<Ort::AllocatedStringPtr> input_node_names_ptr;   // 输入节点名指针,保存它防止释放 https://github.com/microsoft/onnxruntime/issues/13651
    vector<vector<int64_t>> input_dims;                     // 输入形状
    vector<const char*> output_node_names;                  // 输出节点名
    vector<Ort::AllocatedStringPtr> output_node_names_ptr;  // 输入节点名指针
    vector<vector<int64_t>> output_dims;                    // 输出形状

    // 获取可用的provider
    auto availableProviders = Ort::GetAvailableProviders();
    for (const auto& provider : availableProviders) {
        cout << provider << " ";
    }
    cout << endl;

    /********************* load onnx *********************/
    Ort::SessionOptions sessionOptions;
    // 使用0个线程执行op,若想提升速度，增加线程数
    sessionOptions.SetIntraOpNumThreads(threads);
    sessionOptions.SetInterOpNumThreads(threads);
    // ORT_ENABLE_ALL: 启用所有可能的优化
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (device == "cuda" || device == "tensorrt") {
        // https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
        // https://onnxruntime.ai/docs/api/c/struct_ort_c_u_d_a_provider_options.html
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = (size_t)gpu_mem_limit * 1024 * 1024 * 1024; // gpu memory limit
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        if (device == "tensorrt") {
            // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
            // https://onnxruntime.ai/docs/api/c/struct_ort_tensor_r_t_provider_options.html
            OrtTensorRTProviderOptions trt_options;
            trt_options.device_id = 0;
            trt_options.trt_max_workspace_size = (size_t)gpu_mem_limit * 1024 * 1024 * 1024; // gpu memory limit
            trt_options.trt_fp16_enable = 0;
            sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
        }
    }
    wchar_t* model_path1 = new wchar_t[model_path.size()];
    swprintf(model_path1, model_path.size(), L"%S", model_path.c_str());
    // create session
    session = Ort::Session(env, model_path1, sessionOptions);
    /********************* load onnx *********************/

    /********************* onnx info *********************/
    input_nums = session.GetInputCount();
    output_nums = session.GetOutputCount();
    printf("Number of inputs = %zu\n", input_nums); // Number of inputs = 1
    printf("Number of output = %zu\n", output_nums);// Number of output = 1

    // 获取输入输出name
    // 获取维度数量
    for (int i = 0; i < input_nums; i++) {
        // 输入变量名
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        input_node_names_ptr.push_back(move(input_name));

        // 输入形状
        auto input_shape_info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        input_dims.push_back(input_shape_info.GetShape());
    }

    for (int i = 0; i < output_nums; i++) {
        // 输出变量名
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
        output_node_names_ptr.push_back(move(output_name));

        // 输出形状
        auto output_shape_info = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        output_dims.push_back(output_shape_info.GetShape());
    }

    for (int i = 0; i < input_nums; ++i) {
        cout << "input_dims: [ ";
        for (auto& j : input_dims[i]) {
            cout << j << " ";
            // support dynamic batch
            if (j == -1) {
                cout << "(dynamic shape, -1 => 1) ";
                j = 1;
            }
        }
        cout << "]" << endl;
    }

    for (int i = 0; i < output_nums; ++i) {
        cout << "output_dims: [ ";
        for (auto& j : output_dims[i]) {
            cout << j << " ";
            // support dynamic batch
            if (j == -1){
                cout << "(dynamic shape, -1 => 1) ";
                j = 1;
            }
        }
        cout << "]" << endl;
    }

    // classes length
    int output_size = 1;
    for (auto dim : output_dims[0]) {
        output_size *= dim;
    }
    /********************* onnx info *********************/

    // 申请内存空间
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // 创建输入tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, image.ptr<float>(), image.total(), input_dims[0].data(), input_dims[0].size());
    // 推理 只传递输入
    vector<Ort::Value> output_tensor;
    try {
        output_tensor = session.Run(
            runOptions,
            input_node_names.data(),
            &input_tensor,
            input_nums,
            output_node_names.data(),
            output_nums
        );
    }
    catch (Ort::Exception& e) {
        cout << e.what() << endl;
    }

    // 获取输出
    float* floatarr = output_tensor[0].GetTensorMutableData<float>();
    /**************************** onnxruntime *****************************/

    /**************************** postprocess *****************************/
    // 可以将结果取出放入vector中
    vector<float> scores(floatarr, floatarr + output_size);
    // vector softmax
    scores = vectorSoftmax(scores);

    // float*转化为cv::Mat rows=1000, cols=1
    cv::Mat out_mat = cv::Mat(output_size, 1, CV_32FC1, floatarr);
    // opencv softmax
    // out_mat = opencvSoftmax(out_mat);
    /**************************** postprocess *****************************/

    // 读取classes name
    ifstream infile;
    infile.open(classes_name_path);
    string l;
    vector<string> classes;
    while (std::getline(infile, l)) {
        classes.push_back(l);
    }
    infile.close();
    // 确保模型输出长度和classes长度相同
    assert(classes.size() == output_size);

    // 打印topk
    print_topk(scores, classes, 5);

    return 0;
}
