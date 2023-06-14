#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>

using namespace std;

/**
* https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx
*/
int main() {
    string image_path = "../../../../cat.jpg";
    string model_path = "../../../../shufflenet_v2_x0_5.onnx";
    string classes_name_path = "../../../../imagenet_class_index.txt";
    int threads = 4;
    string device = "cuda";
    int gpu_mem_limit = 4;

    cv::Mat image = cv::imread(image_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    /***************************** preprocess *****************************/
    // resize
    cv::resize(image, image, { 224, 224 });
    // ת��Ϊfloat����һ��
    image.convertTo(image, CV_32FC3, 1.0 / 255, 0);
    // ��׼��
    vector<float> mean = { 0.485, 0.456, 0.406 };
    vector<float> std = { 0.229, 0.224, 0.225 };
    cv::Scalar mean_scalar = cv::Scalar(mean[0], mean[1], mean[2]);
    cv::Scalar std_scalar = cv::Scalar(std[0], std[1], std[2]);
    cv::subtract(image, mean_scalar, image);
    cv::divide(image, std_scalar, image);

    // [H, W, C] -> [N, C, H, W]
    image = cv::dnn::blobFromImage(image);
    /***************************** preprocess *****************************/

    /**************************** onnxruntime *****************************/
    Ort::Env env{};                                         // ����ort����
    Ort::AllocatorWithDefaultOptions allocator{};
    Ort::RunOptions runOptions{};
    Ort::Session session = Ort::Session(nullptr);           // onnxruntime session
    size_t input_nums{};                                    // ģ������ֵ����
    size_t output_nums{};                                   // ģ�����ֵ����
    vector<const char*> input_node_names;                   // ����ڵ���
    vector<Ort::AllocatedStringPtr> input_node_names_ptr;   // ����ڵ���ָ��,��������ֹ�ͷ� https://github.com/microsoft/onnxruntime/issues/13651
    vector<vector<int64_t>> input_dims;                     // ������״
    vector<const char*> output_node_names;                  // ����ڵ���
    vector<Ort::AllocatedStringPtr> output_node_names_ptr;  // ����ڵ���ָ��
    vector<vector<int64_t>> output_dims;                    // �����״

    // ��ȡ���õ�provider
    auto availableProviders = Ort::GetAvailableProviders();
    for (const auto& provider : availableProviders) {
        cout << provider << " ";
    }
    cout << endl;

    /********************* load onnx *********************/
    Ort::SessionOptions sessionOptions;
    // ʹ��0���߳�ִ��op,���������ٶȣ������߳���
    sessionOptions.SetIntraOpNumThreads(threads);
    sessionOptions.SetInterOpNumThreads(threads);
    // ORT_ENABLE_ALL: �������п��ܵ��Ż�
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
    swprintf(model_path1, 4096, L"%S", model_path.c_str());
    // create session
    session = Ort::Session(env, model_path1, sessionOptions);
    /********************* load onnx *********************/

    /********************* onnx info *********************/
    input_nums = session.GetInputCount();
    output_nums = session.GetOutputCount();
    printf("Number of inputs = %zu\n", input_nums); // Number of inputs = 1
    printf("Number of output = %zu\n", output_nums);// Number of output = 1

    // ��ȡ�������name
    // ��ȡά������
    for (int i = 0; i < input_nums; i++) {
        // ���������
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        input_node_names_ptr.push_back(move(input_name));

        // ������״
        auto input_shape_info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        input_dims.push_back(input_shape_info.GetShape());
    }

    for (int i = 0; i < output_nums; i++) {
        // ���������
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
        output_node_names_ptr.push_back(move(output_name));

        // �����״
        auto output_shape_info = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        output_dims.push_back(output_shape_info.GetShape());
    }

    for (int i = 0; i < input_nums; ++i) {
        cout << "input_dims: ";
        for (const auto j : input_dims[i]) {
            cout << j << " ";
        }
        cout << endl;
    }

    for (int i = 0; i < output_nums; ++i) {
        cout << "output_dims: ";
        for (const auto j : output_dims[i]) {
            cout << j << " ";
        }
        cout << endl;
    }

    // classes length
    int output_size = 1;
    for (auto dim : output_dims[0]) {
        output_size *= dim;
    }
    /********************* onnx info *********************/

    // �����ڴ�ռ�
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // ��������tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, image.ptr<float>(), image.total(), input_dims[0].data(), input_dims[0].size());
    // ���� ֻ��������
    vector<Ort::Value> output_tensors;
    try {
        output_tensors = session.Run(
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

    // ��ȡ���
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    /**************************** onnxruntime *****************************/

    /**************************** postprocess *****************************/
    // ���Խ����ȡ������vector��
    std::vector<float> scores;
    scores.resize(output_size);
    for (int i = 0; i < output_size; i++)
    {
        scores[i] = floatarr[i];
    }

    // float*ת��Ϊcv::Mat
    cv::Mat out_mat = cv::Mat(output_size, 1, CV_32FC1, floatarr);
    // softmax
    double minValue, maxValue;
    int minLoc, maxLoc;
    cv::minMaxIdx(out_mat, &minValue, &maxValue, &minLoc, &maxLoc);
    cv::exp((out_mat - maxValue), out_mat);
    float sum = cv::sum(out_mat)[0];
    out_mat /= sum;
    cv::minMaxIdx(out_mat, &minValue, &maxValue, &minLoc, &maxLoc);

    ///**************************** postprocess *****************************/

    // ��ȡclasses name
    ifstream infile;
    infile.open(classes_name_path);
    string l;
    vector<string> classes;
    while (std::getline(infile, l)) {
        classes.push_back(l);
    }
    infile.close();
    // ȷ��ģ��������Ⱥ�classes������ͬ
    assert(classes.size() == out_mat.size[0]);

    cout << maxLoc << " => " << maxValue << " => " << classes[maxLoc] << endl;
    // 285 => 0.374836 => Egyptian_cat

    return 0;
}
