#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include "logger.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <math.h>
#include <numeric>

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
    for (auto& result : results) {
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


inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}


inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}


/**
*
*/
int main() {
    int trt_version = nvinfer1::kNV_TENSORRT_VERSION_IMPL;
    cout << "trt_version = " << trt_version << endl;    // 8601

    string image_path = "../../../../../cat.jpg";
    string classes_name_path = "../../../../../imagenet_class_index.txt";
    string model_path;

    // 支持dynamic batch infer,需要注意batch为几,就要输入几张图片,这个和python版本不同,python可以输入比batch少的图片数量
    bool dynamic_batch = true;
    // dynamic batch requires explicit specification of batch
    int dynamic_batches = 4;
    if (dynamic_batch) {
        model_path = "../../../../../models/shufflenet_v2_x0_5-dynamic_batch.engine";
    }
    else {
        model_path = "../../../../../models/shufflenet_v2_x0_5.engine";
    }

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

    vector<cv::Mat> images;
    cv::Mat blob;
    if (dynamic_batch) {
        // 初始化dynamic_batches张图片
        images = vector<cv::Mat>(dynamic_batches, image);
        blob = cv::dnn::blobFromImages(images);
    }
    else {
        blob = cv::dnn::blobFromImage(image);
    }


    /***************************** preprocess *****************************/

    /******************************* engine *******************************/
    // https://github.com/linghu8812/tensorrt_inference/
    // https://github.com/linghu8812/tensorrt_inference/blob/master/code/src/model.cpp

    /******************** load engine ********************/
    string cached_engine;
    std::fstream file;
    std::cout << "loading filename from:" << model_path << std::endl;
    file.open(model_path, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        std::cout << "read file error: " << model_path << std::endl;
        cached_engine = "";
    }
    while (file.peek() != EOF) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();

    int cudaCount;
    cudaGetDeviceCount(&cudaCount);
    // cuda set device
    cudaSetDevice(0);
    printf("cudaCount = %d, set device %d\n", cudaCount, 0);

    nvinfer1::IRuntime* trtRuntime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
    initLibNvInferPlugins(&sample::gLogger, "");
    nvinfer1::ICudaEngine* engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size());
    assert(engine != nullptr);
    std::cout << "deserialize done" << std::endl;
    /******************** load engine ********************/

    /********************** binding **********************/
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    int min_batches;
    int max_batches;
    int nbBindings = engine->getNbBindings();
    assert(nbBindings == 2);
    vector<int> bufferSize(nbBindings);
    void* cudaBuffers[2];
    for (int i = 0; i < nbBindings; i++) {
        const char* name;
        int mode;
        nvinfer1::DataType dtype;
        nvinfer1::Dims dims;

        if (trt_version < 8500) {
            mode = engine->bindingIsInput(i);
            name = engine->getBindingName(i);
            dtype = engine->getBindingDataType(i);
            dims = context->getBindingDimensions(i);

            // dynamic batch
            if ((*dims.d == -1) && mode) {
                nvinfer1::Dims minDims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMIN);
                min_batches = minDims.d[0];
                // nvinfer1::Dims optDims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kOPT);
                nvinfer1::Dims maxDims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
                max_batches = maxDims.d[0];
                if (dynamic_batch) {
                    // 设置为最大batch
                    context->setBindingDimensions(i, maxDims);
                }
                else {
                    // 显式设置batch为1
                    context->setBindingDimensions(i, nvinfer1::Dims4(1, maxDims.d[1], maxDims.d[2], maxDims.d[3]));
                }
                dims = context->getBindingDimensions(i);
            }
        }
        else {
            name = engine->getIOTensorName(i);
            mode = int(engine->getTensorIOMode(name));
            // cout << "mode: " << mode << endl; // 0:input or output  1:input  2:output
            dtype = engine->getTensorDataType(name);
            dims = context->getTensorShape(name);

            // dynamic batch
            if ((*dims.d == -1) && (mode == 1)) {
                nvinfer1::Dims minDims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMIN);
                min_batches = minDims.d[0];
                // nvinfer1::Dims optDims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT);
                nvinfer1::Dims maxDims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
                max_batches = maxDims.d[0];
                if (dynamic_batch) {
                    // 设置为最大batch
                    context->setInputShape(name, maxDims);
                }
                else {
                    // 显式设置batch为1
                    context->setInputShape(name, nvinfer1::Dims4(1, maxDims.d[1], maxDims.d[2], maxDims.d[3]));
                }
                dims = context->getTensorShape(name);
            }
        }

        int totalSize = volume(dims) * getElementSize(dtype);
        bufferSize[i] = totalSize;
        cudaMalloc(&cudaBuffers[i], totalSize);

        fprintf(stderr, "name: %s, mode: %d, dims: [%d, %d, %d, %d], totalSize: %d Byte\n", name, mode, dims.d[0], dims.d[1], dims.d[2], dims.d[3], totalSize);
        // name: images, mode : 1, dims : [8, 3, 224, 224] , totalSize : 4816896 Byte
        // name : classes, mode : 2, dims : [8, 1000, 0, 0] , totalSize : 32000 Byte
    }
    /********************** binding **********************/
    if (dynamic_batch)
        assert(dynamic_batches >= min_batches && dynamic_batches <= max_batches);

    /*********************** infer ***********************/
    // float长度
    int inLength = bufferSize[0];
    int outLength = bufferSize[1];
    int outNums = int(bufferSize[1] / sizeof(float)); // sizeof(float) = 4 Byte = 32bit
    if (dynamic_batch) {
        inLength = int(bufferSize[0] / max_batches * images.size());
        outLength = int(bufferSize[1] / max_batches * images.size());
        outNums = int(bufferSize[1] / max_batches * images.size() / sizeof(float));
    }
    cout << "outNums: " << outNums << " nums" << endl;

    vector<float> scores(outNums);
    /****** sync infer ******/
    cudaMemcpy(cudaBuffers[0], blob.ptr<float>(), inLength, cudaMemcpyHostToDevice);
    context->executeV2(cudaBuffers);
    cudaMemcpy(scores.data(), cudaBuffers[1], outLength, cudaMemcpyDeviceToHost);
    /****** sync infer ******/

    ///****** async infer ******/
    //cudaStream_t stream;
    //cudaStreamCreate(&stream);
    //cudaMemcpyAsync(cudaBuffers[0], image.ptr<float>(), inLength, cudaMemcpyHostToDevice, stream);
    //// context->enqueueV2(cudaBuffers, stream, nullptr);
    //context->enqueueV3(stream);
    //cudaMemcpyAsync(scores.data(), cudaBuffers[1], outLength, cudaMemcpyDeviceToHost, stream);
    //cudaStreamSynchronize(stream);
    ///****** async infer ******/
    /*********************** infer ***********************/
    /******************************* engine *******************************/

    /**************************** postprocess *****************************/
    // vector softmax
    scores = vectorSoftmax(scores);
    /**************************** postprocess *****************************/

    // 单张图片结果的长度
    int batch1ResultLength = outNums;
    if (dynamic_batch) {
        batch1ResultLength = outNums / dynamic_batches;
    }

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
    assert(classes.size() == batch1ResultLength);

    // 获取图片的分数
    vector<vector<float>> batchScores = vector<vector<float>>(1, vector<float>(batch1ResultLength, 0));;
    if (dynamic_batch) {
        batchScores = vector<vector<float>>(dynamic_batches, vector<float>(batch1ResultLength, 0));
    }
    for (size_t i = 0; i < outNums; i++) {
        batchScores[i / batch1ResultLength][i % batch1ResultLength] = scores[i];
    };

    // 打印topk
    for (auto& batchScore : batchScores) {
        print_topk(batchScore, classes, 5);
        cout << endl;
    }

    // 析构顺序很重要
    // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html#ab3ace89a0eb08cd7e4b4cba7bedac5a2
    delete context;
    delete engine;
    delete trtRuntime;

    return 0;
}
