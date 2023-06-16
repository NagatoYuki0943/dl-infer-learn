#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "logger.h"
#include "util.h"
#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

using namespace std;

/**
* https://github.com/NVIDIA/TensorRT/blob/main/quickstart/SemanticSegmentation/tutorial-runtime.cpp
*
* https://github.com/NVIDIA/TensorRT/tree/main/quickstart/common
*/
int main() {
    string image_path = "../../../../../cat.jpg";
    string model_path = "../../../../../models/shufflenet_v2_x0_5.onnx";
    string classes_name_path = "../../../../../imagenet_class_index.txt";

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

    /******************************** dnn *********************************/
    cv::dnn::Net model = cv::dnn::readNetFromONNX(model_path);
    model.setInput(image);
    cv::Mat out_mat = model.forward();
    /******************************** dnn *********************************/

    /**************************** postprocess *****************************/
    int output_size = out_mat.size().height * out_mat.size().width * out_mat.channels();
    out_mat = out_mat.reshape(0, out_mat.size().width); // rows=1 -> 1000, cols=1000->1

    // 可以将结果取出放入vector中
    std::vector<float> scores;
    scores.resize(output_size);
    for (int i = 0; i < output_size; i++)
    {
        scores[i] = out_mat.at<float>(i, 0); // at(h, w)
    }

    // softmax
    double minValue, maxValue;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(out_mat, &minValue, &maxValue, &minLoc, &maxLoc);
    cv::exp((out_mat - maxValue), out_mat);
    float sum = cv::sum(out_mat)[0];
    out_mat /= sum;
    cv::minMaxLoc(out_mat, &minValue, &maxValue, &minLoc, &maxLoc);
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
    assert(classes.size() == out_mat.size[0]);

    cout << maxLoc.y << " => " << maxValue << " => " << classes[maxLoc.y] << endl;
    // 285 => 0.374837 => Egyptian_cat

    return 0;
}
