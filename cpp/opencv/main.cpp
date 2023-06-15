#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;

/**
*
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
    image.convertTo(image, CV_32FC3, 1.0 / 255, 0);
    // 标准化
    cv::Scalar mean_scalar = cv::Scalar(0.485, 0.456, 0.406);
    cv::Scalar std_scalar = cv::Scalar(0.229, 0.224, 0.225);
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

    cout << maxLoc << " => " << maxValue << " => " << classes[maxLoc.y] << endl;
    // 285 => 0.374837 => Egyptian_cat

    return 0;
}
