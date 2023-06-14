#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <iostream>
#include <fstream>

using namespace std;

/**
* https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet.zh
*/
int main() {
    string image_path = "../../../../cat.jpg";
    string param_path = "../../../../shufflenet_v2_x0_5-sim-opt.param";
    string model_path = "../../../../shufflenet_v2_x0_5-sim-opt.bin";
    string classes_name_path = "../../../../imagenet_class_index.txt";

    cv::Mat image = cv::imread(image_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    /***************************** preprocess *****************************/
    // resize
    cv::resize(image, image, { 224, 224 });
    // 转换为float并归一化
    image.convertTo(image, CV_32FC3, 1.0 / 255, 0);
    // 标准化
    vector<float> mean = { 0.485, 0.456, 0.406 };
    vector<float> std = { 0.229, 0.224, 0.225 };
    cv::Scalar mean_scalar = cv::Scalar(mean[0], mean[1], mean[2]);
    cv::Scalar std_scalar = cv::Scalar(std[0], std[1], std[2]);
    cv::subtract(image, mean_scalar, image);
    cv::divide(image, std_scalar, image);
    /***************************** preprocess *****************************/

    /******************************** ncnn ********************************/
    ncnn::Net net;
    net.load_param(param_path.c_str());
    net.load_model(model_path.c_str());
    
    // 输入输出名字
    vector<const char*> input_names = net.input_names();
    cout << "input numbers = " << input_names.size() << endl << "input name: ";
    for (auto name : input_names) {
        cout << name << " ";
    };
    cout << endl;
    vector<const char*> output_names = net.output_names();
    cout << "output numbers = " << output_names.size() << endl << "input name: ";
    for (auto name : output_names) {
        cout << name << " ";
    };
    cout << endl;

    // 推理器
    ncnn::Extractor ex = net.create_extractor();
    // light mode
    ex.set_light_mode(true);
    // 多线程加速
    ex.set_num_threads(8);

    // 推理
    ncnn::Mat in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, image.size[0], image.size[1]);
    ncnn::Mat out;
    ex.input(input_names[0], in);
    ex.extract(output_names[0], out);
    // 获取输出
    ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);

    /******************************** ncnn ********************************/

    /**************************** postprocess *****************************/
    // 可以将结果取出放入vector中
    std::vector<float> scores;
    scores.resize(out_flatterned.w);
    for (int j = 0; j < out_flatterned.w; j++)
    {
        scores[j] = out_flatterned[j];
    }

    // 将ncnn::Mat转化为cv::Mat
    cv::Mat out_mat = cv::Mat(out_flatterned.w, out_flatterned.h, CV_32FC1, out_flatterned.data);

    // softmax
    double minValue, maxValue;
    int minLoc, maxLoc;
    cv::minMaxIdx(out_mat, &minValue, &maxValue, &minLoc, &maxLoc);
    cv::exp((out_mat - maxValue), out_mat);
    float sum = cv::sum(out_mat)[0];
    out_mat /= sum;
    cv::minMaxIdx(out_mat, &minValue, &maxValue, &minLoc, &maxLoc);

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

    cout << maxLoc << " => " << maxValue << " => " << classes[maxLoc] << endl;
    return 0;
}
