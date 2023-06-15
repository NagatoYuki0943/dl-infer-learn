#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <iostream>
#include <fstream>

using namespace std;

/**
* https://github.com/Tencent/ncnn/wiki
* https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet.zh
*/
int main() {
    string image_path = "../../../../../cat.jpg";
    string param_path = "../../../../../models/shufflenet_v2_x0_5-sim-opt.param";
    string model_path = "../../../../../models/shufflenet_v2_x0_5-sim-opt.bin";
    string classes_name_path = "../../../../../imagenet_class_index.txt";

    cv::Mat image = cv::imread(image_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    /***************************** preprocess *****************************/
    // resize
    cv::resize(image, image, { 224, 224 });
    // 这里乘以255相当于归一化和标准化同时计算
    float mean[3] = { 0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0 };
    // 这里用倒数是因为使用的相乘,而不是触发
    float std[3] = { 1.0 / (0.229 * 255.0), 1.0 / (0.224 * 255.0), 1.0 / (0.225 * 255.0) };

    // 将cv::Mat转换为ncnn::Mat再计算,转换时必须为uint8类型
    ncnn::Mat in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, image.size[0], image.size[1]);
    in.substract_mean_normalize(mean, std);
    /***************************** preprocess *****************************/

    /******************************** ncnn ********************************/
    ncnn::Net net;
    net.load_param(param_path.c_str());
    net.load_model(model_path.c_str());

    // net.opt.use_fp16_packed = false;
    // net.opt.use_fp16_storage = false;
    // net.opt.use_fp16_arithmetic = false;
    // net.opt.use_bf16_storage = false;

    // 输入输出名字
    vector<const char*> input_names = net.input_names();
    cout << "input numbers = " << input_names.size() << endl << "input name: ";
    for (auto name : input_names) {
        cout << name << " ";
    };
    cout << endl;
    vector<const char*> output_names = net.output_names();
    cout << "output numbers = " << output_names.size() << endl << "output name: ";
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

    // 将ncnn::Mat转化为cv::Mat row=1000, cols=1
    cv::Mat out_mat = cv::Mat(out_flatterned.w, out_flatterned.h, CV_32FC1, out_flatterned);

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
    // 285 => 0.374477 => Egyptian_cat

    return 0;
}
