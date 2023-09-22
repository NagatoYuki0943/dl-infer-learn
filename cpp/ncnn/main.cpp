#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ncnn/net.h>
#if NCNN_VULKAN
#include <ncnn/gpu.h>
#endif // NCNN_VULKAN


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
* https://github.com/Tencent/ncnn/wiki
* https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet.zh
*/
int main() {
    cout << NCNN_VERSION_STRING << endl;    // 1.0.20230816

    string image_path = "../../../../../cat.jpg";
    string classes_name_path = "../../../../../imagenet_class_index.txt";

    string param_path = "../../../../../models/shufflenet_v2_x0_5-ncnn/shufflenet_v2_x0_5-sim-opt.param";
    string model_path = "../../../../../models/shufflenet_v2_x0_5-ncnn/shufflenet_v2_x0_5-sim-opt.bin";

    cv::Mat image = cv::imread(image_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    /***************************** preprocess *****************************/
    // resize
    cv::resize(image, image, { 224, 224 });
    // 这里乘以255相当于归一化和标准化同时计算
    float mean[3] = { 0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f };
    // 这里用倒数是因为使用的相乘,而不是触发
    float std[3] = { 1.0f / (0.229f * 255.0f), 1.0f / (0.224f * 255.0f), 1.0f / (0.225f * 255.0f) };

    // 将cv::Mat转换为ncnn::Mat再计算,转换时必须为uint8类型
    ncnn::Mat in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, image.size[0], image.size[1]);
    in.substract_mean_normalize(mean, std);
    /***************************** preprocess *****************************/

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

    {

        /******************************** ncnn ********************************/
        ncnn::Net net;

#if NCNN_VULKAN
        cout << "use vulkan" << endl;
        net.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

        // net.opt.use_fp16_packed = false;
        // net.opt.use_fp16_storage = false;
        // net.opt.use_fp16_arithmetic = false;
        // net.opt.use_bf16_storage = false;

        // 载入模型
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
        cout << "output numbers = " << output_names.size() << endl << "output name: ";
        for (auto name : output_names) {
            cout << name << " ";
        };
        cout << endl;

        // 推理器 每次都重新实例化一个extractor
        // always create Extractor
        // it's cheap and almost instantly !
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
        // 将结果取出放入vector中
        vector<float> scores(out_flatterned.w);
        for (int j = 0; j < out_flatterned.w; j++) {
            scores[j] = out_flatterned[j];
        }
        // vector softmax
        scores = vectorSoftmax(scores);

        // 将ncnn::Mat转化为cv::Mat row=1000, cols=1
        // cv::Mat out_mat = cv::Mat(out_flatterned.w, out_flatterned.h, CV_32FC1, out_flatterned);
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
        assert(classes.size() == scores.size());

        // 打印topk
        print_topk(scores, classes, 5);
    }

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    return 0;
}
