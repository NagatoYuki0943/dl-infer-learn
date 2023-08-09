#include <opencv2/opencv.hpp>
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
        cout << index << " = " << score << " => " << name << endl;
        // fprintf(stderr, "%d = %f => %s\n", index, score, name);
    }

    return 0;
}

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
    // 载入模型
    cv::dnn::Net model = cv::dnn::readNetFromONNX(model_path);

    // 推理
    model.setInput(image);
    cv::Mat out_mat = model.forward();
    /******************************** dnn *********************************/

    /**************************** postprocess *****************************/
    int output_size = out_mat.size().height * out_mat.size().width * out_mat.channels();
    // 可以将结果取出放入vector中
    std::vector<float> scores;
    scores.resize(output_size);
    for (int i = 0; i < output_size; i++) {
        scores[i] = out_mat.at<float>(0, i); // at(h, w)
    }
    // vector softmax
    scores = vectorSoftmax(scores);

    out_mat = out_mat.reshape(0, out_mat.size().width); // rows=1 -> 1000, cols=1000->1
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
    assert(classes.size() == out_mat.size[0]);

    // 打印topk
    print_topk(scores, classes, 5);

    return 0;
}
