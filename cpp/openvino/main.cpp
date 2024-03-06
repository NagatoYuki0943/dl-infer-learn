#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

/**
 *  https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw
 *  https://blog.csdn.net/sandmangu/article/details/107181289
 *  https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html
 *
 *  input(0)/output(0) 按照id找指定的输入输出，不指定找全部的输入输出
 *
 *  input().tensor()       有7个方法
 *  ppp.input().tensor().set_color_format().set_element_type().set_layout()
 *                      .set_memory_type().set_shape().set_spatial_dynamic_shape().set_spatial_static_shape();
 *
 *  output().tensor()      有2个方法
 *  ppp.output().tensor().set_layout().set_element_type();
 *
 *  input().preprocess()   有8个方法
 *  ppp.input().preprocess().convert_color().convert_element_type().mean().scale()
 *                          .convert_layout().reverse_channels().resize().custom();
 *
 *  output().postprocess() 有3个方法
 *  ppp.output().postprocess().convert_element_type().convert_layout().custom();
 *
 *  input().model()  只有1个方法
 *  ppp.input().model().set_layout();
 *
 *  output().model() 只有1个方法
 *  ppp.output().model().set_layout();
 **/

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
*
*/
int main() {
    cout << ov::get_openvino_version() << endl;
    //OpenVINO Runtime
    //    Version : 2023.1.0
    //    Build : 2023.1.0 - 12185 - 9e6b00e51cd - releases / 2023 / 1

    string image_path = "../../../../../cat.jpg";
    string classes_name_path = "../../../../../imagenet_class_index.txt";

    string model_path = "../../../../../models/shufflenet_v2_x0_5-dynamic-half-openvino/model.xml";
    string device = "CPU";              // CPU or GPU or GPU.0
    bool openvino_preprocess = true;    // 是否使用openvino图片预处理,使用dynamic shape必须要用openvino_preprocess

    cv::Mat image = cv::imread(image_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    /***************************** preprocess *****************************/
    // resize
    cv::resize(image, image, { 224, 224 });

    if (!openvino_preprocess) {
        // 转换为float并归一化
        image.convertTo(image, CV_32FC3, 1.0f / 255.0f, 0);
        // 标准化
        cv::Scalar mean_scalar = cv::Scalar(0.485f, 0.456f, 0.406f);
        cv::Scalar std_scalar = cv::Scalar(0.229f, 0.224f, 0.225f);
        cv::subtract(image, mean_scalar, image);
        cv::divide(image, std_scalar, image);
        // [H, W, C] -> [N, C, H, W]
        image = cv::dnn::blobFromImage(image);
    }
    /***************************** preprocess *****************************/

    /****************************** openvino ******************************/
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    ov::CompiledModel compiled_model;
    if (openvino_preprocess) {
        vector<float> mean = { 0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f };
        vector<float> std = { 0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f };

        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

        // Specify input image format
        ppp.input(0).tensor()
            .set_color_format(ov::preprocess::ColorFormat::RGB)     // BGR -> RGB
            .set_element_type(ov::element::u8)
            .set_layout(ov::Layout("HWC"));                         // HWC NHWC NCHW

        // Specify preprocess pipeline to input image without resizing
        ppp.input(0).preprocess()
            // .convert_color(ov::preprocess::ColorFormat::RGB)
            .convert_element_type(ov::element::f32)
            .mean(mean)
            .scale(std);

        // Specify model's input layout
        ppp.input(0).model().set_layout(ov::Layout("NCHW"));

        // Specify output results format
        for (int i = 0; i < model->get_output_size(); i++)
            ppp.output(i).tensor().set_element_type(ov::element::f32);

        // Embed above steps in the graph
        model = ppp.build();
    }

    compiled_model = core.compile_model(model, device);                     // 编译好的模型
    ov::InferRequest infer_request = compiled_model.create_infer_request(); // 推理请求

    vector<ov::Output<const ov::Node>> inputs = compiled_model.inputs();    // 模型的输入列表名称
    vector<ov::Output<const ov::Node>> outputs = compiled_model.outputs();  // 模型的输出列表名称

    // 打印输入输出形状
    //dynamic shape model without openvino_preprocess coundn't print input and output shape
    for (auto input : inputs) {
        cout << "Input: " << input.get_any_name() << ": [ ";
        for (auto j : input.get_shape()) {
            cout << j << " ";
        }
        cout << "] ";
        cout << "dtype: " << input.get_element_type() << endl;
    }

    for (auto output : outputs) {
        cout << "Output: " << output.get_any_name() << ": [ ";
        for (auto j : output.get_shape()) {
            cout << j << " ";
        }
        cout << "] ";
        cout << "dtype: " << output.get_element_type() << endl;
    }

    // classes length
    int output_size = 1;
    for (auto i : outputs[0].get_shape()) {
        output_size *= i;
    }

    // 创建tensor
    ov::Tensor input_tensor = ov::Tensor(
        compiled_model.input(0).get_element_type(), // inputs[0].get_element_type()
        compiled_model.input(0).get_shape(),        // inputs[0]..get_shape()
        (float*)image.data
    );

    // 推理
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    // 获取输出
    ov::Tensor result = infer_request.get_output_tensor(0);
    float* floatarr = result.data<float>();
    /****************************** openvino ******************************/

    /**************************** postprocess *****************************/
    // 可以将结果取出放入vector中
    vector<float> scores(floatarr, floatarr + output_size);
    // vector softmax
    scores = vectorSoftmax(scores);

    // 将ncnn::Mat转化为cv::Mat rows=1000, cols=1
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
