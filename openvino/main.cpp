#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <fstream>

using namespace std;


/**
 * https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw
 * https://blog.csdn.net/sandmangu/article/details/107181289
 * https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html
 * 
 * input(0)/output(0) ����id��ָ���������������ָ����ȫ�����������
 *
 *  input().tensor()       ��7������
 *  ppp.input().tensor().set_color_format().set_element_type().set_layout()
 *                      .set_memory_type().set_shape().set_spatial_dynamic_shape().set_spatial_static_shape();
 *
 *  output().tensor()      ��2������
 *  ppp.output().tensor().set_layout().set_element_type();
 *
 *  input().preprocess()   ��8������
 *  ppp.input().preprocess().convert_color().convert_element_type().mean().scale()
 *                          .convert_layout().reverse_channels().resize().custom();
 *
 *  output().postprocess() ��3������
 *  ppp.output().postprocess().convert_element_type().convert_layout().custom();
 *
 *  input().model()  ֻ��1������
 *  ppp.input().model().set_layout();
 *
 *  output().model() ֻ��1������
 *  ppp.output().model().set_layout();
 **/


/**
* 
*/
int main() {
    string image_path = "../../../../cat.jpg";
    string model_path = "../../../../shufflenet_v2_x0_5.xml";
    string classes_name_path = "../../../../imagenet_class_index.txt";
    string device = "CPU"; // CPU GPU
    bool openvino_preprocess = false;

    cv::Mat image = cv::imread(image_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    /***************************** preprocess *****************************/
    // resize
    cv::resize(image, image, { 224, 224 });

    if (!openvino_preprocess) {
        // ת��Ϊfloat����һ��
        image.convertTo(image, CV_32FC3, 1.0 / 255, 0);
        // ��׼��
        vector<float> mean = { 0.485, 0.456, 0.406 };
        vector<float> std = { 0.229, 0.224, 0.225 };
        cv::Scalar mean_scalar = cv::Scalar(mean[0], mean[1], mean[2]);
        cv::Scalar std_scalar = cv::Scalar(std[0], std[1], std[2]);
        cv::subtract(image, mean_scalar, image);
        cv::divide(image, std_scalar, image);
        cout << image << endl;
        // [H, W, C] -> [N, C, H, W]
        image = cv::dnn::blobFromImage(image);
    }
    else {
        image.convertTo(image, CV_32FC3, 1.0, 0);
    }
    /***************************** preprocess *****************************/

    /****************************** openvino ******************************/
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    ov::CompiledModel compiled_model;
    if (openvino_preprocess) {
        vector<float> mean = { 0.485 * 255, 0.456 * 255, 0.406 * 255 };
        vector<float> std = { 0.229 * 255, 0.224 * 255, 0.225 * 255 };

        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

        // Specify input image format
        ppp.input(0).tensor()
            .set_color_format(ov::preprocess::ColorFormat::RGB)     // BGR -> RGB
            .set_element_type(ov::element::f32)                     // u8 -> f32
            .set_layout(ov::Layout("HWC"));                         // HWC NHWC NCHW

        // Specify preprocess pipeline to input image without resizing
        ppp.input(0).preprocess()
            //  .convert_color(ov::preprocess::ColorFormat::RGB)
            //  .convert_element_type(ov::element::f32)
            .mean(mean)
            .scale(std);

        // Specify model's input layout
        ppp.input(0).model().set_layout(ov::Layout("NCHW"));

        // Specify output results format
        ppp.output(0).tensor().set_element_type(ov::element::f32);

        // Embed above steps in the graph
        model = ppp.build();
    }

    compiled_model = core.compile_model(model, device);                     // ����õ�ģ��
    ov::InferRequest infer_request = compiled_model.create_infer_request(); // ��������

    vector<ov::Output<const ov::Node>> inputs = compiled_model.inputs();    // ģ�͵������б�����
    vector<ov::Output<const ov::Node>> outputs = compiled_model.outputs();  // ģ�͵�����б�����
    // classes length
    int output_size = 1;
    for (auto i : outputs[0].get_shape()) {
        output_size *= i;
    }

    // ����tensor
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input(0).get_element_type(),
        compiled_model.input(0).get_shape(), (float*)image.data);

    // ����
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    // ��ȡ���
    ov::Tensor result = infer_request.get_output_tensor(0);
    float* floatarr = result.data<float>();

    /****************************** openvino ******************************/

    /**************************** postprocess *****************************/
    // ���Խ����ȡ������vector��
    std::vector<float> scores;
    scores.resize(output_size);
    for (int i = 0; i < output_size; i++)
    {
        scores[i] = floatarr[i];
    }

    // ��ncnn::Matת��Ϊcv::Mat
    cv::Mat out_mat = cv::Mat(output_size, 1, CV_32FC1, floatarr);

    // softmax
    double minValue, maxValue;
    int minLoc, maxLoc;
    cv::minMaxIdx(out_mat, &minValue, &maxValue, &minLoc, &maxLoc);
    cv::exp((out_mat - maxValue), out_mat);
    float sum = cv::sum(out_mat)[0];
    out_mat /= sum;
    cv::minMaxIdx(out_mat, &minValue, &maxValue, &minLoc, &maxLoc);

    /**************************** postprocess *****************************/

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
