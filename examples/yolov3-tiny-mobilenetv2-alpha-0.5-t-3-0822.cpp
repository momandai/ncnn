// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sys/time.h>

#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

using namespace std;

int net_in_size = 352;

string model_param = "yolov3.param";
string model_bin = "yolov3.bin";
ncnn::Net yolov3;


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

cv::Mat paddedResize(cv::Mat &src) {
    int h = src.rows;
    int w = src.cols;
    float ratio = float(net_in_size) / float(max(h, w));
    int resized_h = int(h*ratio);
    int resized_w = int(w*ratio);
    float dw = (net_in_size-resized_w) / 2.0;
    float dh = (net_in_size-resized_h) / 2.0;
    int left_pad = round(dw - 0.1);
    int right_pad = round(dw + 0.1);
    int top_pad = round(dh - 0.1);
    int bottom_pad = round(dh + 0.1);
    cv::Mat img;
    cv::resize(src, img, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_AREA);
    cv::copyMakeBorder(img, img, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    return img;
}

static int detect_yolov3(const cv::Mat& bgr, std::vector<Object>& objects)
{


#if NCNN_VULKAN
    yolov3.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN


    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR, net_in_size, net_in_size);

    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    ncnn::Extractor ex = yolov3.create_extractor();
    ex.set_num_threads(2);
    ex.input(0, in);
    ex.extract("output", out);
    gettimeofday(&t1, NULL);
//    float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
//    std::cout << "time per run is " << mytime << "ms" << endl;

    objects.clear();
    for (int i=0; i<out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
        "person", "face", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(33);
}

int main(int argc, char** argv)
{
//    if (argc != 2)
//    {
//        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
//        return -1;
//    }

    const char* imagepath = argv[1];

    struct timeval t0, t1;


    model_param = string(argv[2]);
    model_bin = string(argv[3]);

    yolov3.load_param(model_param.c_str());
    yolov3.load_model(model_bin.c_str());
//    cv::VideoCapture capture = cv::VideoCapture(0);
    cv::VideoCapture capture = cv::VideoCapture("/home/neptune/github/ultralytics_yolov3/test.mp4");
    cv::Mat frame;
    while (true) {
        capture.read(frame);

        gettimeofday(&t0, NULL);
        cv::Mat padded_resize_img = paddedResize(frame);
//        const unsigned char* value_padded_resize = padded_resize_img.data;
        cv::Mat padded_resize_img_origin = padded_resize_img.clone();
        cv::cvtColor(padded_resize_img, padded_resize_img, cv::COLOR_BGR2RGB);
        gettimeofday(&t1, NULL);
        float resize_time = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        std::cout << "resize time per run is " << resize_time << "ms" << endl;
        std::vector<Object> objects;

        gettimeofday(&t0, NULL);
        detect_yolov3(padded_resize_img, objects);
        gettimeofday(&t1, NULL);
        float detect_time = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        std::cout << "detect time per run is " << detect_time << "ms" << endl;

        gettimeofday(&t0, NULL);
        draw_objects(padded_resize_img_origin, objects);
        gettimeofday(&t1, NULL);
        float draw_time = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        std::cout << "draw time per run is " << draw_time << "ms" << endl;
    }

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    cv::Mat padded_resize_img = paddedResize(m);
    const unsigned char* value_padded_resize = padded_resize_img.data;
    cout << "paddedResize mat: " << endl;
    for (int i = 0; i < 3; i++) {
        cout << (int)(*value_padded_resize) << endl;
        value_padded_resize++;
    }


    cv::cvtColor(padded_resize_img, padded_resize_img, cv::COLOR_BGR2RGB);
//    cv::imshow("padded_resize_img", padded_resize_img);
//    cv::imwrite("padded_resize_bus.jpg", padded_resize_img);
//    cv::waitKey(0);

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

    std::vector<Object> objects;
    detect_yolov3(padded_resize_img, objects);

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    draw_objects(padded_resize_img, objects);

    return 0;
}
