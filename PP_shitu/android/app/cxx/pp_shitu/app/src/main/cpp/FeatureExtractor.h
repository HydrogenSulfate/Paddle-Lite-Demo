// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "Utils.h"            // NOLINT
#include "paddle_api.h"       // NOLINT
#include <arm_neon.h>         // NOLINT
#include <fstream>            // NOLINT
#include <iostream>           // NOLINT
#include <math.h>             // NOLINT
#include <memory>             // NOLINT
#include <opencv2/opencv.hpp> // NOLINT
#include <stdlib.h>           // NOLINT
#include <string>             // NOLINT
#include <sys/time.h>         // NOLINT
#include <vector>             // NOLINT
#include "numeric"
using namespace paddle::lite_api; // NOLINT
using namespace std;              // NOLINT

class FeatureExtract
{
public: // NOLINT
    explicit FeatureExtract(std::string model_path, std::vector<int> input_shape, int cpu_nums,
                            std::string cpu_power)
    {
        MobileConfig config;
        config.set_threads(cpu_nums);
        config.set_power_mode(ParsePowerMode(cpu_power));
        config.set_model_from_file(model_path);
        this->predictor_ = CreatePaddlePredictor<MobileConfig>(config);
//        LoadLabel(label_path);
//        warm_up_ = warm_up;
//        repeats_ = repeats;
//        size_ = input_shape[2];
    }

//    void LoadLabel(std::string path)
//    {
//        std::ifstream file;
//        std::vector <std::string> label_list;
//        file.open(path);
//        while (file)
//        {
//            std::string line;
//            std::getline(file, line);
//            std::string::size_type pos = line.find(" ");
//            if (pos != std::string::npos)
//            {
//                line = line.substr(pos);
//            }
//            this->label_list_.push_back(line);
//        }
//        file.clear();
//        file.close();
//    }

    void RunRecModel(const cv::Mat &img, double &cost_time, std::vector<float> &feature); // NOLINT
    void FeatureNorm(std::vector<float> &feature);

//    std::vector <RectResult> PostProcess(const float *output_data, int output_size, cv::Mat &output_image); // NOLINT

private: // NOLINT
    std::shared_ptr <PaddlePredictor> predictor_;
//    std::vector <std::string> label_list_;
    std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
    double scale_ = 1 / 255.0; // 1/255.0
    int size = 224;
//    int topk_ = 5;
//    int warm_up_;
//    int repeats_;

    cv::Mat ResizeImage(const cv::Mat img);
};