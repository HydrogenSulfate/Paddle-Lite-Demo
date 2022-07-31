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

#include "Pipeline.h"
#include "VectorSearch.h"
#include "ObjectDetector.h"
#include "FeatureExtractor.h"
#include <algorithm>
#include <functional>
#include <utility>


void PrintResult(std::vector <ObjectResult> &det_result,
                 std::shared_ptr <VectorSearch> vector_search,
                 SearchResult &search_result)
{
//    LOGD("%s:\n", img_path.c_str());
    for (int i = 0; i < det_result.size(); ++i)
    {
        int t = i;
        LOGD("\tresult%d: bbox[%d, %d, %d, %d], score: %f, label: %s\n", i,
             det_result[t].rect[0], det_result[t].rect[1], det_result[t].rect[2],
             det_result[t].rect[3], det_result[t].confidence,
             vector_search->GetLabel(search_result.I[search_result.return_k * t])
                     .c_str());
    }
}

void VisualResult(cv::Mat &img, std::vector <ObjectResult> results)
{ // NOLINT
    for (int i = 0; i < results.size(); i++)
    {
        int w = results[i].rect[2] - results[i].rect[0];
        int h = results[i].rect[3] - results[i].rect[1];
        cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[1], w, h);
        cv::rectangle(img, roi, cv::Scalar(255, i * 50, i * 20), 3);
    }
}


PipeLine::PipeLine(std::string det_model_path, std::string rec_model_path,
                   std::string label_path, std::vector<int> det_input_shape,
                   std::vector<int> rec_input_shape, int cpu_num_threads,
                   int warm_up, int repeats, int topk, std::string cpu_power)
{
    det_model_path_ = det_model_path;
    rec_model_path_ = rec_model_path;
    label_path_ = label_path;
    det_input_shape_ = det_input_shape;
    rec_input_shape_ = rec_input_shape;
    cpu_num_threads_ = cpu_num_threads;
//    warm_up_ = warm_up;
//    repeats_ = repeats;
    max_det_num_ = topk;
    cpu_pow_ = cpu_power;
    det_model_path_ =
            det_model_path_ + "/mainbody_PPLCNet_x2_5_640_v1.2_lite.nb";
    rec_model_path_ =
            rec_model_path_ + "/general_PPLCNet_x2_5_lite_v1.2_infer.nb";

    // create object detector
    det_ = std::make_shared<ObjectDetector>(det_model_path_, det_input_shape_,
                                            cpu_num_threads_,
                                            cpu_pow_);

    // create rec model
    rec_ = std::make_shared<FeatureExtract>(rec_model_path_,
                                            rec_input_shape_, cpu_num_threads_,
                                            cpu_pow_);
    searcher_ = std::make_shared<VectorSearch>("app/src/main/assets/index", 5, 0.5);
}

std::string PipeLine::run(std::vector <cv::Mat> &batch_imgs,      // NOLINT
                          std::vector <ObjectResult> &det_result, // NOLINT
                          int batch_size)
{
    std::fill(times_.begin(), times_.end(), 0);

//  DetPredictImage(batch_imgs, &det_result, batch_size, det_, max_det_num_); // det_result获取[l,d,r,u]
    DetPredictImage(batch_imgs, &det_result, batch_size, det_, max_det_num_); // det_result获取[l,d,r,u]
    LOGD("debugINFO: len of det_result %d, len of det_result[0] %d", (int) det_result.size(), (int) det_result[0].rect.size());
    // add the whole image for recognition to improve recall

    int item_start_idx = 0;
    for (int i = 0; i < batch_imgs.size(); i++)
    {
        ObjectResult result_whole_img = {
                {0, 0, batch_imgs[i].cols, batch_imgs[i].rows}, 0, 1.0};
        det_result.push_back(result_whole_img); // 加入整图的坐标，提升召回率
    }
    LOGD("finished det");
    // get rec result
    for (int j = 0; j < det_result.size(); ++j)
    {
        double rec_time = 0.0;    // .rect:vector = {l, d, r, u}
        int w = det_result[j].rect[2] - det_result[j].rect[0];
        int h = det_result[j].rect[3] - det_result[j].rect[1];
        cv::Rect rect(det_result[j].rect[0], det_result[j].rect[1], w, h);
        cv::Mat crop_img = batch_imgs[0](rect);
        LOGD("finished crop_img");
        rec_->RunRecModel(crop_img, rec_time, feature);
        features.insert(features.end(), feature.begin(), feature.end());
//        times_[3] += rec_time[0];
//        times_[4] += rec_time[1];
//        times_[5] += rec_time[2];
    }

    // do vectore search
    SearchResult search_result = searcher_->Search(features.data(), det_result.size());
    for (int j = 0; j < det_result.size(); ++j)
    {
        det_result[j].confidence = search_result.D[search_result.return_k * j];
    }
    float rec_nms_threshold = 0.05;
    NMSBoxes(det_result, searcher_->GetThreshold(), rec_nms_threshold,
             indices);
    LOGD("================== result summary =========================");
    PrintResult(det_result, searcher_, search_result);

    batch_imgs.clear();
    det_result.clear();
    features.clear();
    feature.clear();
    indices.clear();
    // rec nms
//    auto nms_cost0 = GetCurrentUS();
//    nms(&det_result, rec_nms_thresold_, true);
//    auto nms_cost1 = GetCurrentUS();
//    times_[6] += (nms_cost1 - nms_cost0) / 1000.f;

    // results
    std::string res = "";
    res += std::to_string(times_[1] + times_[4]) + "\n";
    for (int i = 0; i < det_result.size(); i++)
    {
        res = res + "class id: " + det_result[i].rec_result[0].class_name + "\n";
    }
//    LOGD("================== benchmark summary ======================");
//    LOGD("ObjectDetect Preprocess:  %8.3f ms", times_[0]);
//    LOGD("ObjectDetect inference:   %8.3f ms", times_[1]);
//    LOGD("ObjectDetect Postprocess: %8.3f ms", times_[2]);
//    LOGD("Recognise Preprocess:     %8.3f ms", times_[3]);
//    LOGD("Recognise inference:      %8.3f ms", times_[4]);
//    LOGD("Recognise Postprocess:    %8.3f ms", times_[5]);
//    LOGD("nms process:              %8.3f ms", times_[6]);
//    PrintResult(det_result);
//    VisualResult(batch_imgs[0], det_result);

//    features.clear();
//    feature.clear();
//    indices.clear();
    return res;
}

void PipeLine::DetPredictImage(const std::vector <cv::Mat> batch_imgs,
                               std::vector <ObjectResult> *im_result,
                               const int batch_size_det,
                               std::shared_ptr <ObjectDetector> det,
                               const int max_det_num)
{
    int steps = ceil(float(batch_imgs.size()) / batch_size_det);
    for (int idx = 0; idx < steps; idx++)
    {
        int left_image_cnt = batch_imgs.size() - idx * batch_size_det;
        if (left_image_cnt > batch_size_det)
        {
            left_image_cnt = batch_size_det;
        }
        // Store all detected result
        std::vector <ObjectResult> result;
        std::vector<int> bbox_num;
        std::vector<double> det_times;

        bool is_rbox = false;
        det->Predict(batch_imgs, 0, 1, &result, &bbox_num, &det_times);
        int item_start_idx = 0;
        for (int i = 0; i < left_image_cnt; i++)
        {
            cv::Mat im = batch_imgs[i];
            int detect_num = 0;
            for (int j = 0; j < min(bbox_num[i], max_det_num); j++)
            {
                ObjectResult item = result[item_start_idx + j];
                if (item.class_id == -1)
                {
                    continue;
                }
                detect_num += 1;
                im_result->push_back(item);
            }
            item_start_idx = item_start_idx + bbox_num[i];
        }
        times_[0] += det_times[0];
        times_[1] += det_times[1];
        times_[2] += det_times[2];
    }
}

//void PipeLine::DetPredictImage2(const std::vector <cv::Mat> &batch_imgs,
//                                std::vector <ObjectResult> *im_result,
//                                const int batch_size_det, const int max_det_num,
//                                const bool run_benchmark, std::shared_ptr <ObjectDetector> det)
//{
//    int steps = ceil(batch_imgs.size() * 1.f / batch_size_det);
//    for (int idx = 0; idx < steps; idx++)
//    {
//        int left_image_cnt = batch_imgs.size() - idx * batch_size_det;
//        if (left_image_cnt > batch_size_det)
//        {
//            left_image_cnt = batch_size_det;
//        }
//        // Store all detected result
//        std::vector <ObjectResult> result;
//        std::vector<int> bbox_num;
//        std::vector<double> det_times;
//        bool is_rbox = false;
//        det->Predict2(batch_imgs, 0, 1, &result, &bbox_num, &det_times);
//        int item_start_idx = 0;
//        for (int i = 0; i < left_image_cnt; i++)
//        {
//            cv::Mat im = batch_imgs[i];
//            int detect_num = 0;
//            for (int j = 0; j < min(bbox_num[i], max_det_num); j++)
//            {
//                ObjectResult item = result[item_start_idx + j];
//                if (item.class_id == -1)
//                {
//                    continue;
//                }
//                detect_num += 1;
//                im_result->push_back(item);
//            }
//            item_start_idx = item_start_idx + bbox_num[i];
//        }
//        times_[0] += det_times[0];
//        times_[1] += det_times[1];
//        times_[2] += det_times[2];
//    }
//}

template<typename T>
static inline bool SortScorePairDescend(const std::pair<float, T> &pair1, const std::pair<float, T> &pair2)
{
    return pair1.first > pair2.first;
}

inline void GetMaxScoreIndex(const std::vector <ObjectResult> &det_result,
                             const float threshold,
                             std::vector <std::pair<float, int>> &score_index_vec)
{
    // Generate index score pairs.
    for (size_t i = 0; i < det_result.size(); ++i)
    {
        if (det_result[i].confidence > threshold)
        {
            score_index_vec.push_back(std::make_pair(det_result[i].confidence, i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
}

float RectOverlap(const ObjectResult &a, const ObjectResult &b)
{
    float Aa = (a.rect[2] - a.rect[0] + 1) * (a.rect[3] - a.rect[1] + 1);
    float Ab = (b.rect[2] - b.rect[0] + 1) * (b.rect[3] - b.rect[1] + 1);

    int iou_w = max(min(a.rect[2], b.rect[2]) - max(a.rect[0], b.rect[0]) + 1, 0);
    int iou_h = max(min(a.rect[3], b.rect[3]) - max(a.rect[1], b.rect[1]) + 1, 0);
    float Aab = iou_w * iou_h;
    return Aab / (Aa + Ab - Aab);
}

void PipeLine::NMSBoxes(const std::vector <ObjectResult> det_result,
                        const float score_threshold, const float nms_threshold,
                        std::vector<int> &indices)
{
    int a = 1;
    // Get top_k scores (with corresponding indices).
    std::vector <std::pair<float, int>> score_index_vec;
    GetMaxScoreIndex(det_result, score_threshold, score_index_vec);

    // Do nms
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i)
    {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int) indices.size() && keep; ++k)
        {
            const int kept_idx = indices[k];
            float overlap = RectOverlap(det_result[idx], det_result[kept_idx]);
            keep = overlap <= nms_threshold;
        }
        if (keep)
            indices.push_back(idx);
    }
}



