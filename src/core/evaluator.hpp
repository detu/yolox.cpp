// C++ port of yolox/evaluators/coco_evaluator.py
#pragma once

#include <memory>
#include <string>

#include "data/coco_dataset.hpp"
#include "models/yolox.hpp"

namespace yolox {

struct EvalOutput {
  double ap50_95 = 0.0;
  double ap50 = 0.0;
  std::string summary;
};

class COCOEvaluator {
 public:
  COCOEvaluator(std::shared_ptr<COCODataset> dataset, int batch_size, int num_workers,
                cv::Size img_size_wh, double confthre, double nmsthre, int num_classes);

  // model is put in eval mode; caller restores training mode.
  EvalOutput evaluate(YOLOX& model, torch::Device device, bool half = false,
                      const std::string& dump_json = "");

 private:
  std::shared_ptr<COCODataset> dataset_;
  int batch_size_, num_workers_;
  cv::Size img_size_;
  double confthre_, nmsthre_;
  int num_classes_;
};

}  // namespace yolox
