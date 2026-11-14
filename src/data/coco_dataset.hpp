// C++ port of yolox/data/datasets/coco.py and mosaicdetection.py
#pragma once

#include <opencv2/core.hpp>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "data/coco_api.hpp"
#include "data/data_augment.hpp"

namespace yolox {

struct PulledItem {
  cv::Mat img;          // resized so that it fits img_size (aspect kept)
  Labels labels;        // x1, y1, x2, y2, cls (in resized-image pixels)
  int orig_h = 0, orig_w = 0;
  int64_t img_id = 0;
};

class COCODataset {
 public:
  // data_dir/annotations/<json_file>, images in data_dir/<name>/
  COCODataset(const std::string& data_dir, const std::string& json_file,
              const std::string& name, cv::Size img_size_wh, bool cache_ram = false,
              int cache_threads = 8);

  size_t size() const { return annotations_.size(); }
  const Labels& load_anno(size_t index) const { return annotations_[index].labels; }
  cv::Mat load_image(size_t index) const;          // original BGR image
  cv::Mat load_resized_img(size_t index) const;    // resized to fit img_size
  cv::Mat read_img(size_t index) const;            // cached or loaded, always a private copy
  PulledItem pull_item(size_t index) const;

  const CocoApi& coco() const { return *coco_; }
  const std::vector<int64_t>& class_ids() const { return class_ids_; }
  std::vector<std::string> class_names() const;
  cv::Size img_size() const { return img_size_; }
  int64_t image_id(size_t index) const { return ids_[index]; }
  std::pair<int, int> orig_hw(size_t index) const {
    return {annotations_[index].orig_h, annotations_[index].orig_w};
  }

 private:
  struct AnnoRecord {
    Labels labels;
    int orig_h = 0, orig_w = 0;
    std::string file_name;
  };
  std::string data_dir_, name_;
  cv::Size img_size_;
  std::shared_ptr<CocoApi> coco_;
  std::vector<int64_t> ids_;
  std::vector<int64_t> class_ids_;
  std::vector<AnnoRecord> annotations_;
  std::vector<cv::Mat> cache_;
};

struct MosaicConfig {
  bool enable_mosaic = true;
  double degrees = 10.0;
  double translate = 0.1;
  std::pair<double, double> mosaic_scale{0.1, 2.0};
  std::pair<double, double> mixup_scale{0.5, 1.5};
  double shear = 2.0;
  bool enable_mixup = true;
  double mosaic_prob = 1.0;
  double mixup_prob = 1.0;
};

class MosaicDetection {
 public:
  MosaicDetection(std::shared_ptr<COCODataset> dataset, cv::Size input_dim_wh,
                  MosaicConfig cfg, TrainTransform preproc);

  size_t size() const { return dataset_->size(); }
  // returns preprocessed image (input_dim) and padded labels [max_labels x 5]
  std::pair<cv::Mat, std::vector<float>> get_item(size_t idx, Rng& rng) const;

  void close_mosaic() { mosaic_enabled_.store(false); }
  bool mosaic_enabled() const { return mosaic_enabled_.load(); }
  int max_labels() const { return preproc_.max_labels; }
  cv::Size input_dim() const { return input_dim_; }

 private:
  void mixup(cv::Mat& origin_img, Labels& origin_labels, Rng& rng) const;

  std::shared_ptr<COCODataset> dataset_;
  cv::Size input_dim_;
  MosaicConfig cfg_;
  TrainTransform preproc_;
  std::atomic<bool> mosaic_enabled_;
  bool has_labelled_images_ = false;
};

}  // namespace yolox
