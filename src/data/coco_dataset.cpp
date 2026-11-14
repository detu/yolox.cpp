#include "data/coco_dataset.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace yolox {

// ================================================================ COCODataset
COCODataset::COCODataset(const std::string& data_dir, const std::string& json_file,
                         const std::string& name, cv::Size img_size, bool cache_ram,
                         int cache_threads)
    : data_dir_(data_dir), name_(name), img_size_(img_size) {
  coco_ = std::make_shared<CocoApi>(data_dir + "/annotations/" + json_file);
  class_ids_ = coco_->sorted_cat_ids();

  for (const auto& im : coco_->images) {
    ids_.push_back(im.id);
    AnnoRecord rec;
    rec.orig_h = im.height;
    rec.orig_w = im.width;
    if (!im.file_name.empty()) {
      rec.file_name = im.file_name;
    } else {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%012lld.jpg", static_cast<long long>(im.id));
      rec.file_name = buf;
    }
    const double width = im.width, height = im.height;
    const double r = std::min(img_size_.height / height, img_size_.width / width);

    auto it = coco_->img_to_anns.find(im.id);
    if (it != coco_->img_to_anns.end()) {
      for (size_t ai : it->second) {
        const auto& obj = coco_->anns[ai];
        if (obj.iscrowd) continue;  // getAnnIds(iscrowd=False)
        const double x1 = std::max(0.0, obj.bbox[0]);
        const double y1 = std::max(0.0, obj.bbox[1]);
        const double x2 = std::min(width, x1 + std::max(0.0, obj.bbox[2]));
        const double y2 = std::min(height, y1 + std::max(0.0, obj.bbox[3]));
        if (obj.area > 0 && x2 >= x1 && y2 >= y1) {
          auto cit = std::find(class_ids_.begin(), class_ids_.end(), obj.category_id);
          if (cit == class_ids_.end())
            throw std::runtime_error("annotation with unknown category_id");
          const float cls = static_cast<float>(cit - class_ids_.begin());
          rec.labels.push_back({static_cast<float>(x1 * r), static_cast<float>(y1 * r),
                                static_cast<float>(x2 * r), static_cast<float>(y2 * r), cls});
        }
      }
    }
    annotations_.push_back(std::move(rec));
  }

  if (cache_ram) {
    std::cout << "Caching " << annotations_.size() << " images of " << name_
              << " into RAM..." << std::endl;
    cache_.resize(annotations_.size());
    const int nt = std::max(1, cache_threads);
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) {
      th.emplace_back([this, t, nt] {
        for (size_t i = t; i < annotations_.size(); i += nt) cache_[i] = load_resized_img(i);
      });
    }
    for (auto& x : th) x.join();
  }
}

std::vector<std::string> COCODataset::class_names() const {
  std::vector<std::string> names;
  for (auto id : class_ids_) {
    for (const auto& c : coco_->cats)
      if (c.id == id) names.push_back(c.name);
  }
  return names;
}

cv::Mat COCODataset::load_image(size_t index) const {
  const std::string path = data_dir_ + "/" + name_ + "/" + annotations_[index].file_name;
  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
  if (img.empty()) throw std::runtime_error("file named " + path + " not found");
  return img;
}

cv::Mat COCODataset::load_resized_img(size_t index) const {
  cv::Mat img = load_image(index);
  const double r = std::min(static_cast<double>(img_size_.height) / img.rows,
                            static_cast<double>(img_size_.width) / img.cols);
  cv::Mat resized;
  cv::resize(img, resized,
             cv::Size(std::max(1, static_cast<int>(img.cols * r)),
                      std::max(1, static_cast<int>(img.rows * r))),
             0, 0, cv::INTER_LINEAR);
  return resized;
}

cv::Mat COCODataset::read_img(size_t index) const {
  if (!cache_.empty()) return cache_[index].clone();
  return load_resized_img(index);
}

PulledItem COCODataset::pull_item(size_t index) const {
  PulledItem it;
  it.img = read_img(index);
  it.labels = annotations_[index].labels;  // deep copy
  it.orig_h = annotations_[index].orig_h;
  it.orig_w = annotations_[index].orig_w;
  it.img_id = ids_[index];
  return it;
}

// ================================================================ Mosaic
MosaicDetection::MosaicDetection(std::shared_ptr<COCODataset> dataset, cv::Size input_dim,
                                 MosaicConfig cfg, TrainTransform preproc)
    : dataset_(std::move(dataset)),
      input_dim_(input_dim),
      cfg_(cfg),
      preproc_(preproc),
      mosaic_enabled_(cfg.enable_mosaic) {
  for (size_t i = 0; i < dataset_->size(); ++i)
    if (!dataset_->load_anno(i).empty()) { has_labelled_images_ = true; break; }
}

namespace {
struct Coord { int x1, y1, x2, y2; };

std::pair<Coord, Coord> get_mosaic_coordinate(int idx, int xc, int yc, int w, int h,
                                              int input_h, int input_w) {
  Coord l{}, s{};
  if (idx == 0) {
    l = {std::max(xc - w, 0), std::max(yc - h, 0), xc, yc};
    s = {w - (l.x2 - l.x1), h - (l.y2 - l.y1), w, h};
  } else if (idx == 1) {
    l = {xc, std::max(yc - h, 0), std::min(xc + w, input_w * 2), yc};
    s = {0, h - (l.y2 - l.y1), std::min(w, l.x2 - l.x1), h};
  } else if (idx == 2) {
    l = {std::max(xc - w, 0), yc, xc, std::min(input_h * 2, yc + h)};
    s = {w - (l.x2 - l.x1), 0, w, std::min(l.y2 - l.y1, h)};
  } else {
    l = {xc, yc, std::min(xc + w, input_w * 2), std::min(input_h * 2, yc + h)};
    s = {0, 0, std::min(w, l.x2 - l.x1), std::min(l.y2 - l.y1, h)};
  }
  return {l, s};
}
}  // namespace

std::pair<cv::Mat, std::vector<float>> MosaicDetection::get_item(size_t idx, Rng& rng) const {
  if (mosaic_enabled_.load() && rand01(rng) < cfg_.mosaic_prob) {
    const int input_h = input_dim_.height, input_w = input_dim_.width;
    const int yc = static_cast<int>(uniform(rng, 0.5 * input_h, 1.5 * input_h));
    const int xc = static_cast<int>(uniform(rng, 0.5 * input_w, 1.5 * input_w));
    const int64_t n = static_cast<int64_t>(dataset_->size());
    size_t indices[4] = {idx, static_cast<size_t>(randint(rng, 0, n - 1)),
                         static_cast<size_t>(randint(rng, 0, n - 1)),
                         static_cast<size_t>(randint(rng, 0, n - 1))};

    cv::Mat mosaic_img(input_h * 2, input_w * 2, CV_8UC3, cv::Scalar(114, 114, 114));
    Labels mosaic_labels;
    for (int i_mosaic = 0; i_mosaic < 4; ++i_mosaic) {
      PulledItem item = dataset_->pull_item(indices[i_mosaic]);
      const int h0 = item.img.rows, w0 = item.img.cols;
      const double scale = std::min(1.0 * input_h / h0, 1.0 * input_w / w0);
      cv::Mat img;
      cv::resize(item.img, img,
                 cv::Size(std::max(1, static_cast<int>(w0 * scale)),
                          std::max(1, static_cast<int>(h0 * scale))),
                 0, 0, cv::INTER_LINEAR);
      const int h = img.rows, w = img.cols;
      auto cs = get_mosaic_coordinate(i_mosaic, xc, yc, w, h, input_h, input_w);
      const Coord& l = cs.first;
      const Coord& s = cs.second;
      const int cw = l.x2 - l.x1, ch = l.y2 - l.y1;
      if (cw > 0 && ch > 0 && s.x2 - s.x1 == cw && s.y2 - s.y1 == ch) {
        img(cv::Rect(s.x1, s.y1, cw, ch)).copyTo(mosaic_img(cv::Rect(l.x1, l.y1, cw, ch)));
      }
      const float padw = static_cast<float>(l.x1 - s.x1), padh = static_cast<float>(l.y1 - s.y1);
      for (auto lab : item.labels) {
        lab[0] = static_cast<float>(scale * lab[0] + padw);
        lab[1] = static_cast<float>(scale * lab[1] + padh);
        lab[2] = static_cast<float>(scale * lab[2] + padw);
        lab[3] = static_cast<float>(scale * lab[3] + padh);
        mosaic_labels.push_back(lab);
      }
    }
    for (auto& lab : mosaic_labels) {
      lab[0] = std::clamp(lab[0], 0.f, 2.f * input_w);
      lab[1] = std::clamp(lab[1], 0.f, 2.f * input_h);
      lab[2] = std::clamp(lab[2], 0.f, 2.f * input_w);
      lab[3] = std::clamp(lab[3], 0.f, 2.f * input_h);
    }

    random_affine(mosaic_img, mosaic_labels, cv::Size(input_w, input_h), rng, cfg_.degrees,
                  cfg_.translate, cfg_.mosaic_scale, cfg_.shear);

    if (cfg_.enable_mixup && !mosaic_labels.empty() && rand01(rng) < cfg_.mixup_prob) {
      mixup(mosaic_img, mosaic_labels, rng);
    }
    return preproc_(mosaic_img, mosaic_labels, input_dim_, rng);
  }

  PulledItem item = dataset_->pull_item(idx);
  return preproc_(item.img, item.labels, input_dim_, rng);
}

void MosaicDetection::mixup(cv::Mat& origin_img, Labels& origin_labels, Rng& rng) const {
  if (!has_labelled_images_) return;
  const double jit_factor = uniform(rng, cfg_.mixup_scale.first, cfg_.mixup_scale.second);
  const bool FLIP = uniform(rng, 0, 1) > 0.5;
  const int64_t n = static_cast<int64_t>(dataset_->size());
  size_t cp_index = 0;
  do {
    cp_index = static_cast<size_t>(randint(rng, 0, n - 1));
  } while (dataset_->load_anno(cp_index).empty());

  PulledItem item = dataset_->pull_item(cp_index);
  const cv::Mat& img = item.img;
  const int in_h = input_dim_.height, in_w = input_dim_.width;
  cv::Mat cp_img(in_h, in_w, CV_8UC3, cv::Scalar(114, 114, 114));
  double cp_scale_ratio = std::min(static_cast<double>(in_h) / img.rows,
                                   static_cast<double>(in_w) / img.cols);
  {
    const int rw = std::max(1, static_cast<int>(img.cols * cp_scale_ratio));
    const int rh = std::max(1, static_cast<int>(img.rows * cp_scale_ratio));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
    resized(cv::Rect(0, 0, std::min(rw, in_w), std::min(rh, in_h)))
        .copyTo(cp_img(cv::Rect(0, 0, std::min(rw, in_w), std::min(rh, in_h))));
  }
  {
    cv::Mat j;
    cv::resize(cp_img, j,
               cv::Size(std::max(1, static_cast<int>(cp_img.cols * jit_factor)),
                        std::max(1, static_cast<int>(cp_img.rows * jit_factor))),
               0, 0, cv::INTER_LINEAR);
    cp_img = j;
  }
  cp_scale_ratio *= jit_factor;
  if (FLIP) cv::flip(cp_img, cp_img, 1);

  const int origin_h = cp_img.rows, origin_w = cp_img.cols;
  const int target_h = origin_img.rows, target_w = origin_img.cols;
  cv::Mat padded_img = cv::Mat::zeros(std::max(origin_h, target_h), std::max(origin_w, target_w),
                                      CV_8UC3);
  cp_img.copyTo(padded_img(cv::Rect(0, 0, origin_w, origin_h)));

  int x_offset = 0, y_offset = 0;
  if (padded_img.rows > target_h)
    y_offset = static_cast<int>(randint(rng, 0, padded_img.rows - target_h - 1));
  if (padded_img.cols > target_w)
    x_offset = static_cast<int>(randint(rng, 0, padded_img.cols - target_w - 1));
  cv::Mat padded_cropped = padded_img(cv::Rect(x_offset, y_offset, target_w, target_h));

  for (const auto& lab : item.labels) {
    // adjust_box_anns(..., cp_scale_ratio, 0, 0, origin_w, origin_h)
    double x1 = std::clamp(lab[0] * cp_scale_ratio, 0.0, static_cast<double>(origin_w));
    double y1 = std::clamp(lab[1] * cp_scale_ratio, 0.0, static_cast<double>(origin_h));
    double x2 = std::clamp(lab[2] * cp_scale_ratio, 0.0, static_cast<double>(origin_w));
    double y2 = std::clamp(lab[3] * cp_scale_ratio, 0.0, static_cast<double>(origin_h));
    if (FLIP) {
      const double nx1 = origin_w - x2, nx2 = origin_w - x1;
      x1 = nx1;
      x2 = nx2;
    }
    x1 = std::clamp(x1 - x_offset, 0.0, static_cast<double>(target_w));
    x2 = std::clamp(x2 - x_offset, 0.0, static_cast<double>(target_w));
    y1 = std::clamp(y1 - y_offset, 0.0, static_cast<double>(target_h));
    y2 = std::clamp(y2 - y_offset, 0.0, static_cast<double>(target_h));
    origin_labels.push_back({static_cast<float>(x1), static_cast<float>(y1),
                             static_cast<float>(x2), static_cast<float>(y2), lab[4]});
  }

  // 0.5 * a + 0.5 * b, then astype(uint8) (truncation) == (a + b) >> 1
  cv::Mat blended(target_h, target_w, CV_8UC3);
  for (int y = 0; y < target_h; ++y) {
    const uchar* a = origin_img.ptr<uchar>(y);
    const uchar* b = padded_cropped.ptr<uchar>(y);
    uchar* o = blended.ptr<uchar>(y);
    for (int x = 0; x < target_w * 3; ++x) o[x] = static_cast<uchar>((a[x] + b[x]) >> 1);
  }
  origin_img = blended;
}

}  // namespace yolox
