// C++/OpenCV port of yolox/data/data_augment.py
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace yolox {

// One label row as used by the dataset/mosaic stage: x1, y1, x2, y2, class
using Label = std::array<float, 5>;
using Labels = std::vector<Label>;

using Rng = std::mt19937_64;

// Python random.uniform(a, b)
inline double uniform(Rng& rng, double a, double b) {
  return a + (b - a) * std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}
// Python random.randint(a, b) (inclusive)
inline int64_t randint(Rng& rng, int64_t a, int64_t b) {
  return std::uniform_int_distribution<int64_t>(a, b)(rng);
}
inline double rand01(Rng& rng) { return uniform(rng, 0.0, 1.0); }

void augment_hsv(cv::Mat& img, Rng& rng, double hgain = 5, double sgain = 30,
                 double vgain = 30);

// Mosaic image -> target_size with random rotation/scale/shear/translation
void random_affine(cv::Mat& img, Labels& targets, cv::Size target_size, Rng& rng,
                   double degrees = 10.0, double translate = 0.1,
                   std::pair<double, double> scales = {0.1, 2.0}, double shear = 10.0);

// Letterbox-resize to input_size (h, w) with 114 padding at bottom/right.
// Returns padded HxWx3 uint8 image and ratio r.
std::pair<cv::Mat, double> preproc(const cv::Mat& img, cv::Size input_size_wh);

struct TrainTransform {
  int max_labels = 50;
  double flip_prob = 0.5;
  double hsv_prob = 1.0;

  // img: BGR uint8 (modified in place), targets: x1y1x2y2,cls.
  // Outputs padded image (input_dim) and a max_labels x 5 array of
  // (cls, cx, cy, w, h), zero padded.
  std::pair<cv::Mat, std::vector<float>> operator()(cv::Mat img, const Labels& targets,
                                                    cv::Size input_dim_wh, Rng& rng) const;
};

}  // namespace yolox
