#include "data/data_augment.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace yolox {

void augment_hsv(cv::Mat& img, Rng& rng, double hgain, double sgain, double vgain) {
  const double gains[3] = {hgain, sgain, vgain};
  double r[3];
  for (int i = 0; i < 3; ++i) r[i] = uniform(rng, -1.0, 1.0) * gains[i];
  int d[3];
  for (int i = 0; i < 3; ++i) {
    const int sel = static_cast<int>(randint(rng, 0, 1));
    d[i] = static_cast<int>(static_cast<int16_t>(r[i] * sel));  // astype(int16) truncates
  }

  cv::Mat hsv;
  cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
  cv::Mat lut_h(1, 256, CV_8U), lut_s(1, 256, CV_8U), lut_v(1, 256, CV_8U);
  for (int i = 0; i < 256; ++i) {
    lut_h.at<uchar>(i) = static_cast<uchar>((((i + d[0]) % 180) + 180) % 180);
    lut_s.at<uchar>(i) = static_cast<uchar>(std::clamp(i + d[1], 0, 255));
    lut_v.at<uchar>(i) = static_cast<uchar>(std::clamp(i + d[2], 0, 255));
  }
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);
  cv::LUT(ch[0], lut_h, ch[0]);
  cv::LUT(ch[1], lut_s, ch[1]);
  cv::LUT(ch[2], lut_v, ch[2]);
  cv::merge(ch, hsv);
  cv::cvtColor(hsv, img, cv::COLOR_HSV2BGR);
}

namespace {

// get_affine_matrix() from data_augment.py
cv::Mat get_affine_matrix(cv::Size target_size, Rng& rng, double degrees, double translate,
                          std::pair<double, double> scales, double shear) {
  const double angle = uniform(rng, -degrees, degrees);
  const double scale = uniform(rng, scales.first, scales.second);
  if (scale <= 0.0) throw std::invalid_argument("Argument scale should be positive");

  // cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
  const double rad = angle * CV_PI / 180.0;
  const double a = scale * std::cos(rad), b = scale * std::sin(rad);
  const double R[2][3] = {{a, b, 0.0}, {-b, a, 0.0}};

  const double shear_x = std::tan(uniform(rng, -shear, shear) * CV_PI / 180.0);
  const double shear_y = std::tan(uniform(rng, -shear, shear) * CV_PI / 180.0);

  cv::Mat M(2, 3, CV_64F);
  for (int j = 0; j < 3; ++j) {
    M.at<double>(0, j) = R[0][j] + shear_y * R[1][j];
    M.at<double>(1, j) = R[1][j] + shear_x * R[0][j];
  }
  M.at<double>(0, 2) = uniform(rng, -translate, translate) * target_size.width;
  M.at<double>(1, 2) = uniform(rng, -translate, translate) * target_size.height;
  return M;
}

void apply_affine_to_bboxes(Labels& targets, cv::Size target_size, const cv::Mat& M) {
  const double* m0 = M.ptr<double>(0);
  const double* m1 = M.ptr<double>(1);
  for (auto& t : targets) {
    const double xs[4] = {t[0], t[2], t[0], t[2]};  // x1y1, x2y2, x1y2, x2y1
    const double ys[4] = {t[1], t[3], t[3], t[1]};
    double minx = 1e30, miny = 1e30, maxx = -1e30, maxy = -1e30;
    for (int k = 0; k < 4; ++k) {
      const double x = m0[0] * xs[k] + m0[1] * ys[k] + m0[2];
      const double y = m1[0] * xs[k] + m1[1] * ys[k] + m1[2];
      minx = std::min(minx, x); maxx = std::max(maxx, x);
      miny = std::min(miny, y); maxy = std::max(maxy, y);
    }
    t[0] = static_cast<float>(std::clamp(minx, 0.0, static_cast<double>(target_size.width)));
    t[1] = static_cast<float>(std::clamp(miny, 0.0, static_cast<double>(target_size.height)));
    t[2] = static_cast<float>(std::clamp(maxx, 0.0, static_cast<double>(target_size.width)));
    t[3] = static_cast<float>(std::clamp(maxy, 0.0, static_cast<double>(target_size.height)));
  }
}

}  // namespace

void random_affine(cv::Mat& img, Labels& targets, cv::Size target_size, Rng& rng,
                   double degrees, double translate, std::pair<double, double> scales,
                   double shear) {
  cv::Mat M = get_affine_matrix(target_size, rng, degrees, translate, scales, shear);
  cv::Mat out;
  cv::warpAffine(img, out, M, target_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                 cv::Scalar(114, 114, 114));
  img = out;
  if (!targets.empty()) apply_affine_to_bboxes(targets, target_size, M);
}

std::pair<cv::Mat, double> preproc(const cv::Mat& img, cv::Size input_size) {
  cv::Mat padded(input_size, CV_8UC3, cv::Scalar(114, 114, 114));
  const double r = std::min(static_cast<double>(input_size.height) / img.rows,
                            static_cast<double>(input_size.width) / img.cols);
  const int nw = std::max(1, static_cast<int>(img.cols * r));
  const int nh = std::max(1, static_cast<int>(img.rows * r));
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
  resized.copyTo(padded(cv::Rect(0, 0, std::min(nw, input_size.width),
                                 std::min(nh, input_size.height))));
  return {padded, r};
}

std::pair<cv::Mat, std::vector<float>> TrainTransform::operator()(cv::Mat image,
                                                                   const Labels& targets,
                                                                   cv::Size input_dim,
                                                                   Rng& rng) const {
  std::vector<float> padded(static_cast<size_t>(max_labels) * 5, 0.f);
  if (targets.empty()) {
    return {preproc(image, input_dim).first, padded};
  }

  cv::Mat image_o = image.clone();
  // xyxy -> cxcywh copies of the originals
  auto to_cxcywh = [](const Label& t) {
    const float w = t[2] - t[0], h = t[3] - t[1];
    return std::array<float, 4>{t[0] + w * 0.5f, t[1] + h * 0.5f, w, h};
  };

  if (rand01(rng) < hsv_prob) augment_hsv(image, rng);

  Labels boxes = targets;
  if (rand01(rng) < flip_prob) {  // _mirror
    const float width = static_cast<float>(image.cols);
    cv::flip(image, image, 1);
    for (auto& b : boxes) {
      const float x1 = b[0], x2 = b[2];
      b[0] = width - x2;
      b[2] = width - x1;
    }
  }
  auto pr = preproc(image, input_dim);
  const float r = static_cast<float>(pr.second);

  std::vector<std::array<float, 5>> out;  // cls, cx, cy, w, h
  for (const auto& b : boxes) {
    auto c = to_cxcywh(b);
    for (auto& v : c) v *= r;
    if (std::min(c[2], c[3]) > 1.f) out.push_back({b[4], c[0], c[1], c[2], c[3]});
  }
  cv::Mat image_t = pr.first;
  if (out.empty()) {
    auto pr_o = preproc(image_o, input_dim);
    image_t = pr_o.first;
    const float ro = static_cast<float>(pr_o.second);
    for (const auto& b : targets) {
      auto c = to_cxcywh(b);
      out.push_back({b[4], c[0] * ro, c[1] * ro, c[2] * ro, c[3] * ro});
    }
  }
  const size_t n = std::min(out.size(), static_cast<size_t>(max_labels));
  for (size_t i = 0; i < n; ++i)
    for (int j = 0; j < 5; ++j) padded[i * 5 + j] = out[i][j];
  return {image_t, padded};
}

}  // namespace yolox
