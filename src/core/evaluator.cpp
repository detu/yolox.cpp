#include "core/evaluator.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <thread>

#include "data/dataloader.hpp"
#include "utils/boxes.hpp"

namespace yolox {

COCOEvaluator::COCOEvaluator(std::shared_ptr<COCODataset> dataset, int batch_size,
                             int num_workers, cv::Size img_size, double confthre, double nmsthre,
                             int num_classes)
    : dataset_(std::move(dataset)),
      batch_size_(std::max(1, batch_size)),
      num_workers_(std::max(1, num_workers)),
      img_size_(img_size),
      confthre_(confthre),
      nmsthre_(nmsthre),
      num_classes_(num_classes) {}

namespace {
// per_class_AP_table / per_class_AR_table (tabulate "pipe" format, 3 pairs per row)
std::string per_class_table(const std::vector<std::string>& names, const std::vector<double>& vals,
                            const std::string& metric) {
  const size_t pairs_per_row = 3;
  std::ostringstream os;
  char buf[128];
  for (size_t c = 0; c < pairs_per_row && c < names.size(); ++c) {
    std::snprintf(buf, sizeof(buf), "| %-14s | %-7s ", "class", metric.c_str());
    os << buf;
  }
  os << "|\n";
  for (size_t c = 0; c < pairs_per_row && c < names.size(); ++c) os << "|:---------------|:--------";
  os << "|\n";
  for (size_t i = 0; i < names.size(); i += pairs_per_row) {
    for (size_t c = 0; c < pairs_per_row; ++c) {
      if (i + c < names.size()) {
        const double v = vals[i + c];
        if (v < 0) std::snprintf(buf, sizeof(buf), "| %-14s | %-7s ", names[i + c].c_str(), "nan");
        else std::snprintf(buf, sizeof(buf), "| %-14s | %-7.3f ", names[i + c].c_str(), v * 100);
      } else if (names.size() > pairs_per_row) {
        std::snprintf(buf, sizeof(buf), "| %-14s | %-7s ", "", "");
      } else {
        buf[0] = '\0';
      }
      os << buf;
    }
    os << "|\n";
  }
  return os.str();
}
}  // namespace

EvalOutput COCOEvaluator::evaluate(YOLOX& model, torch::Device device, bool half,
                                   const std::string& dump_json) {
  torch::NoGradGuard ng;
  model->eval();
  const size_t N = dataset_->size();
  const auto& class_ids = dataset_->class_ids();

  std::vector<CocoDet> data_list;
  double inference_time = 0, nms_time = 0;
  size_t n_batches = 0;
  const size_t total_batches = (N + batch_size_ - 1) / batch_size_;

  for (size_t start = 0; start < N; start += batch_size_) {
    const size_t end = std::min(N, start + batch_size_);
    std::vector<cv::Mat> imgs(end - start);
    // ValTransform: preproc(read_img(i), test_size), loaded in parallel
    std::vector<std::thread> th;
    const size_t nt = std::min<size_t>(num_workers_, imgs.size());
    for (size_t t = 0; t < nt; ++t) {
      th.emplace_back([&, t] {
        for (size_t i = t; i < imgs.size(); i += nt)
          imgs[i] = preproc(dataset_->read_img(start + i), img_size_).first;
      });
    }
    for (auto& x : th) x.join();

    auto input = images_to_tensor(imgs, false).to(device);
    if (half) input = input.to(torch::kHalf);

    const bool is_time_record = (start / batch_size_) < total_batches - 1;
    auto t0 = std::chrono::steady_clock::now();
    auto outputs = model->forward(input);
    if (device.is_cuda()) torch::cuda::synchronize();
    auto t1 = std::chrono::steady_clock::now();
    auto dets = postprocess(outputs, num_classes_, confthre_, nmsthre_);
    auto t2 = std::chrono::steady_clock::now();
    if (is_time_record) {
      inference_time += std::chrono::duration<double>(t1 - t0).count();
      nms_time += std::chrono::duration<double>(t2 - t1).count();
      ++n_batches;
    }

    // convert_to_coco_format
    for (size_t i = 0; i < dets.size(); ++i) {
      if (!dets[i].defined()) continue;
      auto out = dets[i].to(torch::kCPU, torch::kFloat).contiguous();
      const auto hw = dataset_->orig_hw(start + i);
      const double scale = std::min(static_cast<double>(img_size_.height) / hw.first,
                                    static_cast<double>(img_size_.width) / hw.second);
      auto a = out.accessor<float, 2>();
      for (int64_t k = 0; k < out.size(0); ++k) {
        CocoDet d;
        d.image_id = dataset_->image_id(start + i);
        d.category_id = class_ids.at(static_cast<size_t>(a[k][6]));
        const double x1 = a[k][0] / scale, y1 = a[k][1] / scale;
        const double x2 = a[k][2] / scale, y2 = a[k][3] / scale;
        d.bbox = {x1, y1, x2 - x1, y2 - y1};
        d.score = static_cast<double>(a[k][4]) * static_cast<double>(a[k][5]);
        data_list.push_back(d);
      }
    }
    std::cout << "\rEvaluating " << end << "/" << N << std::flush;
  }
  std::cout << std::endl;

  // evaluate_prediction
  const double n_samples = static_cast<double>(std::max<size_t>(n_batches, 1));
  const double a_infer = 1000 * inference_time / (n_samples * batch_size_);
  const double a_nms = 1000 * nms_time / (n_samples * batch_size_);
  char tbuf[256];
  std::snprintf(tbuf, sizeof(tbuf),
                "Average forward time: %.2f ms, Average NMS time: %.2f ms, Average inference "
                "time: %.2f ms\n",
                a_infer, a_nms, a_infer + a_nms);

  EvalOutput res;
  res.summary = tbuf;
  if (!dump_json.empty()) write_coco_results(dump_json, data_list);
  if (data_list.empty()) return res;

  auto ce = coco_evaluate_bbox(dataset_->coco(), data_list);
  res.summary += ce.summary;
  const auto names = dataset_->class_names();
  res.summary += "per class AP:\n" + per_class_table(names, ce.per_class_ap, "AP");
  res.summary += "per class AR:\n" + per_class_table(names, ce.per_class_ar, "AR");
  res.ap50_95 = ce.stats[0];
  res.ap50 = ce.stats[1];
  return res;
}

}  // namespace yolox
