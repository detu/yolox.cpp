#include "data/dataloader.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>

namespace yolox {

torch::Tensor images_to_tensor(const std::vector<cv::Mat>& imgs, bool pin_memory) {
  const int64_t B = static_cast<int64_t>(imgs.size());
  const int64_t H = imgs[0].rows, W = imgs[0].cols;
  auto out = torch::empty({B, 3, H, W},
                          torch::TensorOptions().dtype(torch::kFloat).pinned_memory(pin_memory));
  for (int64_t b = 0; b < B; ++b) {
    cv::Mat m = imgs[b].isContinuous() ? imgs[b] : imgs[b].clone();
    TORCH_CHECK(m.rows == H && m.cols == W && m.type() == CV_8UC3, "batch images differ in shape");
    auto hwc = torch::from_blob(m.data, {H, W, 3}, torch::kUInt8);
    out[b].copy_(hwc.permute({2, 0, 1}));  // HWC uint8 -> CHW float (BGR kept, no normalisation)
  }
  return out;
}

TrainLoader::TrainLoader(std::shared_ptr<MosaicDetection> dataset, int batch_size,
                         int num_workers, uint64_t seed, int prefetch_batches)
    : dataset_(std::move(dataset)),
      batch_size_(batch_size),
      prefetch_(std::max(1, prefetch_batches)),
      pin_memory_(torch::cuda::is_available()),
      sampler_rng_(seed),
      seed_(seed) {
  perm_.resize(dataset_->size());
  std::iota(perm_.begin(), perm_.end(), 0);
  std::shuffle(perm_.begin(), perm_.end(), sampler_rng_);
  const int nw = std::max(1, num_workers);
  for (int w = 0; w < nw; ++w) threads_.emplace_back(&TrainLoader::worker, this, w);
}

TrainLoader::~TrainLoader() {
  stop_ = true;
  q_not_full_.notify_all();
  q_not_empty_.notify_all();
  for (auto& t : threads_)
    if (t.joinable()) t.join();
}

int64_t TrainLoader::iters_per_epoch() const {
  return (static_cast<int64_t>(dataset_->size()) + batch_size_ - 1) / batch_size_;
}

std::vector<size_t> TrainLoader::next_indices() {
  std::lock_guard<std::mutex> lk(sampler_mu_);
  std::vector<size_t> idx;
  idx.reserve(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    if (perm_pos_ >= perm_.size()) {
      std::shuffle(perm_.begin(), perm_.end(), sampler_rng_);
      perm_pos_ = 0;
    }
    idx.push_back(perm_[perm_pos_++]);
  }
  return idx;
}

void TrainLoader::worker(int wid) {
  // worker_init_reset_seed: every worker gets its own random stream
  Rng rng(seed_ * 1000003ULL + static_cast<uint64_t>(wid) * 7919ULL + std::random_device{}());
  try {
    while (!stop_) {
      auto indices = next_indices();
      std::vector<cv::Mat> imgs;
      const int L = dataset_->max_labels();
      auto targets = torch::zeros({static_cast<int64_t>(indices.size()), L, 5},
                                  torch::TensorOptions().dtype(torch::kFloat).pinned_memory(pin_memory_));
      for (size_t i = 0; i < indices.size(); ++i) {
        auto sample = dataset_->get_item(indices[i], rng);
        imgs.push_back(sample.first);
        std::memcpy(targets[static_cast<int64_t>(i)].data_ptr<float>(), sample.second.data(),
                    sizeof(float) * sample.second.size());
      }
      Batch b{images_to_tensor(imgs, pin_memory_), targets};

      std::unique_lock<std::mutex> lk(q_mu_);
      q_not_full_.wait(lk, [&] { return stop_ || static_cast<int>(queue_.size()) < prefetch_; });
      if (stop_) break;
      queue_.push_back(std::move(b));
      q_not_empty_.notify_one();
    }
  } catch (...) {
    std::lock_guard<std::mutex> lk(q_mu_);
    error_ = std::current_exception();
    q_not_empty_.notify_all();
  }
}

Batch TrainLoader::next() {
  std::unique_lock<std::mutex> lk(q_mu_);
  q_not_empty_.wait(lk, [&] { return !queue_.empty() || error_ != nullptr; });
  if (queue_.empty() && error_) std::rethrow_exception(error_);
  Batch b = std::move(queue_.front());
  queue_.pop_front();
  q_not_full_.notify_one();
  return b;
}

}  // namespace yolox
