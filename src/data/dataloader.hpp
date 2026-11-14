// Multi-threaded replacement for torch DataLoader + InfiniteSampler +
// YoloBatchSampler + DataPrefetcher used by YOLOX training.
#pragma once

#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "data/coco_dataset.hpp"

namespace yolox {

struct Batch {
  torch::Tensor images;   // [B, 3, H, W] float32, BGR 0..255
  torch::Tensor targets;  // [B, max_labels, 5] float32 (cls, cx, cy, w, h)
};

class TrainLoader {
 public:
  TrainLoader(std::shared_ptr<MosaicDetection> dataset, int batch_size, int num_workers,
              uint64_t seed, int prefetch_batches = 4);
  ~TrainLoader();
  TrainLoader(const TrainLoader&) = delete;
  TrainLoader& operator=(const TrainLoader&) = delete;

  Batch next();
  // len(train_loader): ceil(N / batch_size) (drop_last=False)
  int64_t iters_per_epoch() const;
  void close_mosaic() { dataset_->close_mosaic(); }

 private:
  void worker(int wid);
  std::vector<size_t> next_indices();

  std::shared_ptr<MosaicDetection> dataset_;
  int batch_size_;
  int prefetch_;
  bool pin_memory_;

  // InfiniteSampler: an endless stream of shuffled permutations
  std::mutex sampler_mu_;
  std::vector<size_t> perm_;
  size_t perm_pos_ = 0;
  Rng sampler_rng_;

  std::mutex q_mu_;
  std::condition_variable q_not_empty_, q_not_full_;
  std::deque<Batch> queue_;
  std::exception_ptr error_;
  std::atomic<bool> stop_{false};
  std::vector<std::thread> threads_;
  uint64_t seed_;
};

// Converts a list of HxWx3 uint8 images into a [B,3,H,W] float tensor
torch::Tensor images_to_tensor(const std::vector<cv::Mat>& imgs, bool pin_memory);

}  // namespace yolox
