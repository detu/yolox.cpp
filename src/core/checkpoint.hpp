// Checkpoint I/O compatible with Python YOLOX:
//  * reads   torch.save({"model": state_dict, ...}) files (yolox_m.pth, best_ckpt.pth)
//  * writes  {"model": state_dict, "start_epoch", "best_ap", "curr_ap"} with
//            torch::pickle_save, which Python's torch.load() can read, so
//            tools/export_onnx.py / demo.py from the original repo keep working.
#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace yolox {

using StateDict = std::vector<std::pair<std::string, torch::Tensor>>;

// parameters + buffers with Python-style dotted names
StateDict state_dict(const torch::nn::Module& m);

struct LoadedCheckpoint {
  std::unordered_map<std::string, torch::Tensor> model;
  std::optional<int64_t> start_epoch;
  std::optional<double> best_ap;
};

LoadedCheckpoint read_checkpoint(const std::string& path);

// utils/checkpoint.py::load_ckpt — copy matching keys, warn on missing/mismatch.
// Returns number of tensors loaded.
size_t load_ckpt(torch::nn::Module& model,
                 const std::unordered_map<std::string, torch::Tensor>& ckpt, bool verbose = true);

void save_checkpoint(const std::string& path, const torch::nn::Module& model,
                     int64_t start_epoch, double best_ap, std::optional<double> curr_ap);

}  // namespace yolox
