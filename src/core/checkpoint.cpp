#include "core/checkpoint.hpp"

#include <torch/serialize.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace yolox {

StateDict state_dict(const torch::nn::Module& m) {
  StateDict sd;
  for (const auto& p : m.named_parameters(/*recurse=*/true)) sd.emplace_back(p.key(), p.value());
  for (const auto& b : m.named_buffers(/*recurse=*/true)) sd.emplace_back(b.key(), b.value());
  return sd;
}

LoadedCheckpoint read_checkpoint(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open checkpoint: " + path);
  std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  c10::IValue iv;
  try {
    iv = torch::pickle_load(bytes);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "failed to read '" + path + "' with torch::pickle_load (" + e.what() +
        ").\nConvert it once with:  python tools/convert_pth.py " + path + " <out.pth>");
  }
  if (!iv.isGenericDict()) throw std::runtime_error(path + ": checkpoint is not a dict");

  LoadedCheckpoint out;
  auto root = iv.toGenericDict();
  c10::impl::GenericDict sd = root;
  auto it = root.find(c10::IValue("model"));
  if (it != root.end() && it->value().isGenericDict()) {
    sd = it->value().toGenericDict();
    auto se = root.find(c10::IValue("start_epoch"));
    if (se != root.end() && se->value().isInt()) out.start_epoch = se->value().toInt();
    auto ba = root.find(c10::IValue("best_ap"));
    if (ba != root.end()) {
      if (ba->value().isDouble()) out.best_ap = ba->value().toDouble();
      else if (ba->value().isInt()) out.best_ap = static_cast<double>(ba->value().toInt());
    }
  }
  for (const auto& e : sd) {
    if (e.key().isString() && e.value().isTensor())
      out.model[e.key().toStringRef()] = e.value().toTensor();
  }
  if (out.model.empty()) throw std::runtime_error(path + ": no tensors found in checkpoint");
  return out;
}

size_t load_ckpt(torch::nn::Module& model,
                 const std::unordered_map<std::string, torch::Tensor>& ckpt, bool verbose) {
  torch::NoGradGuard ng;
  size_t loaded = 0;
  for (auto& kv : state_dict(model)) {
    const auto& key = kv.first;
    auto& v = kv.second;
    auto it = ckpt.find(key);
    if (it == ckpt.end()) {
      if (verbose)
        std::cerr << "[warn] " << key
                  << " is not in the ckpt. Please double check and see if this is desired.\n";
      continue;
    }
    if (it->second.sizes() != v.sizes()) {
      if (verbose)
        std::cerr << "[warn] Shape of " << key << " in checkpoint is " << it->second.sizes()
                  << ", while shape of " << key << " in model is " << v.sizes() << ".\n";
      continue;
    }
    v.copy_(it->second.to(v.device(), v.scalar_type()));
    ++loaded;
  }
  return loaded;
}

void save_checkpoint(const std::string& path, const torch::nn::Module& model,
                     int64_t start_epoch, double best_ap, std::optional<double> curr_ap) {
  c10::Dict<std::string, torch::Tensor> md;
  for (auto& kv : state_dict(model)) md.insert(kv.first, kv.second.detach().to(torch::kCPU).clone());

  c10::impl::GenericDict root(c10::StringType::get(), c10::AnyType::get());
  root.insert(c10::IValue(std::string("start_epoch")), c10::IValue(start_epoch));
  root.insert(c10::IValue(std::string("model")), c10::IValue(md));
  root.insert(c10::IValue(std::string("best_ap")), c10::IValue(best_ap));
  root.insert(c10::IValue(std::string("curr_ap")),
              curr_ap ? c10::IValue(*curr_ap) : c10::IValue());

  std::vector<char> bytes = torch::pickle_save(c10::IValue(root));
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write checkpoint: " + path);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace yolox
