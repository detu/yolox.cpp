// C++ equivalent of tools/eval.py (single GPU, COCO bbox mAP)
//
//   yolox_eval -f exps/yolox_m.cfg -c YOLOX_outputs/yolox_m/best_ckpt.pth -b 8
//
#include <iostream>

#include "cli.hpp"
#include "core/checkpoint.hpp"
#include "core/evaluator.hpp"

int main(int argc, char** argv) {
  try {
    auto p = yolox::cli::parse(argc, argv, {"--fp16", "-h", "--help"});
    if (yolox::cli::has(p, {"-h", "--help"}) || argc == 1) {
      std::cout << "usage: yolox_eval [-f EXP | -n MODEL] -c CKPT [-b N] [--conf C] [--nms T] "
                   "[--tsize S] [--fp16] [--device IDX] [--dump results.json] [key value ...]\n";
      return 0;
    }
    yolox::Exp exp = yolox::cli::build_exp(p);
    const std::string ckpt = yolox::cli::get(p, {"-c", "--ckpt"});
    if (ckpt.empty()) throw std::invalid_argument("-c CKPT is required");
    const int bs = std::stoi(yolox::cli::get(p, {"-b", "--batch-size"}, "64"));
    const auto conf = yolox::cli::get(p, {"--conf"});
    const auto nms = yolox::cli::get(p, {"--nms"});
    const auto tsize = yolox::cli::get(p, {"--tsize"});
    if (!conf.empty()) exp.test_conf = std::stod(conf);
    if (!nms.empty()) exp.nmsthre = std::stod(nms);
    if (!tsize.empty()) exp.test_size = {std::stoi(tsize), std::stoi(tsize)};
    const bool half = yolox::cli::has(p, {"--fp16"});
    const int dev = std::stoi(yolox::cli::get(p, {"--device"}, "0"));
    torch::Device device = (dev >= 0 && torch::cuda::is_available())
                               ? torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(dev))
                               : torch::Device(torch::kCPU);

    auto model = yolox::build_model(exp.model_config());
    auto ck = yolox::read_checkpoint(ckpt);
    yolox::load_ckpt(*model, ck.model);
    model->to(device);
    if (half) model->to(torch::kHalf);
    model->eval();

    const cv::Size test_wh(exp.test_size.second, exp.test_size.first);
    auto ds = std::make_shared<yolox::COCODataset>(exp.data_dir, exp.val_ann, exp.val_name, test_wh);
    yolox::COCOEvaluator ev(ds, bs, exp.data_num_workers, test_wh, exp.test_conf, exp.nmsthre,
                            exp.num_classes);
    auto res = ev.evaluate(model, device, half, yolox::cli::get(p, {"--dump"}));
    std::cout << res.summary << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
