// C++ equivalent of tools/train.py
//
//   yolox_train -f exps/yolox_m.cfg -d 1 -b 9 --fp16 -o -c yolox_m.pth
//
#include <iostream>

#include "cli.hpp"
#include "core/trainer.hpp"

static void usage() {
  std::cout <<
      R"(YOLOX train (LibTorch)
usage: yolox_train [-f EXP_FILE | -n MODEL] [options] [key value ...]
  -f, --exp_file PATH     experiment file (key = value lines, see exps/)
  -n, --name NAME         model preset: yolox-nano|yolox-tiny|yolox-s|yolox-m|yolox-l|yolox-x
  -expn, --experiment-name NAME
  -b, --batch-size N      batch size (default 64)
  -d, --devices N         number of GPUs (only 1 is supported by this port)
  --device IDX            CUDA device index (default 0, -1 = CPU)
  -c, --ckpt PATH         checkpoint to fine-tune from / resume
  --resume                resume training
  -e, --start_epoch N     resume start epoch
  --fp16                  mixed precision training
  --cache                 cache resized images in RAM
  -o, --occupy            accepted for compatibility (no-op)
  -l, --logger NAME       accepted for compatibility (log goes to train_log.txt)
  key value ...           override any exp option, e.g. max_epoch 50
)";
}

int main(int argc, char** argv) {
  try {
    auto p = yolox::cli::parse(argc, argv, {"--resume", "--fp16", "--cache", "-o", "--occupy", "-h", "--help"});
    if (yolox::cli::has(p, {"-h", "--help"}) || argc == 1) {
      usage();
      return 0;
    }
    yolox::Exp exp = yolox::cli::build_exp(p);

    yolox::TrainArgs args;
    args.experiment_name = yolox::cli::get(p, {"-expn", "--experiment-name"});
    args.batch_size = std::stoi(yolox::cli::get(p, {"-b", "--batch-size"}, "64"));
    args.ckpt = yolox::cli::get(p, {"-c", "--ckpt"});
    args.resume = yolox::cli::has(p, {"--resume"});
    const auto se = yolox::cli::get(p, {"-e", "--start_epoch"});
    if (!se.empty()) args.start_epoch = std::stoi(se);
    args.fp16 = yolox::cli::has(p, {"--fp16"});
    args.cache = yolox::cli::has(p, {"--cache"});
    args.device = std::stoi(yolox::cli::get(p, {"--device"}, "0"));
    const int ndev = std::stoi(yolox::cli::get(p, {"-d", "--devices"}, "1"));
    if (ndev > 1)
      std::cerr << "[warn] multi-GPU (DDP) training is not ported; using a single GPU.\n";

    yolox::Trainer trainer(exp, args);
    trainer.train();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
