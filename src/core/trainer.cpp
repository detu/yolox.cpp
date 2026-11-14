#include "core/trainer.hpp"

#include <ATen/Context.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "core/checkpoint.hpp"

namespace fs = std::filesystem;

namespace yolox {

namespace {
double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}
std::string timestamp() {
  std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
  return buf;
}
bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}
std::string fmt_eta(double seconds) {
  auto s = static_cast<long long>(seconds);
  std::ostringstream os;
  const long long d = s / 86400;
  s %= 86400;
  if (d > 0) os << d << " day" << (d > 1 ? "s" : "") << ", ";
  os << s / 3600 << ":" << std::setw(2) << std::setfill('0') << (s % 3600) / 60 << ":"
     << std::setw(2) << std::setfill('0') << s % 60;
  return os.str();
}
}  // namespace

Trainer::Trainer(Exp exp, TrainArgs args)
    : exp_(std::move(exp)), args_(std::move(args)), rng_(std::random_device{}()) {
  max_epoch_ = exp_.max_epoch;
  input_size_ = exp_.input_size;
  if (args_.experiment_name.empty()) args_.experiment_name = exp_.exp_name;
  file_name_ = (fs::path(exp_.output_dir) / args_.experiment_name).string();
  fs::create_directories(file_name_);
  log_file_.open((fs::path(file_name_) / "train_log.txt").string(), std::ios::app);
  if (exp_.seed) rng_.seed(*exp_.seed);
}

void Trainer::log(const std::string& msg) {
  const std::string line = timestamp() + " | " + msg;
  std::cout << line << std::endl;
  if (log_file_) log_file_ << line << std::endl;
}

void Trainer::train() {
  before_train();
  try {
    train_in_epoch();
  } catch (const std::exception& e) {
    log(std::string("Exception in training: ") + e.what());
    after_train();
    throw;
  }
  after_train();
}

void Trainer::train_in_epoch() {
  for (epoch_ = start_epoch_; epoch_ < max_epoch_; ++epoch_) {
    before_epoch();
    for (iter_ = 0; iter_ < max_iter_; ++iter_) {
      train_one_iter();
      after_iter();
    }
    after_epoch();
  }
}

void Trainer::build_optimizer() {
  const double lr = exp_.warmup_epochs > 0 ? exp_.warmup_lr
                                           : exp_.basic_lr_per_img * args_.batch_size;
  std::vector<torch::Tensor> pg0, pg1, pg2;  // bn weights (no decay), weights (decay), biases
  for (const auto& p : model_->named_parameters()) {
    const auto& k = p.key();
    if (ends_with(k, ".bias")) pg2.push_back(p.value());
    else if (k.find(".bn.") != std::string::npos || k.rfind("bn.", 0) == 0) pg0.push_back(p.value());
    else pg1.push_back(p.value());
  }
  using torch::optim::SGDOptions;
  auto mk = [&](double wd) {
    auto o = std::make_unique<SGDOptions>(lr);
    o->momentum(exp_.momentum).nesterov(true).weight_decay(wd);
    return o;
  };
  std::vector<torch::optim::OptimizerParamGroup> groups;
  groups.emplace_back(pg0, mk(0.0));
  groups.emplace_back(pg1, mk(exp_.weight_decay));
  groups.emplace_back(pg2, mk(0.0));
  optimizer_ = std::make_unique<torch::optim::SGD>(
      std::move(groups), SGDOptions(lr).momentum(exp_.momentum).nesterov(true));
  params_ = model_->parameters();
}

void Trainer::before_train() {
  if (args_.device >= 0 && torch::cuda::is_available()) {
    device_ = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(args_.device));
  } else {
    if (args_.device >= 0) log("CUDA not available, training on CPU (slow).");
    device_ = torch::Device(torch::kCPU);
    args_.fp16 = false;
  }
  if (exp_.seed) {
    torch::manual_seed(*exp_.seed);
    at::globalContext().setDeterministicCuDNN(true);
    log("You have chosen to seed training. This will turn on the CUDNN deterministic setting.");
  }
  at::globalContext().setBenchmarkCuDNN(true);

  std::ostringstream a;
  a << "args: batch_size=" << args_.batch_size << " ckpt=" << args_.ckpt
    << " resume=" << args_.resume << " fp16=" << args_.fp16 << " cache=" << args_.cache
    << " experiment_name=" << args_.experiment_name;
  log(a.str());
  log("exp value:\n" + exp_.to_string());

  model_ = build_model(exp_.model_config());
  int64_t n_params = 0;
  for (auto& p : model_->parameters()) n_params += p.numel();
  {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "Model Summary: Params: %.2fM", n_params / 1e6);
    log(buf);
  }
  model_->to(device_);
  build_optimizer();
  scaler_ = GradScaler(args_.fp16);

  resume_train();

  no_aug_ = start_epoch_ >= max_epoch_ - exp_.no_aug_epochs;

  // data
  log("Loading training annotations...");
  const cv::Size in_wh(exp_.input_size.second, exp_.input_size.first);
  auto train_ds = std::make_shared<COCODataset>(exp_.data_dir, exp_.train_ann, exp_.train_name,
                                                in_wh, args_.cache, exp_.data_num_workers);
  MosaicConfig mc;
  mc.enable_mosaic = !no_aug_;
  mc.degrees = exp_.degrees;
  mc.translate = exp_.translate;
  mc.mosaic_scale = exp_.mosaic_scale;
  mc.mixup_scale = exp_.mixup_scale;
  mc.shear = exp_.shear;
  mc.enable_mixup = exp_.enable_mixup;
  mc.mosaic_prob = exp_.mosaic_prob;
  mc.mixup_prob = exp_.mixup_prob;
  TrainTransform tt;
  tt.max_labels = 120;
  tt.flip_prob = exp_.flip_prob;
  tt.hsv_prob = exp_.hsv_prob;
  auto mosaic = std::make_shared<MosaicDetection>(train_ds, in_wh, mc, tt);
  train_loader_ = std::make_unique<TrainLoader>(mosaic, args_.batch_size, exp_.data_num_workers,
                                                exp_.seed ? *exp_.seed : 0);
  max_iter_ = train_loader_->iters_per_epoch();

  lr_scheduler_ = std::make_unique<LRScheduler>(
      exp_.scheduler, exp_.basic_lr_per_img * args_.batch_size, max_iter_, max_epoch_,
      exp_.warmup_epochs, exp_.warmup_lr, exp_.no_aug_epochs, exp_.min_lr_ratio);

  if (exp_.ema) {
    ema_model_ = std::make_unique<ModelEMA>(model_, 0.9998);
    ema_model_->updates = max_iter_ * start_epoch_;
  }

  log("Loading validation annotations...");
  const cv::Size test_wh(exp_.test_size.second, exp_.test_size.first);
  auto val_ds = std::make_shared<COCODataset>(exp_.data_dir, exp_.val_ann, exp_.val_name, test_wh,
                                              false);
  evaluator_ = std::make_unique<COCOEvaluator>(val_ds, args_.batch_size, exp_.data_num_workers,
                                               test_wh, exp_.test_conf, exp_.nmsthre,
                                               exp_.num_classes);
  if (static_cast<int>(train_ds->class_ids().size()) != exp_.num_classes) {
    log("[warn] num_classes=" + std::to_string(exp_.num_classes) + " but the annotation file has " +
        std::to_string(train_ds->class_ids().size()) + " categories");
  }
  log("Training start...");
}

void Trainer::after_train() {
  char buf[128];
  std::snprintf(buf, sizeof(buf), "Training of experiment is done and the best AP is %.2f",
                best_ap_ * 100);
  log(buf);
}

void Trainer::before_epoch() {
  log("---> start train epoch" + std::to_string(epoch_ + 1));
  if (epoch_ + 1 == max_epoch_ - exp_.no_aug_epochs || no_aug_) {
    log("--->No mosaic aug now!");
    train_loader_->close_mosaic();
    log("--->Add additional L1 loss now!");
    model_->head->use_l1 = true;
    exp_.eval_interval = 1;
    if (!no_aug_) save_ckpt("last_mosaic_epoch");
  }
}

void Trainer::after_epoch() {
  save_ckpt("latest");
  if ((epoch_ + 1) % exp_.eval_interval == 0) evaluate_and_save_model();
}

void Trainer::preprocess(torch::Tensor& inputs, torch::Tensor& targets,
                         std::pair<int, int> tsize) const {
  const double scale_y = static_cast<double>(tsize.first) / exp_.input_size.first;
  const double scale_x = static_cast<double>(tsize.second) / exp_.input_size.second;
  if (scale_x != 1 || scale_y != 1) {
    namespace F = torch::nn::functional;
    inputs = F::interpolate(inputs, F::InterpolateFuncOptions()
                                        .size(std::vector<int64_t>{tsize.first, tsize.second})
                                        .mode(torch::kBilinear)
                                        .align_corners(false));
    using torch::indexing::Slice;
    targets.index({"...", Slice(1, torch::indexing::None, 2)}).mul_(scale_x);
    targets.index({"...", Slice(2, torch::indexing::None, 2)}).mul_(scale_y);
  }
}

void Trainer::train_one_iter() {
  const double iter_start = now_s();
  Batch b = train_loader_->next();
  auto inps = b.images.to(device_, /*non_blocking=*/true);
  auto targets = b.targets.to(device_, /*non_blocking=*/true);
  preprocess(inps, targets, input_size_);
  const double data_end = now_s();

  LossOutput out;
  {
    AutocastGuard ag(args_.fp16);
    out = model_->forward_train(inps, targets);
  }
  optimizer_->zero_grad();
  scaler_.scale(out.total_loss).backward();
  scaler_.step(*optimizer_, params_);

  if (ema_model_) ema_model_->update(model_);

  const double lr = lr_scheduler_->update_lr(progress_in_iter() + 1);
  for (auto& g : optimizer_->param_groups())
    static_cast<torch::optim::SGDOptions&>(g.options()).lr(lr);

  const double iter_end = now_s();
  const size_t win = static_cast<size_t>(std::max(1, exp_.print_interval));
  m_iter_time_.update(iter_end - iter_start, win);
  m_data_time_.update(data_end - iter_start, win);
  m_lr_.update(lr, win);
  m_total_.update(out.total_loss.item<double>(), win);
  m_iou_.update(out.iou_loss.item<double>(), win);
  m_l1_.update(out.l1_loss.item<double>(), win);
  m_conf_.update(out.conf_loss.item<double>(), win);
  m_cls_.update(out.cls_loss.item<double>(), win);
  m_numfg_.update(out.num_fg, win);
}

void Trainer::after_iter() {
  if ((iter_ + 1) % exp_.print_interval == 0) {
    const int64_t left_iters = max_iter_ * max_epoch_ - (progress_in_iter() + 1);
    const double eta_seconds = m_iter_time_.global_avg() * left_iters;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "epoch: %d/%d, iter: %lld/%lld, iter_time: %.3fs, data_time: %.3fs, "
                  "total_loss: %.1f, iou_loss: %.1f, l1_loss: %.1f, conf_loss: %.1f, "
                  "cls_loss: %.1f, lr: %.3e, size: %d, ETA: %s",
                  epoch_ + 1, max_epoch_, static_cast<long long>(iter_ + 1),
                  static_cast<long long>(max_iter_), m_iter_time_.avg(), m_data_time_.avg(),
                  m_total_.latest(), m_iou_.latest(), m_l1_.latest(), m_conf_.latest(),
                  m_cls_.latest(), m_lr_.latest(), input_size_.first, fmt_eta(eta_seconds).c_str());
    log(buf);
    for (Meter* m : {&m_iter_time_, &m_data_time_, &m_lr_, &m_total_, &m_iou_, &m_l1_, &m_conf_,
                     &m_cls_, &m_numfg_})
      m->clear();
  }
  if ((progress_in_iter() + 1) % 10 == 0) input_size_ = random_resize();
}

std::pair<int, int> Trainer::random_resize() {
  const double size_factor = exp_.input_size.second * 1.0 / exp_.input_size.first;
  if (!exp_.random_size) {
    const int min_size = exp_.input_size.first / 32 - exp_.multiscale_range;
    const int max_size = exp_.input_size.first / 32 + exp_.multiscale_range;
    exp_.random_size = std::make_pair(min_size, max_size);
  }
  const int size = static_cast<int>(randint(rng_, exp_.random_size->first, exp_.random_size->second));
  return {32 * size, 32 * static_cast<int>(size * size_factor)};
}

void Trainer::resume_train() {
  if (args_.resume) {
    log("resume training");
    std::string ckpt_file = args_.ckpt.empty()
                                ? (fs::path(file_name_) / "latest_ckpt.pth").string()
                                : args_.ckpt;
    auto ckpt = read_checkpoint(ckpt_file);
    load_ckpt(*model_, ckpt.model);
    std::string optim_file = ckpt_file;
    if (ends_with(optim_file, "_ckpt.pth"))
      optim_file = optim_file.substr(0, optim_file.size() - 9) + "_optim.pt";
    else
      optim_file += ".optim.pt";
    if (fs::exists(optim_file)) {
      torch::load(*optimizer_, optim_file);
    } else {
      log("[warn] optimizer state " + optim_file + " not found, starting with fresh momentum");
    }
    best_ap_ = ckpt.best_ap.value_or(0.0);
    start_epoch_ = args_.start_epoch ? *args_.start_epoch - 1
                                     : static_cast<int>(ckpt.start_epoch.value_or(0));
    log("loaded checkpoint '" + ckpt_file + "' (epoch " + std::to_string(start_epoch_) + ")");
  } else {
    if (!args_.ckpt.empty()) {
      log("loading checkpoint for fine tuning");
      auto ckpt = read_checkpoint(args_.ckpt);
      const size_t n = load_ckpt(*model_, ckpt.model);
      log("loaded " + std::to_string(n) + " tensors from " + args_.ckpt);
    }
    start_epoch_ = 0;
  }
}

void Trainer::evaluate_and_save_model() {
  YOLOX evalmodel = ema_model_ ? ema_model_->ema : model_;
  const bool was_training = evalmodel->is_training();
  auto res = evaluator_->evaluate(evalmodel, device_);
  evalmodel->train(was_training);

  const bool update_best_ckpt = res.ap50_95 > best_ap_;
  best_ap_ = std::max(best_ap_, res.ap50_95);
  log("\n" + res.summary);
  save_ckpt("last_epoch", update_best_ckpt, res.ap50_95);
  if (exp_.save_history_ckpt) save_ckpt("epoch_" + std::to_string(epoch_ + 1), false, res.ap50_95);
}

void Trainer::save_ckpt(const std::string& name, bool update_best, std::optional<double> ap) {
  YOLOX save_model = ema_model_ ? ema_model_->ema : model_;
  log("Save weights to " + file_name_);
  const auto path = (fs::path(file_name_) / (name + "_ckpt.pth")).string();
  save_checkpoint(path, *save_model, epoch_ + 1, best_ap_, ap);
  torch::save(*optimizer_, (fs::path(file_name_) / (name + "_optim.pt")).string());
  if (update_best) {
    fs::copy_file(path, fs::path(file_name_) / "best_ckpt.pth", fs::copy_options::overwrite_existing);
    fs::copy_file(fs::path(file_name_) / (name + "_optim.pt"), fs::path(file_name_) / "best_optim.pt",
                  fs::copy_options::overwrite_existing);
  }
}

}  // namespace yolox
