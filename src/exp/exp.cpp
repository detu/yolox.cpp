#include "exp/exp.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>

namespace yolox {

namespace {

std::string trim(std::string s) {
  auto notspace = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
  s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
  return s;
}

std::string unquote(std::string s) {
  s = trim(s);
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\'')))
    return s.substr(1, s.size() - 2);
  return s;
}

bool to_bool(const std::string& v) {
  std::string s = v;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
  if (s == "false" || s == "0" || s == "no" || s == "off") return false;
  throw std::invalid_argument("not a boolean: " + v);
}

// "640,640", "(640, 640)", "[0.1, 2]"
std::vector<double> to_list(std::string v) {
  for (char& c : v)
    if (c == '(' || c == ')' || c == '[' || c == ']' || c == ',') c = ' ';
  std::istringstream is(v);
  std::vector<double> out;
  double d;
  while (is >> d) out.push_back(d);
  return out;
}

template <typename T>
std::pair<T, T> to_pair(const std::string& v) {
  auto l = to_list(v);
  if (l.size() == 1) return {static_cast<T>(l[0]), static_cast<T>(l[0])};
  if (l.size() != 2) throw std::invalid_argument("expected two values: " + v);
  return {static_cast<T>(l[0]), static_cast<T>(l[1])};
}

}  // namespace

ModelConfig Exp::model_config() const {
  ModelConfig c;
  c.num_classes = num_classes;
  c.depth = depth;
  c.width = width;
  c.depthwise = depthwise;
  c.act = act;
  return c;
}

void Exp::apply_preset(const std::string& raw, bool set_exp_name) {
  std::string n = raw;
  std::replace(n.begin(), n.end(), '_', '-');
  std::transform(n.begin(), n.end(), n.begin(), ::tolower);
  if (set_exp_name) {
    exp_name = n;
    std::replace(exp_name.begin(), exp_name.end(), '-', '_');
  }
  if (n == "yolox-s") { depth = 0.33; width = 0.50; }
  else if (n == "yolox-m") { depth = 0.67; width = 0.75; }
  else if (n == "yolox-l") { depth = 1.0; width = 1.0; }
  else if (n == "yolox-x") { depth = 1.33; width = 1.25; }
  else if (n == "yolox-tiny") {
    depth = 0.33; width = 0.375;
    input_size = {416, 416}; test_size = {416, 416};
    mosaic_scale = {0.5, 1.5}; random_size = std::make_pair(10, 20);
    enable_mixup = false;
  } else if (n == "yolox-nano") {
    depth = 0.33; width = 0.25; depthwise = true;
    input_size = {416, 416}; test_size = {416, 416};
    random_size = std::make_pair(10, 20); mosaic_scale = {0.5, 1.5};
    mosaic_prob = 0.5; enable_mixup = false;
  } else {
    throw std::invalid_argument("unknown model preset '" + raw +
                                "' (use yolox-nano/tiny/s/m/l/x; yolov3/Darknet53 is not ported)");
  }
}

void Exp::set(const std::string& key_in, const std::string& value_in) {
  const std::string key = trim(key_in);
  const std::string v = unquote(value_in);
  static const std::map<std::string, std::function<void(Exp&, const std::string&)>> table = {
      {"model", [](Exp& e, const std::string& s) { e.apply_preset(s, false); }},
      {"num_classes", [](Exp& e, const std::string& s) { e.num_classes = std::stoi(s); }},
      {"depth", [](Exp& e, const std::string& s) { e.depth = std::stod(s); }},
      {"width", [](Exp& e, const std::string& s) { e.width = std::stod(s); }},
      {"act", [](Exp& e, const std::string& s) { e.act = s; }},
      {"depthwise", [](Exp& e, const std::string& s) { e.depthwise = to_bool(s); }},
      {"data_num_workers", [](Exp& e, const std::string& s) { e.data_num_workers = std::stoi(s); }},
      {"input_size", [](Exp& e, const std::string& s) { e.input_size = to_pair<int>(s); }},
      {"multiscale_range", [](Exp& e, const std::string& s) { e.multiscale_range = std::stoi(s); }},
      {"random_size", [](Exp& e, const std::string& s) { e.random_size = to_pair<int>(s); }},
      {"data_dir", [](Exp& e, const std::string& s) { e.data_dir = s; }},
      {"train_ann", [](Exp& e, const std::string& s) { e.train_ann = s; }},
      {"val_ann", [](Exp& e, const std::string& s) { e.val_ann = s; }},
      {"train_name", [](Exp& e, const std::string& s) { e.train_name = s; }},
      {"val_name", [](Exp& e, const std::string& s) { e.val_name = s; }},
      {"mosaic_prob", [](Exp& e, const std::string& s) { e.mosaic_prob = std::stod(s); }},
      {"mixup_prob", [](Exp& e, const std::string& s) { e.mixup_prob = std::stod(s); }},
      {"hsv_prob", [](Exp& e, const std::string& s) { e.hsv_prob = std::stod(s); }},
      {"flip_prob", [](Exp& e, const std::string& s) { e.flip_prob = std::stod(s); }},
      {"degrees", [](Exp& e, const std::string& s) { e.degrees = std::stod(s); }},
      {"translate", [](Exp& e, const std::string& s) { e.translate = std::stod(s); }},
      {"mosaic_scale", [](Exp& e, const std::string& s) { e.mosaic_scale = to_pair<double>(s); }},
      {"enable_mixup", [](Exp& e, const std::string& s) { e.enable_mixup = to_bool(s); }},
      {"mixup_scale", [](Exp& e, const std::string& s) { e.mixup_scale = to_pair<double>(s); }},
      {"shear", [](Exp& e, const std::string& s) { e.shear = std::stod(s); }},
      {"warmup_epochs", [](Exp& e, const std::string& s) { e.warmup_epochs = std::stoi(s); }},
      {"max_epoch", [](Exp& e, const std::string& s) { e.max_epoch = std::stoi(s); }},
      {"warmup_lr", [](Exp& e, const std::string& s) { e.warmup_lr = std::stod(s); }},
      {"min_lr_ratio", [](Exp& e, const std::string& s) { e.min_lr_ratio = std::stod(s); }},
      {"basic_lr_per_img", [](Exp& e, const std::string& s) {
         // allow "0.01 / 64.0"
         auto p = s.find('/');
         e.basic_lr_per_img = p == std::string::npos ? std::stod(s)
                                                     : std::stod(s.substr(0, p)) / std::stod(s.substr(p + 1));
       }},
      {"scheduler", [](Exp& e, const std::string& s) { e.scheduler = s; }},
      {"no_aug_epochs", [](Exp& e, const std::string& s) { e.no_aug_epochs = std::stoi(s); }},
      {"ema", [](Exp& e, const std::string& s) { e.ema = to_bool(s); }},
      {"weight_decay", [](Exp& e, const std::string& s) { e.weight_decay = std::stod(s); }},
      {"momentum", [](Exp& e, const std::string& s) { e.momentum = std::stod(s); }},
      {"print_interval", [](Exp& e, const std::string& s) { e.print_interval = std::stoi(s); }},
      {"eval_interval", [](Exp& e, const std::string& s) { e.eval_interval = std::stoi(s); }},
      {"save_history_ckpt", [](Exp& e, const std::string& s) { e.save_history_ckpt = to_bool(s); }},
      {"exp_name", [](Exp& e, const std::string& s) { e.exp_name = s; }},
      {"output_dir", [](Exp& e, const std::string& s) { e.output_dir = s; }},
      {"seed", [](Exp& e, const std::string& s) {
         if (s == "None" || s == "none" || s.empty()) e.seed.reset();
         else e.seed = std::stoull(s);
       }},
      {"test_size", [](Exp& e, const std::string& s) { e.test_size = to_pair<int>(s); }},
      {"test_conf", [](Exp& e, const std::string& s) { e.test_conf = std::stod(s); }},
      {"nmsthre", [](Exp& e, const std::string& s) { e.nmsthre = std::stod(s); }},
  };
  // accept "self.max_epoch" as written in Python exp files
  std::string k = key;
  if (k.rfind("self.", 0) == 0) k = k.substr(5);
  auto it = table.find(k);
  if (it == table.end()) throw std::invalid_argument("unknown exp option '" + key + "'");
  it->second(*this, v);
}

void Exp::load_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open exp file: " + path);
  std::string line;
  int lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    line = trim(line);
    if (line.empty()) continue;
    auto eq = line.find('=');
    if (eq == std::string::npos)
      throw std::runtime_error(path + ":" + std::to_string(lineno) + ": expected key = value");
    set(line.substr(0, eq), line.substr(eq + 1));
  }
}

void Exp::check() const {
  if (input_size.first % 32 != 0 || input_size.second % 32 != 0)
    throw std::invalid_argument("input size must be multiples of 32");
}

std::string Exp::to_string() const {
  std::ostringstream os;
  os << "num_classes=" << num_classes << " depth=" << depth << " width=" << width
     << " act=" << act << " depthwise=" << depthwise << "\n"
     << "input_size=(" << input_size.first << "," << input_size.second << ")"
     << " test_size=(" << test_size.first << "," << test_size.second << ")"
     << " multiscale_range=" << multiscale_range;
  if (random_size) os << " random_size=(" << random_size->first << "," << random_size->second << ")";
  os << "\ndata_dir=" << data_dir << " train_ann=" << train_ann << " val_ann=" << val_ann << "\n"
     << "mosaic_prob=" << mosaic_prob << " mixup_prob=" << mixup_prob << " hsv_prob=" << hsv_prob
     << " flip_prob=" << flip_prob << " degrees=" << degrees << " translate=" << translate
     << " mosaic_scale=(" << mosaic_scale.first << "," << mosaic_scale.second << ")"
     << " enable_mixup=" << enable_mixup << " mixup_scale=(" << mixup_scale.first << ","
     << mixup_scale.second << ") shear=" << shear << "\n"
     << "max_epoch=" << max_epoch << " warmup_epochs=" << warmup_epochs
     << " no_aug_epochs=" << no_aug_epochs << " basic_lr_per_img=" << basic_lr_per_img
     << " min_lr_ratio=" << min_lr_ratio << " scheduler=" << scheduler << " ema=" << ema
     << " weight_decay=" << weight_decay << " momentum=" << momentum
     << " eval_interval=" << eval_interval << " test_conf=" << test_conf
     << " nmsthre=" << nmsthre;
  return os.str();
}

}  // namespace yolox
