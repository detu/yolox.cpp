// Small argument helper shared by the tools (mimics the argparse flags of
// tools/train.py / tools/eval.py so the README commands translate 1:1).
#pragma once

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "exp/exp.hpp"

namespace yolox {
namespace cli {

struct Parsed {
  std::vector<std::pair<std::string, std::string>> flags;  // flag -> value ("" for switches)
  std::vector<std::string> opts;                           // trailing "key value" pairs
};

// switches: flags that take no value
inline Parsed parse(int argc, char** argv, const std::vector<std::string>& switches) {
  Parsed p;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.size() > 1 && a[0] == '-' && !(a.size() > 1 && std::isdigit(static_cast<unsigned char>(a[1])))) {
      std::string val;
      auto eq = a.find('=');
      if (eq != std::string::npos) {
        val = a.substr(eq + 1);
        a = a.substr(0, eq);
      } else if (std::find(switches.begin(), switches.end(), a) == switches.end()) {
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
        val = argv[++i];
      }
      p.flags.emplace_back(a, val);
    } else {
      p.opts.push_back(a);
    }
  }
  return p;
}

inline bool has(const Parsed& p, std::initializer_list<const char*> names) {
  for (auto& f : p.flags)
    for (auto n : names)
      if (f.first == n) return true;
  return false;
}

inline std::string get(const Parsed& p, std::initializer_list<const char*> names,
                       const std::string& def = "") {
  std::string v = def;
  for (auto& f : p.flags)
    for (auto n : names)
      if (f.first == n) v = f.second;
  return v;
}

// -n preset, -f exp file, then trailing "key value" overrides (exp.merge(opts))
inline Exp build_exp(const Parsed& p) {
  Exp exp;
  const std::string name = get(p, {"-n", "--name"});
  const std::string file = get(p, {"-f", "--exp_file"});
  if (name.empty() && file.empty())
    throw std::invalid_argument("please give a model preset (-n yolox-m) or an exp file (-f my.cfg)");
  if (!name.empty()) exp.apply_preset(name);
  if (!file.empty()) {
    // like the Python exp files, exp_name defaults to the exp file's name;
    // an "exp_name = ..." line in the file overrides it
    if (name.empty()) {
      std::string base = file.substr(file.find_last_of("/\\") + 1);
      auto dot = base.find_last_of('.');
      exp.exp_name = dot == std::string::npos ? base : base.substr(0, dot);
    }
    exp.load_file(file);
  }
  if (p.opts.size() % 2 != 0) throw std::invalid_argument("opts must be 'key value' pairs");
  for (size_t i = 0; i < p.opts.size(); i += 2) exp.set(p.opts[i], p.opts[i + 1]);
  exp.check();
  return exp;
}

}  // namespace cli
}  // namespace yolox
