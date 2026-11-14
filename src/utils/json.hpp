// Minimal, dependency-free JSON reader (enough for COCO annotation files).
//
// Large COCO files contain huge "segmentation" polygons that YOLOX never uses
// (remove_useless_info() drops them in Python); keys listed in `skip_keys`
// are parsed and discarded without building values, which keeps memory low.
#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace yolox {
namespace json {

class Value {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Value() = default;

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  double num() const {
    if (type_ == Type::Bool) return b_ ? 1.0 : 0.0;
    check(Type::Number, "number");
    return n_;
  }
  int64_t i64() const { return static_cast<int64_t>(num()); }
  bool boolean() const {
    if (type_ == Type::Number) return n_ != 0.0;
    check(Type::Bool, "bool");
    return b_;
  }
  const std::string& str() const {
    check(Type::String, "string");
    return s_;
  }
  const std::vector<Value>& arr() const {
    check(Type::Array, "array");
    return a_;
  }
  const std::vector<std::pair<std::string, Value>>& obj() const {
    check(Type::Object, "object");
    return o_;
  }
  size_t size() const { return is_array() ? a_.size() : is_object() ? o_.size() : 0; }
  const Value& operator[](size_t i) const { return arr().at(i); }

  // Object lookup; returns nullptr if missing
  const Value* find(const std::string& key) const {
    if (type_ != Type::Object) return nullptr;
    for (const auto& kv : o_)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  bool contains(const std::string& key) const { return find(key) != nullptr; }
  const Value& at(const std::string& key) const {
    const Value* v = find(key);
    if (!v) throw std::runtime_error("json: missing key '" + key + "'");
    return *v;
  }

 private:
  friend class Parser;
  void check(Type t, const char* what) const {
    if (type_ != t) throw std::runtime_error(std::string("json: value is not a ") + what);
  }
  Type type_ = Type::Null;
  bool b_ = false;
  double n_ = 0.0;
  std::string s_;
  std::vector<Value> a_;
  std::vector<std::pair<std::string, Value>> o_;
};

class Parser {
 public:
  Parser(const std::string& text, std::unordered_set<std::string> skip_keys)
      : t_(text), skip_(std::move(skip_keys)) {}

  Value parse() {
    Value v;
    ws();
    value(v, false);
    ws();
    if (p_ != t_.size()) fail("trailing characters");
    return v;
  }

 private:
  const std::string& t_;
  std::unordered_set<std::string> skip_;
  size_t p_ = 0;

  [[noreturn]] void fail(const std::string& msg) const {
    throw std::runtime_error("json parse error at offset " + std::to_string(p_) + ": " + msg);
  }
  void ws() {
    while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\n' || t_[p_] == '\r' || t_[p_] == '\t'))
      ++p_;
  }
  char peek() const { return p_ < t_.size() ? t_[p_] : '\0'; }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++p_;
  }

  // discard=true: parse but do not store anything
  void value(Value& v, bool discard) {
    char c = peek();
    if (c == '{') return object(v, discard);
    if (c == '[') return array(v, discard);
    if (c == '"') {
      std::string s = string();
      if (!discard) {
        v.type_ = Value::Type::String;
        v.s_ = std::move(s);
      }
      return;
    }
    if (c == 't' || c == 'f' || c == 'n') return literal(v, discard);
    return number(v, discard);
  }

  void literal(Value& v, bool discard) {
    auto match = [&](const char* w) {
      size_t n = std::char_traits<char>::length(w);
      if (t_.compare(p_, n, w) == 0) {
        p_ += n;
        return true;
      }
      return false;
    };
    if (match("true")) {
      if (!discard) { v.type_ = Value::Type::Bool; v.b_ = true; }
    } else if (match("false")) {
      if (!discard) { v.type_ = Value::Type::Bool; v.b_ = false; }
    } else if (match("null")) {
      if (!discard) v.type_ = Value::Type::Null;
    } else {
      fail("invalid literal");
    }
  }

  void number(Value& v, bool discard) {
    const char* start = t_.c_str() + p_;
    char* end = nullptr;
    double d = std::strtod(start, &end);
    if (end == start) fail("invalid number");
    p_ += static_cast<size_t>(end - start);
    if (!discard) {
      v.type_ = Value::Type::Number;
      v.n_ = d;
    }
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  uint32_t hex4() {
    if (p_ + 4 > t_.size()) fail("bad \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = t_[p_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
      else fail("bad hex digit");
    }
    return v;
  }

  std::string string() {
    expect('"');
    std::string out;
    while (true) {
      if (p_ >= t_.size()) fail("unterminated string");
      char c = t_[p_++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (p_ >= t_.size()) fail("bad escape");
      char e = t_[p_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF && t_.compare(p_, 2, "\\u") == 0) {
            p_ += 2;
            uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }

  void array(Value& v, bool discard) {
    expect('[');
    if (!discard) v.type_ = Value::Type::Array;
    ws();
    if (peek() == ']') { ++p_; return; }
    Value scratch;
    while (true) {
      ws();
      if (discard) {
        value(scratch, true);
      } else {
        v.a_.emplace_back();
        value(v.a_.back(), false);
      }
      ws();
      if (peek() == ',') { ++p_; continue; }
      expect(']');
      break;
    }
  }

  void object(Value& v, bool discard) {
    expect('{');
    if (!discard) v.type_ = Value::Type::Object;
    ws();
    if (peek() == '}') { ++p_; return; }
    Value scratch;
    while (true) {
      ws();
      std::string key = string();
      ws();
      expect(':');
      ws();
      bool skip = discard || skip_.count(key) > 0;
      if (skip) {
        value(scratch, true);
      } else {
        v.o_.emplace_back(std::move(key), Value());
        value(v.o_.back().second, false);
      }
      ws();
      if (peek() == ',') { ++p_; continue; }
      expect('}');
      break;
    }
  }
};

inline Value parse(const std::string& text, std::unordered_set<std::string> skip_keys = {}) {
  return Parser(text, std::move(skip_keys)).parse();
}

inline Value parse_file(const std::string& path, std::unordered_set<std::string> skip_keys = {}) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open json file: " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return parse(ss.str(), std::move(skip_keys));
}

// Small writer helpers (used to dump COCO-format detections)
inline std::string escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default: o += c;
    }
  }
  return o;
}

}  // namespace json
}  // namespace yolox
