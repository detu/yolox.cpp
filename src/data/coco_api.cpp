#include "data/coco_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "utils/json.hpp"

namespace yolox {

// ----------------------------------------------------------------- loading
CocoApi::CocoApi(const std::string& annotation_file) {
  // "segmentation" is dropped just like remove_useless_info() in coco.py
  auto root = json::parse_file(annotation_file, {"segmentation", "licenses", "info"});

  for (const auto& im : root.at("images").arr()) {
    CocoImage ci;
    ci.id = im.at("id").i64();
    ci.width = static_cast<int>(im.at("width").i64());
    ci.height = static_cast<int>(im.at("height").i64());
    if (const auto* fn = im.find("file_name")) ci.file_name = fn->str();
    img_index[ci.id] = images.size();
    images.push_back(std::move(ci));
  }
  if (const auto* a = root.find("annotations")) {
    for (const auto& an : a->arr()) {
      CocoAnn ca;
      ca.id = an.at("id").i64();
      ca.image_id = an.at("image_id").i64();
      ca.category_id = an.at("category_id").i64();
      const auto& bb = an.at("bbox").arr();
      for (int i = 0; i < 4; ++i) ca.bbox[i] = bb.at(i).num();
      if (const auto* ar = an.find("area")) ca.area = ar->num();
      else ca.area = ca.bbox[2] * ca.bbox[3];
      if (const auto* c = an.find("iscrowd")) ca.iscrowd = c->boolean();
      if (const auto* ig = an.find("ignore")) ca.ignore = ig->boolean();
      img_to_anns[ca.image_id].push_back(anns.size());
      anns.push_back(ca);
    }
  }
  if (const auto* c = root.find("categories")) {
    for (const auto& cat : c->arr()) {
      CocoCategory cc;
      cc.id = cat.at("id").i64();
      if (const auto* n = cat.find("name")) cc.name = n->str();
      cats.push_back(std::move(cc));
    }
  }
}

std::vector<int64_t> CocoApi::sorted_cat_ids() const {
  std::vector<int64_t> ids;
  for (const auto& c : cats) ids.push_back(c.id);
  std::sort(ids.begin(), ids.end());
  return ids;
}

void write_coco_results(const std::string& path, const std::vector<CocoDet>& dets) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.precision(10);
  f << "[";
  for (size_t i = 0; i < dets.size(); ++i) {
    const auto& d = dets[i];
    if (i) f << ",";
    f << "{\"image_id\":" << d.image_id << ",\"category_id\":" << d.category_id
      << ",\"bbox\":[" << d.bbox[0] << "," << d.bbox[1] << "," << d.bbox[2] << ","
      << d.bbox[3] << "],\"score\":" << d.score << ",\"segmentation\":[]}";
  }
  f << "]";
}

// ----------------------------------------------------------------- COCOeval
namespace {

// np.linspace(start, stop, num)
std::vector<double> linspace(double start, double stop, int num) {
  std::vector<double> v(num);
  const double step = (stop - start) / (num - 1);
  for (int i = 0; i < num; ++i) v[i] = i * step + start;
  v[num - 1] = stop;
  return v;
}

struct Box {
  double x, y, w, h;
};

// maskApi.c bbIou
double bb_iou(const Box& d, const Box& g, bool crowd) {
  double w = std::min(d.x + d.w, g.x + g.w) - std::max(d.x, g.x);
  if (w <= 0) return 0;
  double h = std::min(d.y + d.h, g.y + g.h) - std::max(d.y, g.y);
  if (h <= 0) return 0;
  double inter = w * h;
  double da = d.w * d.h, ga = g.w * g.h;
  double u = crowd ? da : da + ga - inter;
  return inter / u;
}

struct GtItem {
  Box box;
  double area;
  bool iscrowd;
  bool ignore;
};
struct DtItem {
  Box box;
  double area;
  double score;
};

struct EvalImg {
  std::vector<double> dt_scores;               // D (sorted desc, <= maxDet)
  std::vector<std::vector<char>> dt_matched;   // T x D
  std::vector<std::vector<char>> dt_ignore;    // T x D
  std::vector<char> gt_ignore;                 // G
};

}  // namespace

CocoEvalResult coco_evaluate_bbox(const CocoApi& gt, const std::vector<CocoDet>& dets) {
  const auto iou_thrs = linspace(0.5, 0.95, 10);
  const auto rec_thrs = linspace(0.0, 1.0, 101);
  const std::vector<int> max_dets{1, 10, 100};
  const double big = 1e10;  // 1e5**2
  const std::vector<std::pair<double, double>> area_rng{
      {0, big}, {0, 32.0 * 32.0}, {32.0 * 32.0, 96.0 * 96.0}, {96.0 * 96.0, big}};
  const std::vector<std::string> area_lbl{"all", "small", "medium", "large"};

  const auto cat_ids = gt.sorted_cat_ids();
  const size_t T = iou_thrs.size(), R = rec_thrs.size(), K = cat_ids.size(),
               A = area_rng.size(), M = max_dets.size(), I = gt.images.size();
  std::unordered_map<int64_t, size_t> cat_index;
  for (size_t k = 0; k < K; ++k) cat_index[cat_ids[k]] = k;

  // _prepare: per (image, category) gt and dt lists
  std::vector<std::vector<GtItem>> gts(I * K);
  std::vector<std::vector<DtItem>> dts(I * K);
  for (const auto& a : gt.anns) {
    auto ii = gt.img_index.find(a.image_id);
    auto kk = cat_index.find(a.category_id);
    if (ii == gt.img_index.end() || kk == cat_index.end()) continue;
    // pycocotools: gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
    gts[ii->second * K + kk->second].push_back(
        {{a.bbox[0], a.bbox[1], a.bbox[2], a.bbox[3]}, a.area, a.iscrowd, a.iscrowd});
  }
  for (const auto& d : dets) {
    auto ii = gt.img_index.find(d.image_id);
    auto kk = cat_index.find(d.category_id);
    if (ii == gt.img_index.end())
      throw std::runtime_error("Results do not correspond to current coco set");
    if (kk == cat_index.end()) continue;
    dts[ii->second * K + kk->second].push_back(
        {{d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]}, d.bbox[2] * d.bbox[3], d.score});
  }

  const int max_det = max_dets.back();
  // evalImgs indexed [k][a][i]
  std::vector<EvalImg> eval_imgs(K * A * I);
  std::vector<char> eval_valid(K * A * I, 0);

  for (size_t k = 0; k < K; ++k) {
    for (size_t i = 0; i < I; ++i) {
      auto& g_all = gts[i * K + k];
      auto& d_all = dts[i * K + k];
      if (g_all.empty() && d_all.empty()) continue;

      // computeIoU: dt sorted by score (stable), truncated to maxDets[-1]
      std::vector<size_t> dord(d_all.size());
      std::iota(dord.begin(), dord.end(), 0);
      std::stable_sort(dord.begin(), dord.end(), [&](size_t x, size_t y) {
        return d_all[x].score > d_all[y].score;
      });
      if (dord.size() > static_cast<size_t>(max_det)) dord.resize(max_det);
      const size_t D = dord.size();

      for (size_t a = 0; a < A; ++a) {
        const auto& rng = area_rng[a];
        const size_t G = g_all.size();
        std::vector<char> g_ig(G);
        for (size_t g = 0; g < G; ++g) {
          g_ig[g] = (g_all[g].ignore || g_all[g].area < rng.first || g_all[g].area > rng.second) ? 1 : 0;
        }
        std::vector<size_t> gord(G);
        std::iota(gord.begin(), gord.end(), 0);
        std::stable_sort(gord.begin(), gord.end(),
                         [&](size_t x, size_t y) { return g_ig[x] < g_ig[y]; });

        EvalImg e;
        e.gt_ignore.resize(G);
        for (size_t g = 0; g < G; ++g) e.gt_ignore[g] = g_ig[gord[g]];
        e.dt_scores.resize(D);
        for (size_t d = 0; d < D; ++d) e.dt_scores[d] = d_all[dord[d]].score;
        e.dt_matched.assign(T, std::vector<char>(D, 0));
        e.dt_ignore.assign(T, std::vector<char>(D, 0));

        if (G > 0 && D > 0) {
          std::vector<double> ious(D * G);
          for (size_t d = 0; d < D; ++d)
            for (size_t g = 0; g < G; ++g)
              ious[d * G + g] = bb_iou(d_all[dord[d]].box, g_all[gord[g]].box,
                                       g_all[gord[g]].iscrowd);
          std::vector<std::vector<char>> gtm(T, std::vector<char>(G, 0));
          for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < D; ++d) {
              double iou = std::min(iou_thrs[t], 1 - 1e-10);
              long m = -1;
              for (size_t g = 0; g < G; ++g) {
                if (gtm[t][g] && !g_all[gord[g]].iscrowd) continue;
                if (m > -1 && e.gt_ignore[m] == 0 && e.gt_ignore[g] == 1) break;
                if (ious[d * G + g] < iou) continue;
                iou = ious[d * G + g];
                m = static_cast<long>(g);
              }
              if (m == -1) continue;
              e.dt_ignore[t][d] = e.gt_ignore[m];
              e.dt_matched[t][d] = 1;
              gtm[t][m] = 1;
            }
          }
        }
        for (size_t d = 0; d < D; ++d) {
          const double ar = d_all[dord[d]].area;
          const bool out_of_rng = ar < rng.first || ar > rng.second;
          if (!out_of_rng) continue;
          for (size_t t = 0; t < T; ++t)
            if (!e.dt_matched[t][d]) e.dt_ignore[t][d] = 1;
        }
        const size_t idx = (k * A + a) * I + i;
        eval_imgs[idx] = std::move(e);
        eval_valid[idx] = 1;
      }
    }
  }

  // accumulate
  std::vector<double> precision(T * R * K * A * M, -1.0);
  std::vector<double> recall(T * K * A * M, -1.0);
  auto P = [&](size_t t, size_t r, size_t k, size_t a, size_t m) -> double& {
    return precision[(((t * R + r) * K + k) * A + a) * M + m];
  };
  auto RC = [&](size_t t, size_t k, size_t a, size_t m) -> double& {
    return recall[((t * K + k) * A + a) * M + m];
  };
  const double eps = std::numeric_limits<double>::epsilon();  // np.spacing(1)

  for (size_t k = 0; k < K; ++k) {
    for (size_t a = 0; a < A; ++a) {
      for (size_t m = 0; m < M; ++m) {
        const size_t md = static_cast<size_t>(max_dets[m]);
        struct Ref { size_t img; size_t d; double score; };
        std::vector<Ref> all;
        size_t npig = 0;
        bool any = false;
        for (size_t i = 0; i < I; ++i) {
          const size_t idx = (k * A + a) * I + i;
          if (!eval_valid[idx]) continue;
          any = true;
          const auto& e = eval_imgs[idx];
          const size_t nd = std::min(md, e.dt_scores.size());
          for (size_t d = 0; d < nd; ++d) all.push_back({idx, d, e.dt_scores[d]});
          for (char gi : e.gt_ignore) npig += (gi == 0);
        }
        if (!any || npig == 0) continue;
        std::stable_sort(all.begin(), all.end(),
                         [](const Ref& x, const Ref& y) { return x.score > y.score; });
        const size_t nd = all.size();
        for (size_t t = 0; t < T; ++t) {
          std::vector<double> rc(nd), pr(nd);
          double tp = 0, fp = 0;
          for (size_t j = 0; j < nd; ++j) {
            const auto& e = eval_imgs[all[j].img];
            const bool ig = e.dt_ignore[t][all[j].d];
            const bool mt = e.dt_matched[t][all[j].d];
            if (!ig) {
              if (mt) tp += 1; else fp += 1;
            }
            rc[j] = tp / static_cast<double>(npig);
            pr[j] = tp / (fp + tp + eps);
          }
          RC(t, k, a, m) = nd ? rc[nd - 1] : 0.0;
          for (size_t j = nd; j-- > 1;)
            if (pr[j] > pr[j - 1]) pr[j - 1] = pr[j];
          for (size_t r = 0; r < R; ++r) {
            // np.searchsorted(rc, recThrs, side='left')
            size_t pi = static_cast<size_t>(
                std::lower_bound(rc.begin(), rc.end(), rec_thrs[r]) - rc.begin());
            P(t, r, k, a, m) = (pi < nd) ? pr[pi] : 0.0;
          }
        }
      }
    }
  }

  // summarize
  CocoEvalResult res;
  std::ostringstream os;
  auto summ = [&](bool ap, int iou_idx, size_t a, size_t m) {
    double sum = 0;
    size_t cnt = 0;
    for (size_t t = 0; t < T; ++t) {
      if (iou_idx >= 0 && static_cast<int>(t) != iou_idx) continue;
      for (size_t k = 0; k < K; ++k) {
        if (ap) {
          for (size_t r = 0; r < R; ++r) {
            double v = P(t, r, k, a, m);
            if (v > -1) { sum += v; ++cnt; }
          }
        } else {
          double v = RC(t, k, a, m);
          if (v > -1) { sum += v; ++cnt; }
        }
      }
    }
    double mean = cnt ? sum / cnt : -1.0;
    char iou_str[32];
    if (iou_idx < 0) std::snprintf(iou_str, sizeof(iou_str), "%0.2f:%0.2f", iou_thrs.front(), iou_thrs.back());
    else std::snprintf(iou_str, sizeof(iou_str), "%0.2f", iou_thrs[iou_idx]);
    char line[256];
    std::snprintf(line, sizeof(line), " %-18s %s @[ IoU=%-9s | area=%6s | maxDets=%3d ] = %0.3f\n",
                  ap ? "Average Precision" : "Average Recall", ap ? "(AP)" : "(AR)", iou_str,
                  area_lbl[a].c_str(), max_dets[m], mean);
    os << line;
    return mean;
  };
  res.stats[0] = summ(true, -1, 0, 2);
  res.stats[1] = summ(true, 0, 0, 2);
  res.stats[2] = summ(true, 5, 0, 2);
  res.stats[3] = summ(true, -1, 1, 2);
  res.stats[4] = summ(true, -1, 2, 2);
  res.stats[5] = summ(true, -1, 3, 2);
  res.stats[6] = summ(false, -1, 0, 0);
  res.stats[7] = summ(false, -1, 0, 1);
  res.stats[8] = summ(false, -1, 0, 2);
  res.stats[9] = summ(false, -1, 1, 2);
  res.stats[10] = summ(false, -1, 2, 2);
  res.stats[11] = summ(false, -1, 3, 2);
  res.summary = os.str();

  // per-class tables, as in coco_evaluator.per_class_AP_table / AR_table
  for (size_t k = 0; k < K; ++k) {
    double s = 0; size_t c = 0;
    for (size_t t = 0; t < T; ++t)
      for (size_t r = 0; r < R; ++r) {
        double v = P(t, r, k, 0, M - 1);
        if (v > -1) { s += v; ++c; }
      }
    res.per_class_ap.push_back(c ? s / c : -1.0);
    s = 0; c = 0;
    for (size_t t = 0; t < T; ++t) {
      double v = RC(t, k, 0, M - 1);
      if (v > -1) { s += v; ++c; }
    }
    res.per_class_ar.push_back(c ? s / c : -1.0);
  }
  return res;
}

}  // namespace yolox
