// Tiny subset of pycocotools.coco.COCO: just what YOLOX training/eval needs.
// Torch- and OpenCV-free.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace yolox {

struct CocoImage {
  int64_t id = 0;
  int width = 0;
  int height = 0;
  std::string file_name;  // may be empty -> "{id:012}.jpg"
};

struct CocoAnn {
  int64_t id = 0;
  int64_t image_id = 0;
  int64_t category_id = 0;
  std::array<double, 4> bbox{};  // x, y, w, h
  double area = 0.0;
  bool iscrowd = false;
  bool ignore = false;  // optional "ignore" flag honoured by COCOeval
};

struct CocoCategory {
  int64_t id = 0;
  std::string name;
};

class CocoApi {
 public:
  explicit CocoApi(const std::string& annotation_file);
  CocoApi() = default;

  std::vector<CocoImage> images;          // in file order (== getImgIds())
  std::vector<CocoAnn> anns;
  std::vector<CocoCategory> cats;         // in file order
  std::unordered_map<int64_t, std::vector<size_t>> img_to_anns;  // image id -> ann indices
  std::unordered_map<int64_t, size_t> img_index;                 // image id -> index in images

  std::vector<int64_t> sorted_cat_ids() const;
};

// One detection in COCO result format (bbox is x, y, w, h in original image px)
struct CocoDet {
  int64_t image_id = 0;
  int64_t category_id = 0;
  std::array<double, 4> bbox{};
  double score = 0.0;
};

struct CocoEvalResult {
  std::array<double, 12> stats{};   // same order as COCOeval.stats
  std::string summary;              // COCOeval.summarize() text
  std::vector<double> per_class_ap; // AP@[.5:.95] per sorted category, -1 if n/a
  std::vector<double> per_class_ar; // AR@100 per sorted category, -1 if n/a
};

// Faithful re-implementation of pycocotools COCOeval for iouType="bbox"
// (default params: 10 IoU thresholds, 101 recall points, maxDets 1/10/100,
// area ranges all/small/medium/large).
CocoEvalResult coco_evaluate_bbox(const CocoApi& gt, const std::vector<CocoDet>& dets);

// Writes detections as a COCO results json (what Python dumps before loadRes)
void write_coco_results(const std::string& path, const std::vector<CocoDet>& dets);

}  // namespace yolox
