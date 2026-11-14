// Torch-free check of the JSON reader + COCOeval port.
//   test_coco_eval instances_val.json detections.json
// prints COCOeval.summarize() and the 12 stats.
#include <cstdio>
#include <iostream>

#include "data/coco_api.hpp"
#include "utils/json.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: test_coco_eval GT.json DETS.json\n";
    return 1;
  }
  yolox::CocoApi gt(argv[1]);
  auto dj = yolox::json::parse_file(argv[2]);
  std::vector<yolox::CocoDet> dets;
  for (const auto& d : dj.arr()) {
    yolox::CocoDet c;
    c.image_id = d.at("image_id").i64();
    c.category_id = d.at("category_id").i64();
    for (int i = 0; i < 4; ++i) c.bbox[i] = d.at("bbox")[i].num();
    c.score = d.at("score").num();
    dets.push_back(c);
  }
  auto r = yolox::coco_evaluate_bbox(gt, dets);
  std::cout << r.summary;
  for (double s : r.stats) std::printf("%.12f\n", s);
  for (double s : r.per_class_ap) std::printf("ap %.12f\n", s);
  return 0;
}
