// Minimal image inference demo (C++ counterpart of tools/demo.py "image" mode)
//
//   yolox_demo -f exps/yolox_m.cfg -c best_ckpt.pth --path img.jpg --conf 0.3 --out vis.jpg
//              [--classes cat,dog,...] [--tsize 640] [--nms 0.45]
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <sstream>

#include "cli.hpp"
#include "core/checkpoint.hpp"
#include "data/data_augment.hpp"
#include "data/dataloader.hpp"
#include "utils/boxes.hpp"

int main(int argc, char** argv) {
  try {
    auto p = yolox::cli::parse(argc, argv, {"-h", "--help"});
    if (yolox::cli::has(p, {"-h", "--help"}) || argc == 1) {
      std::cout << "usage: yolox_demo [-f EXP | -n MODEL] -c CKPT --path IMG [--out OUT.jpg] "
                   "[--conf 0.3] [--nms 0.45] [--tsize 640] [--classes a,b,c] [--device IDX]\n";
      return 0;
    }
    yolox::Exp exp = yolox::cli::build_exp(p);
    const std::string ckpt = yolox::cli::get(p, {"-c", "--ckpt"});
    const std::string path = yolox::cli::get(p, {"--path"});
    const std::string out = yolox::cli::get(p, {"--out"}, "yolox_vis.jpg");
    const double conf = std::stod(yolox::cli::get(p, {"--conf"}, "0.3"));
    const double nms = std::stod(yolox::cli::get(p, {"--nms"}, "0.45"));
    const auto tsize = yolox::cli::get(p, {"--tsize"});
    if (!tsize.empty()) exp.test_size = {std::stoi(tsize), std::stoi(tsize)};
    std::vector<std::string> names;
    {
      std::stringstream ss(yolox::cli::get(p, {"--classes"}));
      std::string n;
      while (std::getline(ss, n, ',')) names.push_back(n);
    }
    const int dev = std::stoi(yolox::cli::get(p, {"--device"}, "0"));
    torch::Device device = (dev >= 0 && torch::cuda::is_available())
                               ? torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(dev))
                               : torch::Device(torch::kCPU);

    auto model = yolox::build_model(exp.model_config());
    yolox::load_ckpt(*model, yolox::read_checkpoint(ckpt).model);
    model->to(device);
    model->eval();
    torch::NoGradGuard ng;

    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("cannot read image " + path);
    const cv::Size test_wh(exp.test_size.second, exp.test_size.first);
    auto pr = yolox::preproc(img, test_wh);
    auto input = yolox::images_to_tensor({pr.first}, false).to(device);
    auto dets = yolox::postprocess(model->forward(input), exp.num_classes, conf, nms);

    int n = 0;
    if (dets[0].defined()) {
      auto d = dets[0].to(torch::kCPU, torch::kFloat).contiguous();
      auto a = d.accessor<float, 2>();
      for (int64_t k = 0; k < d.size(0); ++k) {
        const float r = static_cast<float>(pr.second);
        cv::Point p1(static_cast<int>(a[k][0] / r), static_cast<int>(a[k][1] / r));
        cv::Point p2(static_cast<int>(a[k][2] / r), static_cast<int>(a[k][3] / r));
        const int cls = static_cast<int>(a[k][6]);
        const float score = a[k][4] * a[k][5];
        std::string label = (cls < static_cast<int>(names.size()) ? names[cls] : std::to_string(cls));
        char txt[128];
        std::snprintf(txt, sizeof(txt), "%s:%.1f%%", label.c_str(), score * 100);
        cv::Scalar color((cls * 67) % 255, (cls * 131 + 80) % 255, (cls * 199 + 160) % 255);
        cv::rectangle(img, p1, p2, color, 2);
        cv::putText(img, txt, cv::Point(p1.x, std::max(0, p1.y - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1);
        std::cout << label << " " << score << " [" << p1.x << ", " << p1.y << ", " << p2.x << ", "
                  << p2.y << "]\n";
        ++n;
      }
    }
    cv::imwrite(out, img);
    std::cout << n << " detections, saved " << out << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
