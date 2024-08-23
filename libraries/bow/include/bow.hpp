#ifndef BOW_HPP_
#define BOW_HPP_

#include "../../serialization/include/convert_dataset.hpp"
#include "../../serialization/include/serialize.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <thread>
#include <vector>

namespace ipb {
cv::Mat stackMatrices(const std::vector<cv::Mat> &descriptors);
cv::Mat kMeans(const std::vector<cv::Mat> &descriptors, int k, int max_iter);
cv::Mat computeHistogram(cv::Mat &descriptors, cv::Mat &dictionary);
std::vector<cv::Mat> ComputeAllHistograms(std::vector<cv::Mat> &loaded_bins,
                                          cv::Mat &visual_dictionary);
class BowDictionary {
private:
  // singleton
  BowDictionary() = default;
  ~BowDictionary() = default;
  static BowDictionary *instancePtr;

  cv::Mat dictionary;

public:
  // singleton
  BowDictionary(const BowDictionary &obj) = delete;
  void operator=(const BowDictionary &) = delete;
  BowDictionary(BowDictionary &&other) = delete;
  BowDictionary &operator=(BowDictionary &&other) = delete;
  cv::Mat voc_dictionary() const { return dictionary; };
  cv::Mat &voc_dictionary() { return dictionary; };
  // int total_features() const { return descriptors().size(); };
  bool empty() { return dictionary.empty(); };

  void build(int max_iter, int size, const std::vector<cv::Mat> &descriptors);

  void set_vocabulary(const std::string &filename);

  void save_vocabulary(const std::string &filename);

  static BowDictionary &GetInstance() {
    static BowDictionary instance;
    return instance;
  }
};

} // namespace ipb
#endif