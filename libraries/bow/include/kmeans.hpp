#ifndef KMEANS_HPP_
#define KMEANS_HPP_

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>
#include <vector>

#include "bow.hpp"
#include "convert_dataset.hpp"
#include "serialize.hpp"

namespace ipb {
cv::Mat stackMatrices(const std::vector<cv::Mat> &descriptors);

cv::Mat kMeans(const std::vector<cv::Mat> &descriptors, int k, int max_iter);

} // namespace ipb

#endif // KMEANS_HPP_