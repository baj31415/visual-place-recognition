#ifndef PLACE_RECOGNITION_HPP_
#define PLACE_RECOGNITION_HPP_
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

#include "../libraries/bow/include/bow.hpp"
#include "../libraries/html_writer/include/html_writer.hpp"
#include "../libraries/image_browser/include/image_browser.hpp"
#include "../libraries/serialization/include/convert_dataset.hpp"
#include "../libraries/serialization/include/serialize.hpp"

namespace ipb {

cv::Mat TF_IDF(cv::Mat &stacked_histogram);
double cosine_distance(cv::Mat &hist1, cv::Mat &hist2);

} // namespace ipb
#endif