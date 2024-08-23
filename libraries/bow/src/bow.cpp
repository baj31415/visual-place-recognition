#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
// #include <thread>
#include <utility>
#include <vector>

#include "../include/bow.hpp"

using namespace cv;
using namespace std;

Mat ipb::computeHistogram(Mat &descriptors, Mat &dictionary) {

  // for (const auto &descriptor_matrix : descriptors) {
  int k = 1; // Number of nearest neighbors to search for
  int checks =
      32; // Number of checks to perform when searching for nearest neighbors
  flann::SearchParams searchParams(checks);

  // Convert descriptors and dictionary to CV_32F data type
  descriptors.convertTo(descriptors, CV_32F);
  dictionary.convertTo(dictionary, CV_32F);

  // Build index for dictionary using FLANN
  flann::Index index(dictionary, flann::KDTreeIndexParams(),
                     cvflann::FLANN_DIST_L2);

  // Search for nearest neighbor for each descriptor
  Mat indices(descriptors.rows, k, CV_32F);
  Mat distances(descriptors.rows, k, CV_32F);
  int rows_desc = descriptors.rows;

  index.knnSearch(descriptors, indices, distances, k, searchParams);
  // std::cout << "working" << std::endl;
  Mat histogram = Mat::zeros(1, dictionary.rows, CV_32F);
  for (int i = 0; i < indices.rows; i++) {
    histogram.at<float>(indices.at<int>(i))++;

    // the index value/integer at the specific location in the indices matrix is
    // used to increment the specific bin in the histogram
    //
  } // The histogram.at<float>(indices.at<int>(i))++ expression then increments
    // the bin in the histogram matrix that corresponds to that index.
  return histogram; // type cv::Mat with float (CV_32F) elements
}

vector<Mat> ipb::ComputeAllHistograms(vector<Mat> &loaded_bins,
                                      Mat &visual_dictionary) {
  vector<Mat> histograms_all;
  int loaded_bins_size = loaded_bins.size();

  for (int i = 0; i < loaded_bins_size; i++) {
    Mat histogram = ipb::computeHistogram(loaded_bins[i], visual_dictionary);
    // std::cout << "each histogram: " << histogram << std::endl;
    histograms_all.push_back(histogram);
  }

  return histograms_all;
}

void ipb::BowDictionary::build(int max_iter, int size,
                               const std::vector<cv::Mat> &descriptors) {
  dictionary = ipb::kMeans(descriptors, size, max_iter);
}

void ipb::BowDictionary::set_vocabulary(const std::string &filename) {
  dictionary = ipb::serialization::Deserialize(filename);
  // return dictionary;
}

void ipb::BowDictionary::save_vocabulary(const std::string &filename) {
  ipb::serialization::Serialize(dictionary, filename);
}
