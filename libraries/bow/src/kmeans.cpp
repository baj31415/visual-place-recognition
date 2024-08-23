#include "../include/bow.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>
using namespace cv;
using namespace std;

cv::Mat ipb::stackMatrices(const std::vector<cv::Mat> &descriptors) {
  cv::Mat vector_concat;
  cv::vconcat(descriptors, vector_concat);
  return vector_concat;
}

cv::Mat ipb::kMeans(const std::vector<cv::Mat> &descriptors, int k,
                    int max_iter) {
  Mat centers;
  Mat labels;
  int numBins = 128;
  cv::Mat v_concat = stackMatrices(descriptors);

  for (int col = 0; col < v_concat.cols; col++) {
    for (int row = 0; row < v_concat.rows; row++) {
      if (isdigit(v_concat.at<float>(row, col)) == false) {
        v_concat.at<float>(row, col) = 0;
      }
    }
  } // if NaN, set to zero

  cv::kmeans(
      v_concat, k, labels,
      TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001),
      max_iter, KMEANS_PP_CENTERS, centers);

  // Create visual dictionary using the center of each cluster
  cv::Mat visual_dict(k, numBins, CV_32F);

  for (int i = 0; i < k; i++) {
    cv::Mat cluster_center = centers.row(i);
    float *visual_word = visual_dict.ptr<float>(i);
    // A loop is executed over all the clusters. For each cluster, the
    // corresponding row from the centers matrix is extracted, and the values
    // are copied to a float pointer called visual_word in the visual_dict
    // matrix.
    for (int j = 0; j < numBins; j++) {
      visual_word[j] = cluster_center.at<float>(j);
    }
  }

  return visual_dict;
}