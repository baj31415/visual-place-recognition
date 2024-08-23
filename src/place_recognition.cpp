#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>

#include <cstddef>
#include <cstdio>
#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <regex>
#include <sstream>
#include <string>

#include "../include/place_recognition.hpp"

using namespace std;
using namespace cv;
using namespace flann;

cv::Mat ipb::TF_IDF(cv::Mat &stacked_histogram) {
  int num_docs = stacked_histogram.rows;  // number of images
  int num_terms = stacked_histogram.cols; // number of words

  // std::cout << "num docs: " << num_docs << std::endl;
  // std::cout << "num_terms: " << num_terms << std::endl;

  // term frequency
  std::vector<double> tf(num_docs * num_terms);
  for (int i = 0; i < num_docs; i++) {
    for (int j = 0; j < num_terms; j++) {
      tf[i * num_terms + j] = stacked_histogram.at<float>(i, j);
    } // i*num_terms +j =
  }

  // document frequency df
  std::vector<double> df(num_terms, 0.0);
  for (int j = 0; j < num_terms; j++) {
    for (int i = 0; i < num_docs; i++) {
      if (stacked_histogram.at<float>(i, j) > 0) {
        df[j]++;
      }
    }
  }

  // inverse document frequency (IDF)
  std::vector<double> idf(num_terms);
  for (int j = 0; j < num_terms; j++) {
    if (df[j] > 0) {
      idf[j] = log((double)num_docs / df[j]);
    }
  }

  // tf-idf reweighting
  for (int i = 0; i < num_docs; i++) {
    for (int j = 0; j < num_terms; j++) {
      stacked_histogram.at<float>(i, j) = tf[i * num_terms + j] * idf[j];
    }
  }

  return stacked_histogram;
}

double ipb::cosine_distance(cv::Mat &hist1, cv::Mat &hist2) {
  double dotProduct = hist1.dot(hist2);
  double norm1 = cv::norm(hist1);
  double norm2 = cv::norm(hist2);
  double cosineSimilarity = dotProduct / (norm1 * norm2);
  double cosineDistance = 1 - cosineSimilarity;
  return cosineDistance;
}
