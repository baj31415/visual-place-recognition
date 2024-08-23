#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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

auto return_similarity(cv::Mat data_hist, cv::Mat stacked_query, int i) {
  double dist = ipb::cosine_distance(stacked_query, data_hist);
  std::filesystem::path result_img = "dataset/all_raw_imgs/";
  if (i < 10) {
    std::string path =
        "imageCompressedCam0_00000" + std::to_string(i) + "0.png";
    result_img /= path;
  } else if (i > 9 && i < 100) {
    std::string path = "imageCompressedCam0_0000" + std::to_string(i) + "0.png";
    result_img /= path;
  } else if (i > 99 && i < 1000) {
    std::string path = "imageCompressedCam0_000" + std::to_string(i) + "0.png";
    result_img /= path;
  }
  return make_tuple(result_img, dist);
}

auto copy_file(std::string number, std::filesystem::path source_folder,
               std::filesystem::path target_folder) {
  std::string file_name = "imageCompressedCam0_000" + number + "0.png";
  std::cout << "File Name : " << file_name << std::endl;
  std::filesystem::path source_file = source_folder / file_name;
  std::filesystem::path target_file = target_folder / file_name;

  std::filesystem::copy(source_file, target_file);

  return file_name;
}

int main() {

  std::string query_number;
  std::filesystem::path query_img_path = "../query_image/raw_imgs";
  std::filesystem::path query_bin_path = "../query_image/bin";
  std::filesystem::path img_path = "../dataset/all_raw_imgs";
  std::filesystem::path bin_path =
      "../dataset/complete_bins"; // 10 images only "../dataset/sifts_bin";

  std::cout << "<!--******************************************" << std::endl;
  std::cout << "Welcome to our Bag of Visual Words program " << std::endl;
  std::cout << "******************************************" << std::endl;

  std::cout << " " << std::endl;
  std::cout << "Please provide a query image from the dataset" << std::endl;
  std::cin >> query_number;

  //**************************************************************************************************
  // QUERY IMAGE HISTOGRAM CREATION
  //**************************************************************************************************
  // Delete existing files in query_img_path and query_bin_path
  for (const auto &file : std::filesystem::directory_iterator(query_img_path)) {
    if (std::filesystem::is_regular_file(file)) {
      std::filesystem::remove(file.path());
    }
  }
  for (const auto &file : std::filesystem::directory_iterator(query_bin_path)) {
    if (std::filesystem::is_regular_file(file)) {
      std::filesystem::remove(file.path());
    }
  }

  // copy query image from dataset to query image folder
  std::filesystem::path filename =
      copy_file(query_number, img_path, query_img_path);

  std::cout << "We are starting place recognition now! " << std::endl;
  //**************************************************************************************************
  // DATASET HISTOGRAM CREATION
  //**************************************************************************************************

  //*********************************************************
  //      PART 1 - compute features, load bin files
  //************************************c*********************

  // ipb::serialization::sifts::ConvertDataset(img_path);
  std::vector<Mat> loaded_bins =
      ipb::serialization::sifts::LoadDataset(bin_path);

  //*********************************************************
  //        PART 2 and 3 - Compute Histogram, clusters and dictionary
  //*********************************************************
  // use the pre-computed dictionary created as of March 4, 1 a.m.
  ipb::BowDictionary &dictionary = ipb::BowDictionary::GetInstance();
  // for creation and exporting of dicitonary
  // int max_iter = 500;
  // int K = 70;
  // dictionary.build(max_iter, K, loaded_bins);
  // dictionary.save_vocabulary("dictionary_new.bin");
  // std::cout << "Dictionary exported " << std::endl;

  std::filesystem::path dictionary_path =
      "../dictionary_repo/dictionary_online.bin";
  // std::filesystem::path dictionary_path = "dictionary_new.bin";

  dictionary.set_vocabulary(dictionary_path);

  cv::Mat vocab = dictionary.voc_dictionary();

  vector<Mat> histograms_all = ipb::ComputeAllHistograms(loaded_bins, vocab);

  //   Combine the histograms into one Matrix
  Mat histo_stacked = ipb::stackMatrices(histograms_all);

  //*********************************************************
  //        PART 4 - TF IDF - Algorithm
  //*********************************************************

  cv::Mat reweighted_histo = ipb::TF_IDF(histo_stacked);

  reweighted_histo = histo_stacked;

  //**************************************************************************************************
  // QUERY IMAGE HISTOGRAM CREATION
  //**************************************************************************************************
  // 1 - compute sift features
  ipb::serialization::sifts::ConvertDataset(query_img_path);
  // return 0;

  std::vector<Mat> loaded_bins_query =
      ipb::serialization::sifts::LoadDataset(query_bin_path);

  //  3 - compute histogram
  vector<Mat> histograms_quer_image =
      ipb::ComputeAllHistograms(loaded_bins_query, vocab);

  cv::Mat stacked_query = ipb::stackMatrices(histograms_quer_image);

  //*********************************************************
  //        PART 5 - Cosine Similarity
  //*********************************************************
  // use make_tuple here to return the filepath and cosine distance

  std::string title = "KITTI Dataset";
  std::string stylesheet = "web_app/style.css";
  std::vector<std::tuple<std::filesystem::path, float>> best_matches;
  for (int i = 0; i < reweighted_histo.rows; i++) {
    cv::Mat data_hist = reweighted_histo.row(i);
    auto [path, dist] = return_similarity(data_hist, stacked_query, i);
    // cout << "Cosine Distance for image " << path << " is : " << dist
    //     << std::endl;
    best_matches.emplace_back(path, dist);
  }

  auto cmp = [](const std::tuple<std::filesystem::path, float> &a,
                const std::tuple<std::filesystem::path, float> &b) {
    return std::get<1>(a) > std::get<1>(b);
  };

  // Sort the vector using the custom comparator
  std::sort(best_matches.begin(), best_matches.end(), cmp);

  std::cout << "*****************************************--!>" << std::endl;
  //*********************************************************
  //        PART 6 - Image Viewer Display
  // //*********************************************************
  std::filesystem::path hello = "dataset/all_raw_imgs" / filename;
  const image_browser::ScoredImage query{hello, 1.0};

  const image_browser::ScoredImage img1{std::get<0>(best_matches[0]),
                                        std::get<1>(best_matches[0])};
  const image_browser::ScoredImage img2{std::get<0>(best_matches[1]),
                                        std::get<1>(best_matches[1])};
  const image_browser::ScoredImage img3{std::get<0>(best_matches[2]),
                                        std::get<1>(best_matches[2])};
  const image_browser::ScoredImage img4{std::get<0>(best_matches[3]),
                                        std::get<1>(best_matches[3])};
  const image_browser::ScoredImage img5{std::get<0>(best_matches[4]),
                                        std::get<1>(best_matches[4])};
  const image_browser::ScoredImage img6{std::get<0>(best_matches[5]),
                                        std::get<1>(best_matches[5])};
  const image_browser::ScoredImage img7{std::get<0>(best_matches[6]),
                                        std::get<1>(best_matches[6])};
  const image_browser::ScoredImage img8{std::get<0>(best_matches[7]),
                                        std::get<1>(best_matches[7])};
  const image_browser::ScoredImage img9{std::get<0>(best_matches[8]),
                                        std::get<1>(best_matches[8])};
  const image_browser::ScoredImage img10{std::get<0>(best_matches[9]),
                                         std::get<1>(best_matches[9])};
  const image_browser::ScoredImage img11{std::get<0>(best_matches[10]),
                                         std::get<1>(best_matches[10])};
  // const image_browser::ScoredImage img2{"data/000100.png", 0.96};

  // const image_browser::ImageRow triad_1{img1, img2, img3};
  // const image_browser::ImageRow triad_2{img4, img5, img6};
  // const image_browser::ImageRow triad_3{img7, img8, img9};

  const image_browser::ImageRow triad_1{query, img1, img2};
  const image_browser::ImageRow triad_2{img3, img4, img5};
  const image_browser::ImageRow triad_3{img6, img7, img8};
  const image_browser::ImageRow triad_4{img9, img10, img10};

  std::vector<image_browser::ImageRow> rows = {triad_1, triad_2, triad_3,
                                               triad_4};

  image_browser::CreateImageBrowser(title, stylesheet, rows);

  return 0;
}
