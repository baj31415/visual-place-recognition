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

// vector<ipb::Histogram> ComputeAll_IPBHistograms(vector<Mat> &loaded_bins,
//                                                 Mat &vocabulary) {
//   vector<ipb::Histogram> histograms_all_ipb;
//   int loaded_bins_size = loaded_bins.size();

//   for (int i = 0; i < loaded_bins_size; i++) {
//     ipb::Histogram histogram = ipb::Histogram(loaded_bins[0], vocabulary);
//     // std::cout << "each histogram: " << histogram << std::endl;
//     histograms_all_ipb.push_back(histogram);
//   }

//   return histograms_all_ipb;
// }
// cv::Mat normalize_histogram(cv::Mat &histogram) {
//   cv::Mat normalized_hist;

//   // Compute the sum of all histogram bins
//   float sum = cv::sum(histogram)[0];

//   // If the sum is zero, return the original histogram
//   if (sum == 0.0) {
//     normalized_hist = histogram.clone();
//   } else {
//     // Normalize the histogram by dividing each bin by the sum
//     normalized_hist = histogram / sum;
//   }

//   return normalized_hist;
// }

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
  std::cout << "Your options : " << std::endl;
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
  copy_file(query_number, img_path, query_img_path);

  std::cout << "We are starting place recognition now! -->" << std::endl;
  //**************************************************************************************************
  // DATASET HISTOGRAM CREATION
  //**************************************************************************************************

  //*********************************************************
  //      PART 1 - compute features, load bin files
  //************************************c*********************

  ipb::serialization::sifts::ConvertDataset(img_path);
  std::vector<Mat> loaded_bins =
      ipb::serialization::sifts::LoadDataset(bin_path);

  //*********************************************************
  //        PART 2 - clusters and dictionary
  //*********************************************************

  int max_iter = 500;
  int K = 30;

  Mat visual_dictionary = ipb::kMeans(loaded_bins, K, max_iter);

  // std::cout << "visual dictionary directly from kmeans: "
  // <<visual_dictionary<< std::endl; std::cout << "visual dictionary size: " <<
  // visual_dictionary.size<< std::endl;

  // Computation of dictionary with the BoWDictionary class created in homework
  // 7
  // ipb::BowDictionary &dictionary = ipb::BowDictionary::GetInstance();

  // dictionary.save_vocabulary("dictionary.bin"); // works
  // Mat vocabulary = dictionary.voc_dictionary();

  //*********************************************************
  //        PART 3 - Compute Histogram
  //*********************************************************
  // use the pre-computed dictionary created as of March 4, 1 a.m.
  ipb::BowDictionary &dictionary = ipb::BowDictionary::GetInstance();
  /*for creation and exporting of dicitonary*/
  dictionary.build(max_iter, K, loaded_bins);
  dictionary.save_vocabulary("dictionary_5000.bin");
  std::cout << "Dictionary exported " << std::endl;

  std::filesystem::path dictionary_path =
      "../dictionary_repo/dictionary_5000.bin";
  // std::filesystem::path dictionary_path = "dictionary_new.bin";

  dictionary.set_vocabulary(dictionary_path);

  cv::Mat vocab = dictionary.voc_dictionary();
  // std::cout << "voc dictionary: " << vocab << std::endl;

  // ipb::Histogram histo = ipb::Histogram(loaded_bins[0], vocab);
  // std::cout << "Histogram from histogram class in ipb: " << histo <<
  // std::endl;

  //   /* COMPUTE ALL OF THE HISTOGRAMS FOR THE RAW_IMGS DATASET. PLACE THE
  //   IN A
  //    * VECTOR, THEN CONCATENATE THEM*/
  // ipb::histo starts here
  //  std::vector<ipb::Histogram> histograms_all_IPB =
  //      ComputeAll_IPBHistograms(loaded_bins, vocab);
  //  std::cout << histograms_all_IPB.size() << std::endl;

  // vector<cv::Mat> mat_histograms_from_IPBhisto;
  // for (const auto &histogram : histograms_all_IPB) {
  //   mat_histograms_from_IPBhisto.push_back(
  //       cv::Mat(1, histogram.size(), CV_32F, histogram.data().data()));
  // }

  // cv::Mat concatenated_mat =
  // ipb::stackMatrices(mat_histograms_from_IPBhisto);

  // std::cout << "concatenated mat: " << concatenated_mat << std::endl;
  // ipb histo ends here!
  //    // ipb::Histogram for now bc it shows a dimension of only 1 row
  //    vector<float>
  //    // concatenated_data(concatenated_mat.rows); for (int i = 0; i <
  //    // concatenated_mat.rows; i++) {
  //    //   concatenated_data[i] = concatenated_mat.at<float>(i);
  //    // }
  //    // ipb::Histogram concatenated_histogram(concatenated_data);

  // std::cout << "IPB Histo_all stacked: " << concatenated_histogram <<
  // std::endl; std::cout << "Size: " << concatenated_histogram.size() <<
  // std::endl;

  //   /*same process and output, different method. Without using
  //   ipb::Histogram*/
  vector<Mat> histograms_all = ipb::ComputeAllHistograms(loaded_bins, vocab);
  // std::cout << "all histo size " << histograms_all[0] << std::endl;

  //   Combine the histograms into one Matrix
  Mat histo_stacked = ipb::stackMatrices(histograms_all);

  // std::cout << "[Stacked histogram of all images in the dataset:] "
  //<< histo_stacked << std::endl;
  // std::cout << "Dimensions: " << histo_stacked.size << std::endl;

  //   // with IPB Histogram, use cv::Mat concatenated_mat
  //   // with the native compute histogram function, use cv::Mat histo_stacked

  //*********************************************************
  //        PART 4 - TF IDF - Algorithm
  //*********************************************************

  cv::Mat reweighted_histo = ipb::TF_IDF(histo_stacked);
  // std::cout << "[Reweighted histogram after tf idf] :" << reweighted_histo
  //          << std::endl;

  //**************************************************************************************************
  // QUERY IMAGE HISTOGRAM CREATION
  //**************************************************************************************************
  // 1 - compute sift features
  ipb::serialization::sifts::ConvertDataset(query_img_path);
  // return 0;

  std::vector<Mat> loaded_bins_query =
      ipb::serialization::sifts::LoadDataset(query_bin_path);

  // 2 - find clusters
  // Mat clusters_query = ipb::kMeans(loaded_bins_query, K, max_iter); //don't
  // think we need this bc we already have vocab the exported dictionary.bin
  // file 3 - compute histogram
  vector<Mat> histograms_quer_image =
      ipb::ComputeAllHistograms(loaded_bins_query, vocab);

  cv::Mat stacked_query = ipb::stackMatrices(histograms_quer_image);

  //   //*********************************************************
  //   //        PART 5 - Cosine Similarity
  //   //*********************************************************
  // use make_tuple here to return the filepath and cosine distance

  std::string title = "KITTI Dataset";
  std::string stylesheet = "web_app/style.css";
  std::vector<std::tuple<std::filesystem::path, float>> best_matches;
  for (int i = 0; i < reweighted_histo.rows; i++) {
    cv::Mat data_hist = reweighted_histo.row(i);
    auto [path, dist] = return_similarity(data_hist, stacked_query, i);
    // cout << "Cosine Distance for image " << path << " is : " << dist
    //     << std::endl;
    if (dist > 0.98) {
      best_matches.emplace_back(path, dist);
    }
  }

  //*********************************************************
  //        PART 6 - Image Viewer Display
  // //*********************************************************
  std::vector<std::string> imgVar = {"img1", "img2", "img3", "img4", "img5",
                                     "img6", "img7", "img8", "img9"};

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
  // const image_browser::ScoredImage img2{"data/000100.png", 0.96};

  const image_browser::ImageRow triad_1{img1, img2, img3};
  const image_browser::ImageRow triad_2{img4, img5, img6};
  const image_browser::ImageRow triad_3{img7, img8, img9};

  std::vector<image_browser::ImageRow> rows = {triad_1, triad_2, triad_3};

  image_browser::CreateImageBrowser(title, stylesheet, rows);

  return 0;
}
