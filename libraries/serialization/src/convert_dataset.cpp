#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "../include/convert_dataset.hpp"
#include "../include/serialize.hpp"

using namespace std;
using namespace cv;

namespace ipb::serialization::sifts {

void ConvertDataset(const std::filesystem::path &img_path) {

  const std::filesystem::path bin =
      img_path.parent_path().replace_filename("bin/");
  std::filesystem::create_directory(bin);
  const std::string bin_path = bin.string();
  // if (img_path) {
  //     std::filesystem::is_empty(bin_path);
  //     std::cout << "Does not exist" << std::endl;
  //     EXIT_FAILURE;
  // }
  for (const auto &dirEntry : std::filesystem::directory_iterator(img_path)) {
    // const auto &stem = dirEntry.path().stem().string();
    const auto &extension = dirEntry.path().extension();

    if (extension == ".png") {
      // const auto &descriptors_filename = bin_path + stem + ".bin";
      const auto &image_forSift = dirEntry.path().string();
      // std::cout << descriptors_filename << std::endl;
      // std::cout << image_forSift << std::endl;

      auto [desc, keypts] = SIFT_comp(image_forSift);
      std::string converted_filename = dirEntry.path().stem().string() + ".bin";
      filesystem::create_directories("../query_image/bin/");
      ipb::serialization::Serialize(desc,
                                    "../query_image/bin/" + converted_filename);
    }
  }
}

std::tuple<cv::Mat, cv::Mat> SIFT_comp(const std::string &fileName) {
  const cv::Mat kInput = cv::imread(fileName, cv::IMREAD_GRAYSCALE);

  // detect key points
  auto detector = cv::SiftFeatureDetector::create();
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(kInput, keypoints);

  // present the keypoints on the image
  cv::Mat image_with_keypoints;
  drawKeypoints(kInput, keypoints, image_with_keypoints);

  // extract the SIFT descriptors
  cv::Mat descriptors;
  auto extractor = cv::SiftDescriptorExtractor::create();
  extractor->compute(kInput, keypoints, descriptors);

  return std::make_tuple(descriptors, image_with_keypoints);
}
/*his code defines a function named SIFT_comp that takes a single argument,
fileName, which is a string representing the path to an image file.

The purpose of this function is to use the Scale-Invariant Feature Transform
(SIFT) algorithm to extract feature descriptors from the input image and return
those descriptors along with a new image that has the keypoints overlaid on top.

The first line of the function reads the image file into a grayscale cv::Mat
object named kInput using the cv::imread() function.

The next section of the code uses the cv::SiftFeatureDetector::create() function
to create a SIFT detector object and then applies this detector to the kInput
image to detect key points. The key points are stored in a vector named
keypoints.

The drawKeypoints() function is then used to create a new cv::Mat object named
image_with_keypoints that is a copy of the original image but with the detected
keypoints overlaid on top.

The last section of the code uses the cv::SiftDescriptorExtractor::create()
function to create a SIFT descriptor extractor object and then applies this
extractor to the kInput image and the keypoints vector to extract the SIFT
descriptors. The descriptors are stored in a cv::Mat object named descriptors.

Finally, the function returns a tuple containing the descriptors matrix and the
image_with_keypoints matrix.

Overall, this code is a function that uses the SIFT algorithm to extract feature
descriptors from an input image and return those descriptors along with a new
image that has the keypoints overlaid on top.*/
std::vector<cv::Mat> LoadDataset(const std::filesystem::path &bin_path) {
  std::vector<cv::Mat> vector_of_matrices;

  for (const auto &binEntry : std::filesystem::directory_iterator(bin_path)) {
    // std::cout << binEntry << std::endl;
    if (filesystem::path(binEntry).extension() == ".bin") {
      const auto &bin_string = binEntry.path().string();

      // for (const auto &descriptor : binEntry_path) {
      // std::cout << bin_string << std::endl;
      // std::string bin_string{descriptor.string()};

      cv::Mat to_read = ipb::serialization::Deserialize(bin_string);

      // std::cout << to_read.size() << std::endl;

      vector_of_matrices.push_back(to_read);

    }

    else {
      EXIT_FAILURE;
    }
  }
  // std::cout << vector_of_matrices.size() << std::endl;
  return vector_of_matrices;
}
} // namespace ipb::serialization::sifts