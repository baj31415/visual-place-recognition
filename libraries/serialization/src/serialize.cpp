#include <algorithm>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "../include/convert_dataset.hpp"
#include "../include/serialize.hpp"

namespace ipb::serialization {
void Serialize(const cv::Mat &m, const std::string &filename) {
  // std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
  // int type = m.type();
  // int row = m.rows;
  // int columns = m.cols;
  // auto *data = m.data;
  // file.write(reinterpret_cast<char *>(&row), sizeof(row));         // rows
  // file.write(reinterpret_cast<char *>(&columns), sizeof(columns)); // cols
  // file.write(reinterpret_cast<char *>(&type), sizeof(type));       // type
  // file.write(reinterpret_cast<char *>(&data), sizeof(data));       // data

  std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
  int rows = m.rows;
  int cols = m.cols;
  int channels = m.channels();
  int type = m.type();
  int element_size = int(m.elemSize() / channels);
  uchar *data = m.data;
  file.write(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<char *>(&cols), sizeof(cols));
  file.write(reinterpret_cast<char *>(&channels), sizeof(channels));
  file.write(reinterpret_cast<char *>(&type), sizeof(type));
  file.write(reinterpret_cast<char *>(&element_size), sizeof(element_size));
  // std::cout << rows << " " << element_size << " " << channels << " " << type
  //           << std::endl;
  for (size_t i = 0; i < rows * cols * channels * element_size; i++) {
    file.write(reinterpret_cast<char *>(&data[i]), sizeof(data[i]));
  }
}

cv::Mat Deserialize(const std::string &filename) {

  std::ifstream file(filename, std::ios_base::out | std::ios_base::binary);
  if (!file) {
    EXIT_FAILURE;
  }
  int rows;
  int cols;
  int type;
  int channels;
  int element_size;
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  file.read(reinterpret_cast<char *>(&channels), sizeof(channels));
  file.read(reinterpret_cast<char *>(&type), sizeof(type));
  file.read(reinterpret_cast<char *>(&element_size), sizeof(element_size));

  cv::Mat deserialized_mat(rows, cols, type);
  uchar *data = deserialized_mat.data;
  for (size_t i = 0; i < rows * cols * channels * element_size; i++) {
    file.read(reinterpret_cast<char *>(&data[i]), sizeof(data[i]));
  }

  return deserialized_mat;
}
/*This code defines a function named Deserialize that takes a single argument,
filename, which is a string representing the path to a binary file that contains
serialized data for a cv::Mat object.

The purpose of this function is to deserialize the data in the binary file and
return a new cv::Mat object that contains the same data.

The first section of the code creates an input file stream object named file and
opens the binary file specified by filename in binary mode for reading. If the
file fails to open, the program exits with a failure code.

The next section of the code reads the necessary header information from the
binary file, including the number of rows, columns, channels, data type, and
element size. It then creates a new cv::Mat object named deserialized_mat with
the appropriate size and data type.

The final section of the code reads the data from the binary file into the data
pointer of the deserialized_mat matrix using a loop. The loop iterates over each
element of the data and uses file.read() to read the next byte of data into the
data pointer.

Finally, the function returns the deserialized_mat matrix, which now contains
the same data as the binary file.*/

} // namespace ipb::serialization