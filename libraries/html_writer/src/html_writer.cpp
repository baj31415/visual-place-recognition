#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../include/html_writer.hpp"

using std::string;

void html_writer::OpenDocument() {
  std::cout << "<!DOCTYPE html>" << std::endl;
  std::cout << "<html>" << std::endl;
}

void html_writer::CloseDocument() { std::cout << "</html>" << std::endl; }

void html_writer::AddCSSStyle(const std::string &stylesheet) {
  std::cout << "<head>" << std::endl;
  std::cout << R"(<link rel="stylesheet" type="text/css" href=")" << stylesheet
            << "\" />" << std::endl;
  std::cout << "</head>" << std::endl;
}

void html_writer::AddTitle(const std::string &title) {
  std::cout << "<title>" << title << "</title>" << std::endl;
}

void html_writer::OpenBody() { std::cout << "<body>" << std::endl; }

void html_writer::CloseBody() { std::cout << "</body>" << std::endl; }

void html_writer::OpenRow() {
  std::cout << R"(<div class="row">)" << std::endl;
}

void html_writer::CloseRow() { std::cout << "</div>" << std::endl; }

string get_filename(std::string given_str) {
  string delim1 = "/";
  size_t pos = 0;
  string token1;

  while ((pos = given_str.find(delim1)) != std::string::npos) {
    token1 = given_str.substr(0, pos);
    given_str.erase(0, pos + delim1.length());
  }

  return given_str;
}

string get_extension(std::string given_str) {
  string delim1 = ".";
  size_t pos = 0;
  string token1;

  while ((pos = given_str.find(delim1)) != std::string::npos) {
    token1 = given_str.substr(0, pos);
    given_str.erase(0, pos + delim1.length());
  }
  return given_str;
}

void html_writer::AddImage(const std::string &img_path, float score,
                           bool highlight) {
  static string not_const_imgpath;
  static string filename;
  static string filenumber;
  static string extension;
  not_const_imgpath = img_path;

  filename = get_filename(not_const_imgpath);
  extension = get_extension(filename);

  if ((score >= 0.0 && score <= 1.0) &&
      (extension == "jpg" || extension == "png")) {
    if (highlight) {
      std::cout << R"(<div class="column" )"
                << R"(style="border: 5px solid green;)"
                << "\">" << std::endl;
      std::cout << "<h2>" << filename << "</h2>" << std::endl;
      std::cout << "<img src=\"" << img_path << "\"/>" << std::endl;
      std::cout << "<p>score = " << std::setprecision(2) << std::fixed << score
                << "</p>" << std::endl;
      std::cout << "</div>" << std::endl;

    } else if (!highlight) {
      std::cout << R"(<div class="column")"
                << ">" << std::endl;
      std::cout << "<h2>" << filename << "</h2>" << std::endl;
      std::cout << "<img src=\"" << img_path << "\"/>" << std::endl;
      std::cout << "<p>score = " << std::setprecision(2) << std::fixed << score
                << "</p>" << std::endl;
      std::cout << "</div>" << std::endl;
    } else {
      std::cerr << "[ERROR]" << std::endl;
      EXIT_FAILURE;
    }
  } else {
    std::cerr << "[ERROR]: Wrong Score and File Type!" << std::endl;
    EXIT_FAILURE;
  }
}