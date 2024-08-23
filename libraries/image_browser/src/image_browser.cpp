#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../include/image_browser.hpp"

using namespace std;
using std::stringstream;

using ScoredImage = std::tuple<std::string, float>;
using ImageRow = std::array<ScoredImage, 3>;

void image_browser::AddFullRow(const ImageRow &row, bool first_row) {
  html_writer::OpenRow();
  bool highlight = first_row;

  for (const auto &[img_path, score] : row) {
    html_writer::AddImage(img_path, score, highlight);
    highlight = false;
  }

  html_writer::CloseRow();
}
void image_browser::CreateImageBrowser(const std::string &title,
                                       const std::string &stylesheet,
                                       const std::vector<ImageRow> &rows) {
  html_writer::OpenDocument();
  html_writer::AddTitle(title);
  html_writer::AddCSSStyle(stylesheet);
  html_writer::OpenBody();

  for (int iteration = 0; iteration <= 2; iteration++) {
    if (iteration == 0) {
      image_browser::AddFullRow(rows[iteration], true);
    } else {
      image_browser::AddFullRow(rows[iteration]);
    }
  }

  html_writer::CloseBody();
  html_writer::CloseDocument();
}
