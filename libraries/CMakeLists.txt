find_package(OpenCV 4.6 REQUIRED)
# find_package(Boost 1.76.0 COMPONENTS filesystem)
find_package(Boost 1.74 COMPONENTS filesystem)

if(OpenCV_FOUND)
  message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
  message(STATUS "OpenCV directories: ${OpenCV_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "OpenCV not found, please read the README.md")
endif(OpenCV_FOUND)
find_package(TBB REQUIRED tbb)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS}) 
add_library(html_writer html_writer/src/html_writer.cpp html_writer/include/html_writer.hpp
)

target_link_libraries(html_writer ${Boost_LIBRARIES})
target_link_libraries(html_writer fmt)
target_include_directories(html_writer PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/html_writer/include>)

add_library(image_browser image_browser/src/image_browser.cpp
    image_browser/include/image_browser.hpp
)
target_link_libraries(image_browser html_writer)
target_include_directories(image_browser PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/image_browser/include>)


add_library(serialize serialization/src/serialize.cpp serialization/include/serialize.hpp
)

add_library(convert_dataset serialization/src/convert_dataset.cpp serialization/include/convert_dataset.hpp
)

target_include_directories(convert_dataset PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/serialization/include>)
target_include_directories(serialize PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/serialization/include>)

target_link_libraries(convert_dataset serialize)

add_library(bow bow/src/bow.cpp bow/include/bow.hpp)

target_link_libraries(bow convert_dataset)
target_link_libraries(bow serialize)
target_link_libraries(bow ${OpenCV_LIBS})
target_link_libraries(bow tbb)

target_include_directories(bow PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bow/include>)


add_library(kmeans bow/src/kmeans.cpp bow/include/kmeans.hpp)

target_link_libraries(kmeans convert_dataset)
target_link_libraries(kmeans serialize)
target_link_libraries(kmeans ${OpenCV_LIBS})
target_link_libraries(kmeans tbb)

target_include_directories(kmeans PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bow/include>)
