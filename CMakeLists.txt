# CMakeLists.txt

cmake_minimum_required(VERSION "3.17")
project(C++_DBOD)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIRS})

add_executable(main main.cpp capturing.cpp config.cpp feature_extractor.cpp preprocess.cpp saver.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${YAMLCPP_LIBRARIES} stdc++fs)
