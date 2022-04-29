# CMakeLists.txt

cmake_minimum_required(VERSION "3.17")
project(C++_DBOD)
cmake_policy(VERSION 3.16.3)
set(CMAKE_CXX_STANDARD 14)

# set OpenCV_DIR variable equal to the path to the cmake
# files within the previously installed opencv program
#set(OpenCV_DIR C:/msys64/mingw64/lib/cmake/opencv4)
#set(YAMLCPP C:/msys64/mingw64/lib/cmake/yaml-cpp)


# configure the necessary common CMake environment variables
# needed to include and link the OpenCV program into this
# demo project, namely OpenCV_INCLUDE_DIRS and OpenCV_LIBS
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
##find_package(PkgConfig)
#pkg_check_modules(YAMLCPP REQUIRED yaml-cpp>=0.5)

# tell the build to include the headers from OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} ${YAMLCPP_INCLUDE_DIRS})

# specify the executable target to be built
add_executable(main main.cpp capturing.cpp config.cpp feature_extractor.cpp preprocess.cpp saver.cpp)

# tell it to link the executable target against OpenCV
target_link_libraries(main ${OpenCV_LIBS} ${YAMLCPP_LIBRARIES} stdc++fs)
