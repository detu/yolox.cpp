cmake_minimum_required(VERSION 3.18)
project(yolox_libtorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Point CMake at LibTorch, e.g.
#   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
# or, with a PyTorch pip install:
#   cmake -DCMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path)") ..
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---- torch-free part (COCO json + COCOeval), also used by the unit test
add_library(yolox_coco STATIC src/data/coco_api.cpp)
target_include_directories(yolox_coco PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- main library
add_library(yolox_core STATIC
  src/models/network_blocks.cpp
  src/models/yolo_pafpn.cpp
  src/models/yolo_head.cpp
  src/models/yolox.cpp
  src/utils/boxes.cpp
  src/data/data_augment.cpp
  src/data/coco_dataset.cpp
  src/data/dataloader.cpp
  src/exp/exp.cpp
  src/core/checkpoint.cpp
  src/core/evaluator.cpp
  src/core/trainer.cpp
)
target_include_directories(yolox_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolox_core PUBLIC yolox_coco ${TORCH_LIBRARIES} ${OpenCV_LIBS} Threads::Threads)

foreach(tool train eval demo)
  add_executable(yolox_${tool} tools/${tool}.cpp)
  target_link_libraries(yolox_${tool} PRIVATE yolox_core)
endforeach()

option(YOLOX_BUILD_TESTS "Build unit tests" ON)
if(YOLOX_BUILD_TESTS)
  add_executable(test_coco_eval tests/test_coco_eval.cpp)
  target_link_libraries(test_coco_eval PRIVATE yolox_coco)
endif()

# Windows: copy the LibTorch DLLs next to the executables
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  foreach(tool train eval demo)
    add_custom_command(TARGET yolox_${tool} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:yolox_${tool}>)
  endforeach()
endif()
