cmake_minimum_required(VERSION 3.0)
project(opencvfacerectest)

find_package(PkgConfig REQUIRED)
pkg_search_module(OPENCV REQUIRED opencv)
set(EXEC_NAME facerec_video)
set(EXEC_NAME_TRAIN facerec_train)
set(EXEC_NAME_DETECT facerec_detect)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(SOURCE_FILES
        face_rec/face_recognition.cpp
        face_rec/face_recognition.h
        face_rec/face_training.cpp
        face_rec/face_training.h
)

add_executable(${EXEC_NAME} facerec_video.cpp ${SOURCE_FILES})
add_executable(${EXEC_NAME_TRAIN} facerec_train.cpp ${SOURCE_FILES})
add_executable(${EXEC_NAME_DETECT} facerec_detect.cpp ${SOURCE_FILES})

foreach(EXEC ${EXEC_NAME} ${EXEC_NAME_TRAIN} ${EXEC_NAME_DETECT})
        target_link_libraries(${EXEC} ${OPENCV_LIBRARIES})
        target_include_directories(${EXEC} PUBLIC ${OPENCV_INCLUDE_DIRS})
        target_compile_options(${EXEC} PUBLIC ${OPENCV_CFLAGS_OTHER})
endforeach(EXEC)