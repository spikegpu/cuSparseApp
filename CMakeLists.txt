CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(cuSparseApp)
INCLUDE(cmake/SBELUtils.cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

enable_cuda_support()
MESSAGE(STATUS "Cuda version: ${CUDA_VERSION}")

SET(CUSPARSEAPP_HEADERS
	cusparse/common.h
  )

#SET(CUSPARSEAPP_CUHEADERS
#)

# CUSP dependency. Note that we always include CUSP headers using 
# include <cusp/header.h>.
find_path(CUSP_INCLUDE_DIR 
          NAMES cusp/array1d.h 
          HINTS ${CUDA_TOOLKIT_INCLUDE} 
          DOC "Top-level CUSP directory")
if(CUSP_INCLUDE_DIR)
   message(STATUS "CUSP headers found in ${CUSP_INCLUDE_DIR}/cusp")
else()
   message(FATAL_ERROR "CUSP headers not found. Please set CUSP_INCLUDE_DIR.")
endif()
cuda_include_directories(${CUSP_INCLUDE_DIR})

SOURCE_GROUP("Headers" FILES ${CUSPARSEAPP_HEADERS})
# SOURCE_GROUP("CUDA Headers" FILES ${CUSPARSEAPP_CUHEADERS})
# SOURCE_GROUP("MM_IO" FILES ${MMIO_FILES})

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}
    )

cuda_add_executable(example example.cu ${CUSPARSE_HEADERS})
target_link_libraries(example cusparse cusolver)
