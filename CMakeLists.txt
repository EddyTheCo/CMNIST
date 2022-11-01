cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(CMNIST_dataset VERSION 0.1 DESCRIPTION "MNIST with 196 pixels torch dataset" LANGUAGES CXX)

find_package(Torch REQUIRED)

add_library(CMNIST cmnist.cpp include/custom-datasets/cmnist.hpp)

target_link_libraries(CMNIST PUBLIC  "${TORCH_LIBRARIES}")

target_include_directories(CMNIST PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

install(TARGETS CMNIST EXPORT CMNISTTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT CMNISTTargets
	FILE CMNISTTargets.cmake
	DESTINATION lib/cmake/CMNIST
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/CMNISTConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/CMNIST"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/CMNISTConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/CMNISTConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/CMNISTConfigVersion.cmake
	DESTINATION lib/cmake/CMNIST
	)
export(EXPORT CMNISTTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/CMNISTTargets.cmake"
	)
