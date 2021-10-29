#pragma once
#include <fstream>

namespace MNIST {
	constexpr size_t IMAGE_SIZE = 28 * 28;
	constexpr size_t LABEL_COUNT = 10; // obviously
	constexpr size_t EXAMPLE_COUNT = 70000;
	constexpr const char* IMAGES_FILE_PATH = "mnist.images";
	constexpr const char* LABELS_FILE_PATH = "mnist.labels";

	extern double** images;
	extern double** labels;

	bool load();
	void unload();
}