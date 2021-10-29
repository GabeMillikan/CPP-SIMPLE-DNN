#include "mnist.hpp"

double** MNIST::images = nullptr;
double** MNIST::labels = nullptr;

bool MNIST::load()
{
	std::ifstream images_file(IMAGES_FILE_PATH, std::ios::binary);
	if (!images_file.is_open()) return false;

	std::ifstream labels_file(LABELS_FILE_PATH, std::ios::binary);
	if (!labels_file.is_open()) { labels_file.close(); return false; }

	images = new double*[EXAMPLE_COUNT];
	labels = new double*[EXAMPLE_COUNT];
	for (size_t i = 0; i < EXAMPLE_COUNT; i++)
	{
		images[i] = new double[IMAGE_SIZE];
		labels[i] = new double[LABEL_COUNT];

		for (size_t j = 0; j < LABEL_COUNT; j++) labels[i][j] = 0.0;
	}

	unsigned char image_buffer[IMAGE_SIZE] = {};
	unsigned char label_buffer = 0;
	for (size_t i = 0; i < EXAMPLE_COUNT; i++)
	{
		images_file.read((char*)image_buffer, IMAGE_SIZE);
		labels_file.read((char*)&label_buffer, 1);

		labels[i][label_buffer] = 1.f;
		for (size_t j = 0; j < IMAGE_SIZE; j++)
		{
			images[i][j] = image_buffer[j] / 255.0;
		}
	}

	images_file.close();
	labels_file.close();
	return true;
}

void MNIST::unload()
{
	for (size_t i = 0; i < EXAMPLE_COUNT; i++)
	{
		delete[] labels[i];
		delete[] images[i];
	}
	delete[] labels;
	delete[] images;
}