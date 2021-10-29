#pragma once
#include "includes.hpp"
#include "activation.hpp"

namespace DeepNeuralNetwork
{
	struct Neuron
	{
		size_t batchSize;
		size_t prevLayerHeight;

		double* w;
		double b;

		double* o;
		double* a;
		double* dc_do;

		Neuron() = delete;
		inline void init(
			const size_t& batchSize, 
			const size_t& prevLayerHeight,
			double* weights = nullptr,
			double bias = std::numeric_limits<double>::infinity()
		);
		~Neuron();
	};

	struct LayerDescription
	{
		size_t height;
		Activation::Activator activator;

		LayerDescription(const size_t& height, Activation::Activator activator = Activation::Activator::None);
	};

	struct Layer {
		size_t height;
		Activation::Activator activator;
		Neuron* neurons;

		Layer() = delete;
		inline void init(
			const LayerDescription& description,
			const size_t& batchSize,
			const size_t& prevLayerHeight,
			double** weights = nullptr,
			double* biases = nullptr
		);
		~Layer();
	};

	struct Network {
		double learningRate;
		size_t batchSize;
		Layer* layers;

		size_t S_0;
		size_t L;

		Network(
			const size_t& inputSize,
			const std::initializer_list<LayerDescription>& layers,
			const size_t& batchSize,
			const double& learningRate = 0.001, 
			double*** weights = nullptr, 
			double** biases = nullptr
		);

		void summary(bool showParameterValues = true);

		void predict(double* const inputs, double* results);
		double train(double** const inputs, double** const outputs);

		~Network();
	};
}