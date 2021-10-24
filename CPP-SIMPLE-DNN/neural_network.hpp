#pragma once
#include "includes.hpp"
#include "activation.hpp"

namespace DeepNeuralNetwork
{
	struct Neuron
	{
		double* weights = nullptr;
		double bias = 0.;
		double unactivatedOutput = 0.;
		double activatedOutput = 0.;
		double dc_do = 0.; // derivative of cost with respect to unactivatedOutput

		Neuron(const size_t& inputSize, double* weights = nullptr, double bias = std::numeric_limits<double>::infinity());
		~Neuron();
	};

	struct LayerDescription
	{
		size_t size;
		Activation::Activator activator;

		LayerDescription(size_t size, Activation::Activator activator = Activation::Activator::None);
	};

	struct Layer {
		size_t size = 0;
		Activation::Activator activator = Activation::Activator::Linear;

		Neuron** neurons;

		Layer(const LayerDescription& description, size_t inputSize, double** weights = nullptr, double* biases = nullptr);

		~Layer();
	};

	struct Network {
		double learningRate = 0.001;
		size_t layerCount = 0;
		size_t inputSize = 0;
		size_t* shape = nullptr;
		Layer** layers = nullptr;
		double* outputs = nullptr;
		double* fedInputs = nullptr;

		Network(std::initializer_list<LayerDescription> layers, size_t inputSize, double learningRate = 0.001, double*** weights = nullptr, double** biases = nullptr);

		void feedForward(double* inputs);
		double backPropagate(double* outputs);

		~Network();
	};
}