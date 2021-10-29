#include "network.hpp"

namespace DNN = DeepNeuralNetwork;

inline void DNN::Neuron::init(const size_t& batchSize, const size_t& prevLayerHeight, double* weights, double bias)
{
	this->batchSize = batchSize;
	this->prevLayerHeight = prevLayerHeight;

	// initalize bias
	if (bias == std::numeric_limits<double>::infinity())
	{
		this->b = randf<-1, 1>();
	}
	else
	{
		this->b = bias;
	}

	// initalize weights
	this->w = new double[prevLayerHeight];
	for (size_t i = 0; i < prevLayerHeight; ++i)
	{
		this->w[i] = weights ? weights[i] : (randf<-1, 1>() / prevLayerHeight);
	}

	// initalize all the batched stuff
	this->a = new double[batchSize];
	this->o = new double[batchSize];
	this->dc_do = new double[batchSize];
}

DNN::Neuron::~Neuron()
{
	delete[] this->dc_do;
	delete[] this->o;
	delete[] this->a;
	delete[] this->w;
}

DeepNeuralNetwork::LayerDescription::LayerDescription(const size_t& height, Activation::Activator activator)
{
	this->height = height;
	this->activator = activator;
}

inline void DNN::Layer::init(
	const LayerDescription& description,
	const size_t& batchSize,
	const size_t& prevLayerHeight,
	double** weights,
	double* biases
)
{
	this->height = description.height;
	this->activator = description.activator;

	this->neurons = (Neuron*)malloc(this->height * sizeof(Neuron));
	if (!this->neurons) std::exit(1);
	for (size_t i = 0; i < this->height; ++i)
	{
		this->neurons[i].init(
			batchSize,
			prevLayerHeight,
			weights ? weights[i] : nullptr,
			biases ? biases[i] : std::numeric_limits<double>::infinity()
		);
	}
}

DNN::Layer::~Layer()
{
	for (size_t i = 0; i < this->height; i++)
	{
		this->neurons[i].~Neuron();
	}
	free(this->neurons);
}

DNN::Network::Network(
	const size_t& inputSize,
	const std::initializer_list<LayerDescription>& layers,
	const size_t& batchSize,
	const double& learningRate,
	double*** weights,
	double** biases
)
{
	this->S_0 = inputSize;
	this->batchSize = batchSize;
	this->learningRate = learningRate;
	this->L = layers.size();
	this->layers = (Layer*)malloc(this->L * sizeof(Layer));
	if (!this->layers) std::exit(1);

	size_t prevLayerHeight = inputSize;
	size_t i = 0;
	for (const LayerDescription& ld : layers)
	{
		this->layers[i].init(
			ld,
			batchSize,
			prevLayerHeight,
			weights ? weights[i] : nullptr,
			biases ? biases[i] : nullptr
		);

		i++;
		prevLayerHeight = ld.height;
	}
}

void DNN::Network::summary(bool showParameterValues)
{
	printf("================ NETWORK SUMMARY ================\n");
	printf("    Inputs: %5d             Outputs: %5d\n", (int)this->S_0, (int)this->layers[this->L - 1].height);
	printf("    Layers: %5d          Batch Size: %5d\n", (int)this->L, (int)this->batchSize);
	printf("            Learning Rate: %.4g\n", this->learningRate);
	printf("================     LAYERS     =================\n");
	for (size_t n = 0; n < this->L; n++)
	{
		const Layer& layer = this->layers[n];

		printf("%2d  Neurons: %3d,   Activation: %s\n", (int)n, (int)layer.height, Activation::stringifyActivator(layer.activator));
		if (showParameterValues)
		{
			printf("    Biases: [");
			for (size_t i = 0; i < layer.height; i++)
			{
				printf(i == 0 ? "%.3f" : ", %.3f", layer.neurons[i].b);
			}

			const size_t& prevHeight = n == 0 ? this->S_0 : this->layers[n - 1].height;
			printf("]\n    Weights: [\n");
			for (size_t i = 0; i < layer.height; i++)
			{
				printf("        [");
				for (size_t j = 0; j < prevHeight; j++)
				{
					printf(j == 0 ? "%.3f" : ", %.3f", layer.neurons[i].w[j]);
				}
				printf("]\n");
			}
			printf("    ]\n");
		}
	}
	printf("=================================================\n");

	printf("\n\n");
}

void DNN::Network::predict(double* const inputs, double* results)
{
	constexpr size_t batch = 0; // no batched operations, so pretend that this is batch 0

	size_t prevLayerHeight = this->S_0;
	for (size_t n = 0; n < this->L; n++)
	{
		Layer& layer = this->layers[n];

		for (size_t i = 0; i < layer.height; i++)
		{
			Neuron& neuron = layer.neurons[i];
			
			neuron.o[batch] = neuron.b;
			for (size_t j = 0; j < prevLayerHeight; j++)
			{
				if (n == 0)
					neuron.o[batch] += inputs[j] * neuron.w[j];
				else
					neuron.o[batch] += this->layers[n - 1].neurons[j].a[batch] * neuron.w[j];
			}

			neuron.a[batch] = Activation::activate(layer.activator, neuron.o[batch]);
		}

		prevLayerHeight = layer.height;
	}

	// copy activated outputs to results
	const Layer& lastLayer = this->layers[this->L - 1];
	for (size_t i = 0 ; i < lastLayer.height; i++)
	{
		results[i] = lastLayer.neurons[i].a[batch];
	}
}

double DNN::Network::train(double** const inputs, double** const outputs)
{
	// variable names correlate with this document
	// https://github.com/GabeMillikan/CPP-SIMPLE-DNN/blob/main/DNN_Math.pdf
	const auto& T = outputs;

	/*
		Feed Forward
	*/
	for (size_t batch = 0; batch < this->batchSize; batch++)
	{
		size_t prevLayerSize = this->S_0;
		for (size_t n = 0; n < this->L; n++)
		{
			const bool isFirstLayer = n == 0;
			const Layer& layer = this->layers[n];
			for (size_t i = 0; i < layer.height; i++)
			{
				const Neuron& neuron = layer.neurons[i];

				neuron.o[batch] = neuron.b;
				for (size_t j = 0; j < prevLayerSize; j++)
				{
					const double& w = neuron.w[j];
					neuron.o[batch] += w * (isFirstLayer ? inputs[batch][j] : this->layers[n - 1].neurons[j].a[batch]);
				}

				neuron.a[batch] = Activation::activate(layer.activator, neuron.o[batch]);
			}
			prevLayerSize = layer.height;
		}
	}

	/*
		Back Propagate
		(make the average change across batches)
	*/
	double cost = 0.;
	
	// Last layer uses special equation
	const Layer& lastLayer = this->layers[this->L - 1];
	Layer* previousLayer = this->L == 1 ? nullptr : this->layers + (this->L - 2);
	for (size_t i = 0; i < lastLayer.height; i++)
	{
		Neuron& neuron = lastLayer.neurons[i];

		// record dc_do
		double dc_db_avg = 0.;
		for (size_t batch = 0; batch < this->batchSize; batch++)
		{
			const double error = neuron.a[batch] - T[batch][i];
			neuron.dc_do[batch] = 2.0 / lastLayer.height * error * Activation::differentiate(lastLayer.activator, neuron.o[batch]);

			cost += pow(error, 2.0) / this->batchSize;
			dc_db_avg += neuron.dc_do[batch];
		}
		dc_db_avg /= this->batchSize;

		// apply bias update
		neuron.b -= dc_db_avg * this->learningRate;

		// apply weight update
		for (size_t j = 0; j < (previousLayer ? previousLayer->height : this->S_0); j++)
		{
			double dc_dw_avg = 0.;

			if (previousLayer)
			{
				for (size_t batch = 0; batch < this->batchSize; batch++)
				{
					dc_dw_avg += neuron.dc_do[batch] * previousLayer->neurons[j].a[batch];
				}
			}
			else
			{
				for (size_t batch = 0; batch < this->batchSize; batch++)
				{
					dc_dw_avg += neuron.dc_do[batch] * inputs[batch][j];
				}
			}
			dc_dw_avg /= this->batchSize;

			neuron.w[j] -= dc_dw_avg * this->learningRate;
		}
	}
	cost /= lastLayer.height;

	// don't go through other loop if there is no previous layer
	if (this->L == 1) return cost;

	// all other layers
	for (size_t n = this->L - 2; /*will manually break after n == 0*/; n--)
	{
		const Layer& followingLayer = this->layers[n + 1];
		const Layer& layer = this->layers[n];
		previousLayer = n == 0 ? nullptr : this->layers + (n - 1);

		for (size_t i = 0; i < layer.height; i++)
		{
			Neuron& neuron = layer.neurons[i];

			double dc_db_avg = 0.;
			for (size_t batch = 0; batch < this->batchSize; batch++)
			{
				neuron.dc_do[batch] = 0.;
				for (size_t j = 0; j < followingLayer.height; j++)
				{
					neuron.dc_do[batch] += followingLayer.neurons[j].dc_do[batch] * followingLayer.neurons[j].w[i];
				}
				neuron.dc_do[batch] *= Activation::differentiate(layer.activator, neuron.o[batch]);

				dc_db_avg += neuron.dc_do[batch];
			}

			// apply bias update
			neuron.b -= dc_db_avg * this->learningRate;

			// apply weight update
			for (size_t j = 0; j < (previousLayer ? previousLayer->height : this->S_0); j++)
			{
				double dc_dw_avg = 0.;

				if (previousLayer)
				{
					for (size_t batch = 0; batch < this->batchSize; batch++)
					{
						dc_dw_avg += neuron.dc_do[batch] * previousLayer->neurons[j].a[batch];
					}
				}
				else
				{
					for (size_t batch = 0; batch < this->batchSize; batch++)
					{
						dc_dw_avg += neuron.dc_do[batch] * inputs[batch][j];
					}
				}
				dc_dw_avg /= this->batchSize;

				neuron.w[j] -= dc_dw_avg * this->learningRate;
			}
		}

		if (!previousLayer) break;
	}

	return cost;
}

DNN::Network::~Network()
{
	for (size_t i = 0 ; i < this->L; i++)
	{
		this->layers[i].~Layer();
	}
	free(this->layers);
}