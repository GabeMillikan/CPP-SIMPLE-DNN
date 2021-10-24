#include "neural_network.hpp"
namespace DNN = DeepNeuralNetwork;

DNN::Neuron::Neuron(const size_t& inputSize, double* weights, double bias)
{
	this->weights = new double[inputSize];
	this->bias = isinf(bias) ? randf<-1, 1>() : bias;

	for (size_t i = 0; i < inputSize; i++)
		this->weights[i] = weights ? weights[i] : randf<-1, 1>();
}

DNN::Neuron::~Neuron()
{
	delete[] this->weights;
}

DNN::LayerDescription::LayerDescription(size_t size, Activation::Activator activator)
{
	this->size = size;
	this->activator = activator;
}

DNN::Layer::Layer(const LayerDescription& description, size_t inputSize, double** weights, double* biases)
{
	this->size = description.size;
	this->activator = description.activator;

	this->neurons = new Neuron*[this->size];

	for (size_t node = 0; node < this->size; node++)
	{
		this->neurons[node] = new Neuron(
			inputSize,
			weights ? weights[node] : nullptr,
			biases ? biases[node] : std::numeric_limits<double>::infinity()
		);
	}
}

DNN::Layer::~Layer()
{
	for (size_t i = 0; i < this->size; i++)
	{
		delete this->neurons[i];
	}
	delete[] this->neurons;
}

DNN::Network::Network(std::initializer_list<LayerDescription> layers, size_t inputSize, double learningRate, double*** weights, double** biases)
{
	this->learningRate = learningRate;
	this->layerCount = layers.size();
	this->inputSize = inputSize;

	this->shape = new size_t[this->layerCount];
	this->layers = new Layer*[this->layerCount];
	this->fedInputs = new double[inputSize];

	for (size_t layer = 0; layer < this->layerCount; layer++)
	{
		LayerDescription description = *(layers.begin() + layer);

		this->shape[layer] = description.size;
		this->layers[layer] = new Layer(
			description,
			inputSize,
			weights ? weights[layer] : nullptr,
			biases ? biases[layer] : nullptr
		);

		inputSize = description.size;
	}

	this->outputs = new double[this->shape[this->layerCount - 1]];
}

void DNN::Network::feedForward(double* inputs)
{
	for (size_t i = 0; i < this->inputSize; i++)
	{
		this->fedInputs[i] = inputs[i];
	}

	// handle first layer separately, since it's inputs are not activated,
	// unlike the inputs to the following layers, which are the activated
	// outputs of the previous layer
	const Layer* firstLayer = this->layers[0];
	for (size_t neuronIdx = 0; neuronIdx < firstLayer->size; neuronIdx++)
	{
		Neuron* neuron = firstLayer->neurons[neuronIdx];

		neuron->unactivatedOutput = neuron->bias;
		for (size_t prevNeuronIdx = 0; prevNeuronIdx < this->inputSize; prevNeuronIdx++)
		{
			neuron->unactivatedOutput += (neuron->weights[prevNeuronIdx] * inputs[prevNeuronIdx]) / this->inputSize;
		}

		neuron->activatedOutput = Activation::activate(firstLayer->activator, neuron->unactivatedOutput);
	}

	// other layers
	const Layer* previousLayer = firstLayer;
	for (size_t layerIdx = 1; layerIdx < this->layerCount; layerIdx++)
	{
		Layer* layer = this->layers[layerIdx];

		for (size_t neuronIdx = 0; neuronIdx < layer->size; neuronIdx++)
		{
			Neuron* neuron = layer->neurons[neuronIdx];

			neuron->unactivatedOutput = neuron->bias;
			for (size_t prevNeuronIdx = 0; prevNeuronIdx < previousLayer->size; prevNeuronIdx++)
			{
				neuron->unactivatedOutput += neuron->weights[prevNeuronIdx] * previousLayer->neurons[prevNeuronIdx]->activatedOutput / previousLayer->size;
			}


			neuron->activatedOutput = Activation::activate(layer->activator, neuron->unactivatedOutput);
		}

		previousLayer = layer;
	}

	// copy outputs to outputs
	// previousLayer will hold the last layer at this point
	for (size_t outputIdx = 0; outputIdx < previousLayer->size; outputIdx++)
	{
		this->outputs[outputIdx] = previousLayer->neurons[outputIdx]->activatedOutput;
	}
}

double DNN::Network::backPropagate(double* outputs)
{
	/*
	https://cdn.discordapp.com/attachments/387447746791079957/901257561070796811/342191494872039424.png
	LaTeX source of the same thing, in case discord deletes the file for some reason:
	\frac{dC}{do_{Xi}} = 2(a_{Xi} - t_{i})\sigma ^ \prime (o_{Xi}) \\
	\frac{dC}{do_{xi}} = \frac{dC}{do_{(x+1)j}} \cdot w_{(x+1)ji} \cdot \sigma ^\prime (o_{xi})

	first equation:
		C is the cost function, which is squared error.
		x is the layer number (0 is the first, X is the last)
		i is the index of the node in the layer (0 is the top, I is the bottom)
		o is unactivatedOutput (so o_xi is this->layers[x].neurons[i].unactivatedOutput)
		a is the activatedOutput (so a_xi is the same as o_xi but activatedOutput instead)
		t is the target output, so t_i is outputs[i]
		sigmoid() is the activator, it can be replaced with any differentiable activation function like ReLu
		dC/do_Xi is the derivative of cost w.r.t the unactivated output of the very *last layer* (notice capital X)

		this is the c++ impl of the equation, in full:
		lastLayer.neurons[i].dc_do = 2 *
			(lastLayer.neurons[i].activatedOutput - outputs[i]) *
			Activation::differentiate(lastLayer.activator, lastLayer.neurons[i].unactivatedOutput);

	second equation:
		everything is the same, but:
		(x+1) means the following layer
		j represents the index in the following layer
		w is the weight of a certain neuron's input, so w_(x+1)ji corresponds to this->layers[x + 1].neurons[j].weights[i]

		this is the c++ impl of the equation, in full:
		this->layers[x].neurons[i].dc_do = this->layers[x + 1].neurons[j].dc_do *
			this->layers[x + 1].neurons[j].weights[i] *
			Activation::differentiate(this->layers[x].activator, this->layers[x].neurons[i].unactivatedOutput);

		note: you have to sum for all possible j from 0 to J, since each j will have a different and equally important effect on the output.
		note: also, you can factor out the Activation::differentiate and put it outside the loop

	The first equation applies only to the last layer, and the second equation applies only to every other layer.
	This is because the first equation relies on having a target value `t` which is *only* available at the last layer.
	And the second equation relies on the following layer's derivative, which *isn't* available at the last layer.
	*/

	double loss = 0;
	const Layer* lastLayer = this->layers[this->layerCount - 1];
	for (size_t i = 0; i < lastLayer->size; i++)
	{
		Neuron* outputNeuron = lastLayer->neurons[i];
		double difference = outputNeuron->activatedOutput - outputs[i];
		outputNeuron->dc_do = 2. *
			difference *
			Activation::differentiate(lastLayer->activator, outputNeuron->unactivatedOutput);

		loss += difference * difference;
	}
	loss /= lastLayer->size; // *mean* squared error

	// disclude last layer
	const Layer* followingLayer = lastLayer;
	for (size_t layerIdx = this->layerCount - 2; layerIdx < this->layerCount; layerIdx--) // condition is checking for underflow
	{
		const Layer* layer = this->layers[layerIdx];
		for (size_t i = 0; i < layer->size; i++)
		{
			Neuron* neuron = layer->neurons[i];
			neuron->dc_do = 0.;

			for (size_t j = 0; j < followingLayer->size; j++)
			{
				const Neuron* receiverNeuron = followingLayer->neurons[j];
				neuron->dc_do += receiverNeuron->dc_do * receiverNeuron->weights[i];
			}
			neuron->dc_do *= Activation::differentiate(followingLayer->activator, neuron->unactivatedOutput);
			neuron->dc_do /= followingLayer->size; // make it an average, since cost is *mean* squared error
		}
		followingLayer = layer;
	}

	// SGD on first layer
	const Layer* firstLayer = this->layers[0];
	for (size_t neuronIdx = 0; neuronIdx < firstLayer->size; neuronIdx++)
	{
		Neuron* neuron = firstLayer->neurons[neuronIdx];

		// dc/dw_i = dc/do * do/dw_i
		// dc/do = neuron->dc_do
		// o = (constants + a_(x-1)i * w_i + b + constants)
		// do/dw_i = a_(x-1)i
		for (size_t weightIdx = 0; weightIdx < this->inputSize; weightIdx++)
		{
			double activatedInput = this->fedInputs[weightIdx];
			neuron->weights[weightIdx] -= this->learningRate * neuron->dc_do * activatedInput;
		}

		// dc/db = dc/do * do/db
		// dc/do = neuron->dc_do
		// do/db = 1
		neuron->bias -= this->learningRate * neuron->dc_do;
	}

	// apply SGD for everything but the first layer
	for (size_t layerIdx = 1; layerIdx < this->layerCount; layerIdx++)
	{
		const Layer* previousLayer = this->layers[layerIdx - 1];
		const Layer* layer = this->layers[layerIdx];
		for (size_t neuronIdx = 0; neuronIdx < layer->size; neuronIdx++)
		{
			Neuron* neuron = layer->neurons[neuronIdx];
			

			// dc/dw_i = dc/do * do/dw_i
			// dc/do = neuron->dc_do
			// o = (constants + a_(x-1)i * w_i + b + constants)
			// do/dw_i = a_(x-1)i
			for (size_t weightIdx = 0; weightIdx < previousLayer->size; weightIdx++)
			{
				double activatedInput = previousLayer->neurons[weightIdx]->activatedOutput;
				neuron->weights[weightIdx] -= this->learningRate * neuron->dc_do * activatedInput;
			}

			// dc/db = dc/do * do/db
			// dc/do = neuron->dc_do
			// do/db = 1
			neuron->bias -= this->learningRate * neuron->dc_do;
		}
	}


	return loss;
}

DNN::Network::~Network()
{
	for (size_t layer = 0; layer < this->layerCount; layer++)
		delete this->layers[layer];
	delete[] this->layers;

	delete[] this->shape;
	delete[] this->outputs;
	delete[] this->fedInputs;
}