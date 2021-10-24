#include "includes.hpp"
#include "neural_network.hpp"

namespace DNN = DeepNeuralNetwork;

int main()
{
	// create a network with 3 layers:
	//		the input layer of height 2
	//		then a hidden layer with 1 neuron (linear activation, same as `None` activation)
	//		then the output layer, with one neuron
	DNN::Network nn = DNN::Network(
		{
			DNN::LayerDescription(1, Activation::Activator::Linear),
			1 // exactly identical to above layer, linear is default
		},
		2, // number of inputs
		0.001 // learning rate
	);

	// how many training steps
	size_t N = 10000;

	// running average of loss
	double lossRunningAvg = 0;
	for (size_t i = 1; i <= N; i++)
	{
		// this neural network will be adding together its 2 inputs
		// generate a random example
		double inputs[] = {
			randf<-10, 10>(),
			randf<-10, 10>()
		};
		double output = inputs[0] + inputs[1];

		nn.feedForward(inputs);
		double loss = nn.backPropagate(&output);
		lossRunningAvg = (loss * 0.1 + lossRunningAvg * 9.9) / 10.;

		if (i % 1000 == 0)
		{
			std::cout << "step " << i << ": loss = " << lossRunningAvg << std::endl;
		}
	}

	double inputs[] = {
		randf<-10, 10>(),
		randf<-10, 10>()
	};
	nn.feedForward(inputs);
	std::cout << nn.fedInputs[0] << " + " << nn.fedInputs[1] << " = " << nn.outputs[0] << std::endl;

	return 0;
}