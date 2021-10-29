#include "includes.hpp"
#include "network.hpp"
#include "trainer.hpp"
#include "mnist.hpp"
namespace DNN = DeepNeuralNetwork;

int main()
{
	if (!MNIST::load()) std::exit(1);

	DNN::Network net(
		MNIST::IMAGE_SIZE,
		{
			{64, Activation::Activator::ReLu},
			MNIST::LABEL_COUNT
		}, 
		64,
		0.002
	);

	DNN::Trainer trainer(&net, MNIST::EXAMPLE_COUNT, MNIST::images, MNIST::labels);

	while (true)
	{
		double t = trainer.train(6);
		printf("Trained for %.2f seconds\n", t);

		double accuracy = trainer.evaluate(DNN::EvaluationMethod::categoricalAccuracy);
		printf("Current accuracy: %.2f%%\n", accuracy * 100.0);

		net.learningRate *= 0.75;
	}

	MNIST::unload();

	/*
	size_t batchSize = 10;
	size_t ioSize = 100;
	DNN::Network net({ ioSize }, ioSize, batchSize, 5);

	double** batchedInputs = new double* [batchSize];
	double** batchedOutputs = new double* [batchSize];

	for (size_t i = 0; i < batchSize; i++)
	{
		batchedInputs[i] = new double[ioSize];
		batchedOutputs[i] = new double[ioSize];
	}

	double runningLoss = 100.0;
	size_t i = 0;
	while (runningLoss >= 0.0001)
	{
		for (size_t j = 0; j < batchSize; j++)
		{
			for (size_t k = 0; k < ioSize; k++)
			{
				batchedInputs[j][k] = batchedOutputs[j][k] = randf<-1, 1>();
			}
		}

		double loss = net.train(batchedInputs, batchedOutputs);
		runningLoss = (runningLoss * 99 + loss) / 100.;
		printf("Loss: %.8f    \r", runningLoss);
		i++;
	}
	printf("\nThat took %d iterations.\n", i);
	*/
}