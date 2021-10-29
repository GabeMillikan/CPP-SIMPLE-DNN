#pragma once
#include "network.hpp"

namespace DeepNeuralNetwork {
	namespace EvaluationMethod {
		double categoricalAccuracy(double* predicted, double* actual, size_t N);
		double meanSquaredError(double* predicted, double* actual, size_t N);
	}

	struct Trainer {
		Network* trainee;
		
		struct {
			size_t batchSize;
			size_t inputSize;
			size_t outputSize;
		} networkInfo;
		
		struct {
			double** inputs;
			double** outputs;

			bool handleDeletion;
		} trainData;

		size_t availableBatchCount;
		size_t trainingExampleCount;
		size_t evalExampleCount;

		size_t currentBatch;
		size_t* shuffledHat;
		void shuffle();

		struct {
			double** inputs;
			double** outputs;
		} batch;

		Trainer(Network* network, size_t exampleCount, double** ins, double** outs, double evalSplit = 0.1, bool deleteData = false);

		size_t nextBatch();
		double train(size_t epochs = 1, double lossStop = -1, double lossRollingRatio = 0.9);
		double evaluate(double (*method)(double* predicted, double* actual, size_t N) = EvaluationMethod::meanSquaredError);

		~Trainer();
	};
}