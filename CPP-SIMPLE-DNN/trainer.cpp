#include "trainer.hpp"
#include <chrono>

namespace DNN = DeepNeuralNetwork;

DNN::Trainer::Trainer(Network* network, size_t exampleCount, double** ins, double** outs, double evalSplit, bool deleteData)
{
	if (evalSplit < 0 || evalSplit > 1)
	{
		printf("Invalid evalSplit. Must be between 0 and 1.\n");
		std::exit(1);
	}

	this->networkInfo.batchSize = network->batchSize;
	this->networkInfo.inputSize = network->S_0;
	this->networkInfo.outputSize = network->layers[network->L - 1].height;

	this->trainee = network;
	this->trainData.inputs = ins;
	this->trainData.outputs = outs;
	this->trainData.handleDeletion = deleteData;

	this->batch.inputs = new double* [this->networkInfo.batchSize];
	this->batch.outputs = new double* [this->networkInfo.batchSize];

	size_t trainExamples = (size_t)(exampleCount * (1.0 - evalSplit));
	this->availableBatchCount = trainExamples / this->networkInfo.batchSize;
	this->trainingExampleCount = this->availableBatchCount * this->networkInfo.batchSize;
	this->evalExampleCount = exampleCount - this->trainingExampleCount;

	this->shuffledHat = new size_t[this->trainingExampleCount];
	this->currentBatch = std::numeric_limits<size_t>::max();

	if (this->trainingExampleCount <= 0)
	{
		printf("Not enough train data. You can: get more train data, lower the evalSplit, or lower the batch size.\n");
		std::exit(1);
	}

}

void DNN::Trainer::shuffle()
{
	for (size_t i = 0; i < this->trainingExampleCount; i++)
	{
		this->shuffledHat[i] = i;
	}

	for (size_t i = 0, range = this->trainingExampleCount; i < this->trainingExampleCount; i++, range--)
	{
		size_t r = (rand() % range) + i;
		std::swap(this->shuffledHat[i], this->shuffledHat[r]);
	}
}

size_t DNN::Trainer::nextBatch()
{
	if (this->currentBatch >= this->availableBatchCount)
	{
		this->currentBatch = 0;
		this->shuffle();
	}

	const size_t start = this->currentBatch * this->networkInfo.batchSize;
	for (size_t i = 0; i < this->networkInfo.batchSize; i++)
	{
		const size_t& exampleIdx = this->shuffledHat[start + i];
		this->batch.inputs[i] = this->trainData.inputs[exampleIdx];
		this->batch.outputs[i] = this->trainData.outputs[exampleIdx];
	}

	this->currentBatch++;
	return this->availableBatchCount - this->currentBatch;
}

double DNN::Trainer::train(size_t epochs, double lossStop, double lossRollingRatio)
{
	this->shuffle();
	auto before = std::chrono::steady_clock::now();

	double rollingLoss = -1;
	size_t e = 0, step = 0;
	while (e < epochs)
	{
		const size_t batchesRemainingInEpoch = this->nextBatch();

		double loss = this->trainee->train(this->batch.inputs, this->batch.outputs);
		if (rollingLoss < 0)
			rollingLoss = loss;
		else
			rollingLoss = rollingLoss * lossRollingRatio + loss * (1.0 - lossRollingRatio);

		if (rollingLoss < lossStop) break;


		double innerEpochPct = 1.0 - batchesRemainingInEpoch / (double)this->availableBatchCount;
		double pctDone = (e + innerEpochPct) / epochs;
		step++;
		if (batchesRemainingInEpoch == 0) e++;
		printf("[step %d, epoch %d] %.3f%% Done... Loss = %.4f        \r", (int)step, (int)e, pctDone * 100, rollingLoss);
	}
	printf("[step %d, epoch %d] %.3f%% Done... Loss = %.4f        \n", (int)step, (int)e, 100.0, rollingLoss);

	auto after = std::chrono::steady_clock::now();
	return std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count() / 1000000000.0;
}

double DNN::Trainer::evaluate(double (*method)(double* predicted, double* actual, size_t N))
{
	double averageReturn = 0.0;
	double* predictionResult = new double[this->networkInfo.outputSize];
	for (size_t i = 0; i < this->evalExampleCount; i++)
	{
		const auto& inputs = this->trainData.inputs[this->trainingExampleCount + i];
		const auto& correctOutputs = this->trainData.outputs[this->trainingExampleCount + i];
		this->trainee->predict(inputs, predictionResult);

		averageReturn += method(predictionResult, correctOutputs, this->networkInfo.outputSize) / this->evalExampleCount;
	}
	delete[] predictionResult;

	return averageReturn;
}

DNN::Trainer::~Trainer()
{
	delete[] this->batch.inputs;
	delete[] this->batch.outputs;
	delete[] this->shuffledHat;

	if (this->trainData.handleDeletion)
	{
		for (size_t i = 0; i < this->networkInfo.batchSize; i++)
		{
			delete[] this->trainData.inputs[i];
			delete[] this->trainData.outputs[i];
		}
		delete[] this->trainData.inputs;
		delete[] this->trainData.outputs;
	}
}

double DeepNeuralNetwork::EvaluationMethod::categoricalAccuracy(double* predicted, double* actual, size_t N)
{
	size_t correct = 0;
	size_t guessed = 0;
	for (size_t i = 0; i < N; i++)
	{
		if (actual[i] > actual[correct]) correct = i;
		if (predicted[i] > predicted[guessed]) guessed = i;
	}

	return correct == guessed ? 1.0 : 0.0;
}

double DeepNeuralNetwork::EvaluationMethod::meanSquaredError(double* predicted, double* actual, size_t N)
{
	double avg = 0.0;
	for (size_t i = 0; i < N; i++)
	{
		avg += pow(predicted[i] - actual[i], 2.0) / N;
	}

	return N;
}
