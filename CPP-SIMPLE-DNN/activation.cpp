#include "activation.hpp"

const char* Activation::stringifyActivator(const Activator& activator)
{
	switch (activator)
	{
	default:
	case Activator::Linear:
		return "Linear";
	case Activator::ReLu:
		return "ReLu";
	case Activator::Sigmoid:
		return "Sigmoid";
	}
}

double Activation::activate(const Activator& activator, const double& input)
{
	switch (activator)
	{
	default:
	case Activator::Linear:
		return input;
	case Activator::ReLu:
		return input <= 0 ? 0 : input;
	case Activator::Sigmoid:
		return 1 / (1 + exp(-input));
	}
}

double Activation::differentiate(const Activator& activator, const double& input)
{
	switch (activator)
	{
	default:
	case Activator::Linear:
		return 1;
	case Activator::ReLu:
		return input <= 0 ? 0 : 1;
	case Activator::Sigmoid:
		double sigmoid = activate(activator, input);
		return sigmoid * (1 - sigmoid);
	}
}
