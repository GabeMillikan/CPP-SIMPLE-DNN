#pragma once
#include "includes.hpp"

namespace Activation
{
	enum class Activator
	{
		None = 0,
		Linear = 0,
		ReLu,
		Sigmoid
	};

	double activate(const Activator& activator, const double& input);
	double differentiate(const Activator& activator, const double& input);
}