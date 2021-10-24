#pragma once
#include <random>

template <int min, int max>
double randf()
{
    static std::default_random_engine engine;
    static std::uniform_real_distribution<> distribution(min, max);
    return distribution(engine);
}