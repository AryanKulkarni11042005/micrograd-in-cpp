// KarpathyLecTutorials.cpp : Defines the entry point for the application.
//
#include <iostream>
#include "lib.h"
#include <algorithm>


int main()
{
	//Inputs 
	auto x1 = std::make_shared<Value>(2.0f);
	auto x2 = std::make_shared<Value>(0.0f);
	//Weights
	auto w1 = std::make_shared<Value>(-3.0f);
	auto w2 = std::make_shared<Value>(1.0f);
	auto b = std::make_shared<Value>(6.8813735870195432f);
	auto x1w1 = x1 * w1;
	auto x2w2 = x2 * w2;
	auto x1w1x2w2 = x1w1 + x2w2;
	auto n = x1w1x2w2 + b;
	//After Activation Function 
	auto output = activation(n);
	output->grad = 1.0;
	std::cout << "Topological Sort" << "\n";
	output->backward();
}
