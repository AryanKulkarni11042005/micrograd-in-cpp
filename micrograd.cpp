// KarpathyLecTutorials.cpp : Defines the entry point for the application.
//
#include <iostream>
#include "lib.h"
#include <algorithm>

void printOut(std::vector<ValuePtr> out)
{
	for (size_t i = 0; i < out.size(); i++)
	{
		out[i]->printValue();
		std::cout << "\n";
	}
}
int main()
{
	std::vector<std::vector<ValuePtr>> xs = {
		{
			std::make_shared<Value>(2.0f), std::make_shared<Value>(3.0f),std::make_shared<Value>(-1.0f)
		},
		{	std::make_shared<Value>(3.0f), std::make_shared<Value>(-1.0f), std::make_shared<Value>(0.5f)
		},
		{	std::make_shared<Value>(0.5f), std::make_shared<Value>(1.0f), std::make_shared<Value>(1.0f)
		},
		{
			std::make_shared<Value>(1.0f),std::make_shared<Value>(1.0f),std::make_shared<Value>(-1.0f)
		}
	};
	MLP mlp(3, { 4,4,1 });
	std::vector<ValuePtr> y_preds;
	std::vector<ValuePtr> out;
	f32 lr = 0.05f;
	std::vector<f32> ys = { 1.0f, -1.0f, -1.0f, 1.0f };
	for (int epoch = 0; epoch < 10; epoch++)
	{
		//Calculating Loss using Forward Pass
		ValuePtr loss = std::make_shared<Value>(0.0f);
		for (size_t i = 0; i < xs.size(); i++)
		{
			ValuePtr pred = mlp(xs[i])[0];
			loss = loss + power(pred - ys[i], 2);
		}
		// Zero Grad because we don't want gradients to pile up 
		for (const auto& p : mlp.parameters())
		{
			p->grad = 0.0f;
		}
		// Calculating Gradients in Backward Pass
		loss->backward();

		// Updating Weights using Gradient Descent 
		for (const auto& p : mlp.parameters())
		{
			p->data += -lr * p->grad;
		}
		std::cout << "Epoch " << epoch << "  loss: " << loss->data << "\n";
	}
}
