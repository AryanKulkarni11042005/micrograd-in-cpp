

#pragma once

#include <iostream>
#include <unordered_set>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <algorithm>
#include <vector>
#include <random>
using f32 = float;



struct Value;
using ValuePtr = std::shared_ptr<Value>;


struct Value {
	f32 data = 0.0f;
	f32 grad = 0.0f;
	std::unordered_set<ValuePtr> _children;  
	std::function<void()> _backward = [] {};
	std::string _op;
	Value()
	{
		std::cout << "Constructor" << "\n";
	}
	Value(f32 data, std::unordered_set<ValuePtr> children = {}, std::string op = "")
		: data(data), _children(std::move(children)), _op(std::move(op)) {
	}
	void printValue()
	{
		std::cout << "Value: " << data << " Gradient: " << grad << "\n"; 
	}
	void build_topo(Value* v, std::vector<Value*>& topo, std::unordered_set<Value*>& visited) {
		if (!visited.contains(v)) {
			visited.insert(v);
			for (const auto& child : v->_children) {
				build_topo(child.get(), topo, visited);
			}
			topo.push_back(v);
		}
	};
	void backward()
	{
		grad = 1.0f;
		std::vector<Value*> topo;
		std::unordered_set<Value*> visited;
		build_topo(this, topo, visited);
		std::reverse(topo.begin(), topo.end());
		for (int i = 0; i < topo.size(); i++)
		{
			topo[i]->_backward();
		}
	}
};

//---------Mathematical Functions starts here-----------
//Tanh Activation Function

// Exponential e^(x)
inline ValuePtr exponential(const ValuePtr& x)
{
	f32 t = std::exp(x->data);
	auto out = std::make_shared<Value>(t, std::unordered_set<ValuePtr>{ x }, "exp");
	Value* o = out.get();
	out->_backward = [x, o, t] {
		x->grad += t * o->grad;
	};
	return out;
}

// Power a^b, used for division too

inline ValuePtr power(const ValuePtr& x, f32 n)
{
	f32 t = std::pow(x->data, n);
	auto out = std::make_shared<Value>(t, std::unordered_set<ValuePtr>{ x }, "power");
	Value* o = out.get();
	out->_backward = [x, o, n] {
		x->grad += n * std::pow(x->data, (n - 1)) * o->grad;
		};
	return out;
}


// Addition 
inline ValuePtr operator+(const ValuePtr& a, const ValuePtr& b)
{
	auto out = std::make_shared<Value>(a->data + b->data,
		std::unordered_set<ValuePtr>{ a, b }, "+");
	Value* o = out.get();
	out->_backward = [a, b, o] {
		a->grad += o->grad;
		b->grad += o->grad;
	};

	return out;
}
// Value of a + some float number will be converted to Value Type
inline ValuePtr operator+(const ValuePtr& a, f32 b)
{
	return a + std::make_shared<Value>(b);
}
// float + Value 
inline ValuePtr operator+(f32 a, const ValuePtr& b)
{
	return std::make_shared<Value>(a) + b;
}
//Multiplication 
inline ValuePtr operator*(const ValuePtr& a, const ValuePtr& b)
{
	auto out = std::make_shared<Value>(a->data * b->data,
		std::unordered_set<ValuePtr>{ a, b }, "*");
	Value* o = out.get();
	out->_backward = [a,b,o] {
		a->grad += b->data * o->grad;
		b->grad += a->data * o->grad;
	};
	return out;
}
// Look at Addition Functions :)
inline ValuePtr operator*(const ValuePtr& a, f32 b)
{
	return a * std::make_shared<Value>(b);
}

inline ValuePtr operator*(f32 a, const ValuePtr& b)
{
	return std::make_shared<Value>(a) * b;
}

// Negation: -a = a * -1
inline ValuePtr operator-(const ValuePtr& a)
{
	return a * -1.0f;
}
// Subtraction: a - b = a + (-b)
inline ValuePtr operator-(const ValuePtr& a, const ValuePtr& b)
{
	return a + (-b);
}
inline ValuePtr operator-(const ValuePtr& a, f32 b)
{
	return a - std::make_shared<Value>(b);
}
// Division: a / b = a * b^-1
inline ValuePtr operator/(const ValuePtr& a, const ValuePtr& b)
{
	return a * power(b, -1.0f);
}
inline ValuePtr activation(const ValuePtr& x)
{
	f32 t = std::tanh(x->data);
	auto out = std::make_shared<Value>(t, std::unordered_set<ValuePtr>{ x }, "tanh");
	Value* o = out.get();
	out->_backward = [x, o, t] {
		x->grad += (1.0f - t * t) * o->grad;
		};
	return out;
}
// NEURON CLASS 
inline f32 random_uniform(f32 lo, f32 hi)
{
	static std::mt19937 gen(std::random_device{}());
	std::uniform_real_distribution<f32> dist(lo, hi);
	return dist(gen);
}
struct Neuron
{
	std::vector<ValuePtr> w;
	ValuePtr b;

	Neuron(size_t nin)
	{
		for (size_t i = 0; i < nin; i++)
		{
			w.push_back(std::make_shared<Value>(random_uniform(-1, 1)));
		}
		b = std::make_shared<Value>(random_uniform(-1, 1));
	}
	ValuePtr operator()(const std::vector<ValuePtr>& x) const
	{
		ValuePtr act = b;
		for (size_t i = 0; i < w.size(); i++)
		{
			act = act + w[i] * x[i];
		}
		return activation(act);
	}
	std::vector<ValuePtr> parameters() const
	{
		std::vector<ValuePtr> params = w;
		params.push_back(b);
		return params;
	}

};


struct Layer
{
	std::vector<Neuron> neurons;
	Layer(size_t nin, size_t nout)
	{
		for (size_t i = 0; i < nout; i++)
		{
			neurons.emplace_back(nin);
		}
	}
	std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& x) const
	{
		std::vector<ValuePtr> outs;
		outs.reserve(neurons.size());
		for (const auto& n : neurons)
		{
			outs.push_back(n(x));
		}
		return outs;
	}
	std::vector<ValuePtr> parameters() const
	{
		std::vector<ValuePtr> params;
		for (const auto& n : neurons)
		{
			auto p = n.parameters();
			params.insert(params.end(), p.begin(), p.end());
		}
		return params;
	}
};

struct MLP
{
	std::vector<Layer> layers;
	
	MLP(size_t nin, const std::vector<size_t>& nouts)
	{
		std::vector<size_t> sz;
		sz.push_back(nin);
		sz.insert(sz.end(), nouts.begin(), nouts.end());
		for (size_t i = 0; i < nouts.size(); i++)
		{
			layers.emplace_back(sz[i], sz[i + 1]);
		}
	}
	std::vector<ValuePtr> operator()(std::vector<ValuePtr> x) const
	{
		for (const auto& layer : layers)
		{
			x = layer(x);
		}
		return x;
	}
	std::vector<ValuePtr> parameters() const
	{
		std::vector<ValuePtr> params;
		for (const auto& layer : layers)
		{
			auto p = layer.parameters();
			params.insert(params.end(), p.begin(), p.end());
		}
		return params;
	}
};