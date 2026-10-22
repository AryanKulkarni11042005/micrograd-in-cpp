

#pragma once

#include <iostream>
#include <unordered_set>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <algorithm>
#include <vector>
using f32 = float;



struct Value;
using ValuePtr = std::shared_ptr<Value>;


struct Value {
	f32 data;
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

// Division a/b
inline ValuePtr operator/(const ValuePtr& a, const ValuePtr& b)
{
	auto out = std::make_shared<Value>((a * power(b, -1)),
		std::unordered_set<ValuePtr>{ a, b }, "/");
	return out;

}
// Negation Function  
inline ValuePtr negation(const ValuePtr& a)
{
	auto out = std::make_shared<Value>( (-1 * a->data),
		std::unordered_set<ValuePtr>{ a}, "Negation");
	return out;
}
//Subtraction 
inline ValuePtr operator-(const ValuePtr& a, const ValuePtr& b)
{
	auto out = std::make_shared<Value>((a->data + negation(b)),
		std::unordered_set<ValuePtr>{ a }, "Subtraction");
}
// Power a^b, used for division too
inline ValuePtr power(const ValuePtr& x, f32 n)
{
	f32 t = std::pow(x->data, n);
	auto out = std::make_shared<Value>(t, std::unordered_set<ValuePtr>{ x }, "power");
	Value* o = out.get();
	out->_backward = [x, o,n] {
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

