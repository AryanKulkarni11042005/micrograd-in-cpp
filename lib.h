

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
			for (auto child : v->_children) {
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

