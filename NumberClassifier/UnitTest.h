#pragma once

#include <cassert>
#include <functional>

namespace utest
{
	template<typename T>
	inline std::function<bool(T, T)> equ = [](T a , T b) {
		return a == b;
	};

	template<typename T>
	inline std::function<bool(T, T)> less = [](T a, T b) {
		return a < b;
	};

	template<typename T>
	inline std::function<bool(T, T)> greater = [](T a, T b) {
		return a > b;
	};
	
	template<typename T>
	inline std::function<bool(T, T)> nequ = [](T a, T b) {
		return a != b;
	};

	template<typename T>
	inline std::function<bool(T, T)> lessequ = [](T a, T b) {
		return a <= b;
	};

	template<typename T>
	inline std::function<bool(T, T)> greaterequ = [](T a, T b) {
		return a >= b;
	};

	template<typename T, typename F>
	bool Test(F function, T expected, std::function<bool(T,T)> comparison = equ<T>)
	{
		assert(comparison(function(), expected));
		return comparison(function(), expected);
	}
}