#pragma once

#include "Utility.h"

#define COST(data) net::cstf::CrossEntropy(data)
#define COST_DERIVATIVE(pred, expe) net::cstf::CrossEntropy_derivative(pred, expe)

namespace net
{
	namespace cstf
	{
		template<typename T>
		inline T MSE(T pred, T expe)
		{
			return (pred - expe) * (pred - expe);
		}

		template<typename T>
		inline T MSE(util::DataPoint<T> data)
		{
			T res = 0;
			for (int i = 0; i < data.expected.GetSize(); i++)
			{
				res += MSE(data.output[i], data.expected[i]);
			}
			return res;
		}

		template<typename T>
		inline T MSE(std::vector<util::DataPoint<T>> data)
		{
			T res = (T)0;
			for (util::DataPoint<T> dp : data)
			{
				res += MSE(dp);
			}
			return res / data.size();
		}

		template<typename T>
		inline T MSE(std::vector<std::vector<util::DataPoint<T>>> data)
		{
			T res = (T)0;
			for (std::vector<util::DataPoint<T>> batch : data)
			{
				for (util::DataPoint<T> dp : batch)
				{
					res += MSE(dp);
				}
			}
			return res / (data.size() * data[0].size());
		}

		template<typename T>
		inline T MSE_derivative(T pred, T expe)
		{
			return (T)2.0 * (pred - expe);
		}

		// -----------------------------------------------------------------------------

		template<typename T>
		inline T CrossEntropy(T pred, T expe)
		{
			T v = expe == 1.0 ? -std::log(pred) : -std::log(1.0 - pred);
			return std::isnan(v) ? 0.0 : v;
		}
		
		template<typename T>
		inline T CrossEntropy(util::DataPoint<T> data)
		{
			T cost = 0.0;
			for (int i = 0; i < data.expected.GetSize(); i++)
			{
				cost += CrossEntropy(data.output[i], data.expected[i]);
			}
			return cost;
		}

		template<typename T>
		inline T CrossEntropy(std::vector<util::DataPoint<T>> data)
		{
			T cost = 0.0;
			for (util::DataPoint<T> dp : data)
			{
				cost += CrossEntropy(dp);
			}
			return cost / data.size();
		}

		template<typename T>
		inline T CrossEntropy(std::vector<std::vector<util::DataPoint<T>>> data)
		{
			T res = (T)0;
			for (std::vector<util::DataPoint<T>> batch : data)
			{
				for (util::DataPoint<T> dp : batch)
				{
					res += CrossEntropy(dp);
				}
			}
			return res / (data.size() * data[0].size());
		}

		template<typename T>
		inline T CrossEntropy_derivative(T pred, T expe)
		{
			if (pred == 0.0 || pred == 1.0)
			{
				return 0.0;
			}
			return (-pred + expe) / (pred * (pred - 1));
		}

		// ---------------------------------------------------------------------------------------

		template<typename T>
		inline double Accuracy(std::vector<util::DataPoint<T>> data)
		{
			int correct = 0;
			for (util::DataPoint<T> dp : data)
			{
				int chosen = (int)(std::max_element(dp.output.GetValues().begin(), dp.output.GetValues().end()) - dp.output.GetValues().begin());
				if (chosen == (int)dp.label)
				{
					correct++;
				}
			}
			return (double)correct / data.size();
		}
	}
}