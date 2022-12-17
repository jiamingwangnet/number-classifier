#pragma once

#define NOMINMAX

#include "Utility.h"
#include <cmath>
#include <numeric>

namespace net
{
	namespace actf
	{
		enum class ACTIVATION_TYPE
		{
			SIGMOID,
			RELU,
			SOFTMAX
		};

		template<typename T>
		inline util::Matrix<T> Sigmoid(util::Matrix<T> nodes)
		{
			util::Matrix<T> res{ {}, nodes.GetRows(), nodes.GetColumns() };
			for (int i = 0; i < nodes.GetSize(); i++)
			{
				res[i] = 1.0 / (1.0 + std::exp(-nodes[i]));
			}
			return res;
		}

		template<typename T>
		inline util::Matrix<T> Sigmoid_derivative(util::Matrix<T> nodes)
		{
			util::Matrix<T> res{ {}, nodes.GetRows(), nodes.GetColumns() };
			for (int i = 0; i < nodes.GetSize(); i++)
			{
				T activation = 1.0 / (1.0 + std::exp(-nodes[i]));
				res[i] = activation * (1.0 - activation);
			}
			return res;
		}

		template<typename T>
		inline util::Matrix<T> ReLU(util::Matrix<T> nodes)
		{
			util::Matrix<T> res{ {}, nodes.GetRows(), nodes.GetColumns() };
			for (int i = 0; i < nodes.GetSize(); i++)
			{
				res[i] = std::max(0.0, nodes[i]);
			}
			return res;
		}

		template<typename T>
		inline util::Matrix<T> ReLU_derivative(util::Matrix<T> nodes)
		{
			util::Matrix<T> res{ {}, nodes.GetRows(), nodes.GetColumns() };
			for (int i = 0; i < nodes.GetSize(); i++)
			{
				res[i] = nodes[i] <= 0.0 ? 0.0 : 1.0;
			}
			return res;
		}
		
		template<typename T>
		inline util::Matrix<T> Softmax(util::Matrix<T> nodes)
		{
			util::Matrix<T> res{ {}, nodes.GetRows(), nodes.GetColumns() };
			T expSum = 0.0;
			for (T& v : nodes.GetValues())
			{
				expSum += std::exp(v);
			}
			for (int i = 0; i < nodes.GetSize(); i++)
			{
				res[i] = std::exp(nodes[i]) / expSum;
			}
			return res;
		}

		template<typename T>
		inline util::Matrix<T> Softmax_derivative(util::Matrix<T> nodes)
		{
			util::Matrix<T> res{ {}, nodes.GetRows(), nodes.GetColumns() };

			T sum = 0.0;
			for (T& v : nodes.GetValues())
			{
				sum += std::exp(v);
			}

			for (int i = 0; i < nodes.GetSize(); i++)
			{
				T ex = std::exp(nodes[i]);

				res[i] = (ex * sum - ex * ex) / (sum * sum);
			}
			return res;
		}

		// -----------------------------------------------------------------------------------------------------------

		template<typename T>
		inline util::Matrix<T> Activation(ACTIVATION_TYPE type, util::Matrix<T> nodes)
		{
			switch (type)
			{
			case net::actf::ACTIVATION_TYPE::SIGMOID:
				return Sigmoid(nodes);
				break;
			case net::actf::ACTIVATION_TYPE::RELU:
				return ReLU(nodes);
				break;
			case net::actf::ACTIVATION_TYPE::SOFTMAX:
				return Softmax(nodes); // softmax must only be used on the output layer
				break;
			default:
				return { {},0,0 };
				break;
			}
		}

		template<typename T>
		inline util::Matrix<T> Activation_derivative(ACTIVATION_TYPE type, util::Matrix<T> nodes)
		{
			switch (type)
			{
			case net::actf::ACTIVATION_TYPE::SIGMOID:
				return Sigmoid_derivative(nodes);
				break;
			case net::actf::ACTIVATION_TYPE::RELU:
				return ReLU_derivative(nodes);
				break;
			case net::actf::ACTIVATION_TYPE::SOFTMAX:
				return Softmax_derivative(nodes);
				break;
			default:
				return { {},0,0 };
				break;
			}
		}
	}
}