#pragma once

#include "Activation.h"
#include "Utility.h"
#include "Cost.h"

namespace net
{
	class Layer
	{
	public:
		Layer(Layer* in, util::Matrix<double> biases, int n_nodes, actf::ACTIVATION_TYPE activation, double wmin = -1.0, double wmax = 1.0)
			:
			in(in), n_nodes(n_nodes), weights({}, in->n_nodes, n_nodes), biases(biases), activation(activation)
		{
			for (double& w : this->weights.GetValues())
			{
				w = util::Random<double>(std::uniform_real_distribution<double>(wmin, wmax)) / std::sqrt((double)in->n_nodes);
			}
		}
		Layer(int n_nodes) : n_nodes(n_nodes) {}

		util::Matrix<double> Forward(util::Matrix<double>& input, bool start = false)
		{
			if (start)
			{
				outputs = input;
				return input;
			}

			util::Matrix<double> z = input * weights + biases;
			weightedInputs = z;
			util::Matrix<double> a = actf::Activation(activation, z);
			outputs = a;
			return a;
		}
	public: // Getters/setters
		util::Matrix<double>& GetWeights() { return weights; }
		util::Matrix<double>& GetBiases() { return biases; }
		util::Matrix<double>& GetWeightedInputs() { return weightedInputs; }
		util::Matrix<double>& GetOutputs() { return outputs; }
	private:
		int n_nodes = 0;
		actf::ACTIVATION_TYPE activation;
		util::Matrix<double> weights; // inputs x outputs
		util::Matrix<double> biases;
		util::Matrix<double> weightedInputs;
		util::Matrix<double> outputs;
		Layer* in = nullptr;
	};
}