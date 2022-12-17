#pragma once

#include "Layer.h"

namespace net
{
	class Network
	{
	public:
		Network(std::vector<int> layer_c, actf::ACTIVATION_TYPE hiddenActiv, actf::ACTIVATION_TYPE outputActiv, double bias = 0.0)
			: n_layers((int)layer_c.size()), layer_c(layer_c), hiddenActiv(hiddenActiv), outputActiv(outputActiv)
		{
			layers.reserve(layer_c.size());
			int i = 0;
			for (int& s : layer_c)
			{
				if (i == 0)
				{
					layers.emplace_back(s);
				}
				else
				{
					Layer& before = layers[i - 1];
					layers.emplace_back(&before, util::Matrix<double>{ {}, 1, s, bias}, s, (i < layer_c.size() - 1) ? hiddenActiv : outputActiv);
				}
				i++;
			}

			weight_grad.resize(layers.size());
			bias_grad.resize(layers.size());

			ClearGradients();
		}

		Network(std::string path)
		{
			Load(path);
		}
	public: // network input/output
		void CalculateOutputs(util::DataPoint<double>& data)
		{
			util::Matrix<double> res = Feed(data.input);
			data.output = res;
		}

		void CalculateOutputs(std::vector<util::DataPoint<double>>& batch)
		{
			for (util::DataPoint<double>& dp : batch)
			{
				CalculateOutputs(dp);
			}
		}

		util::Matrix<double> Feed(util::Matrix<double> input)
		{
			util::Matrix<double> output = layers[0].Forward(input, true);
			for (auto layer_p = layers.begin() + 1; layer_p != layers.end(); ++layer_p)
			{
				output = layer_p->Forward(output);
			}
			return output;
		}

		void Save(std::string name)
		{
			std::ofstream out(name);

			// n layers
			out << layer_c.size() << '\n';

			// layer sizes
			// l0 l1 l2 ...
			for (int& c : layer_c)
			{
				out << c << ' ';
			}
			out << '\n';

			// activation functions
			// hidden output
			out << (int)hiddenActiv << ' ' << (int)outputActiv << '\n';

			// biases and weights
			// l0 weights ...
			// l0 biases ...
			// l1 weights ...
			// l1 biases ...
			bool f = true;
			for (Layer& layer : layers)
			{
				if (f)
				{
					f = false; continue;
				}
				for (double& w : layer.GetWeights().GetValues())
				{
					out << w << ' ';
				}
				out << '\n';
				for (double& b : layer.GetBiases().GetValues())
				{
					out << b << ' ';
				}
				out << '\n';
			}
			out.close();
		}

		void Load(std::string path)
		{
			std::ifstream in(path);
			in >> n_layers;

			std::vector<int> layer_c;

			for (int i = 0; i < n_layers; i++)
			{
				int c = 0;
				in >> c;
				layer_c.push_back(c);
			}

			this->layer_c = layer_c;

			int hiddenActiv = 0;
			int outputActiv = 0;

			in >> hiddenActiv;
			in >> outputActiv;

			this->hiddenActiv = (actf::ACTIVATION_TYPE)hiddenActiv;
			this->outputActiv = (actf::ACTIVATION_TYPE)outputActiv;

			// do the thing
			layers.reserve(layer_c.size());
			int i = 0;
			for (int& s : layer_c)
			{
				if (i == 0)
				{
					layers.emplace_back(s);
				}
				else
				{
					Layer& before = layers[i - 1];
					// initialize bias to 0 and set them later
					layers.emplace_back(&before, util::Matrix<double>{ {}, 1, s, 0.0}, s, (i < layer_c.size() - 1) ? this->hiddenActiv : this->outputActiv);
				}
				i++;
			}

			for (Layer& layer : layers)
			{
				for (int i = 0; i < layer.GetWeights().GetSize(); i++)
				{
					double w = 0.0;
					in >> w;
					layer.GetWeights()[i] = w;
				}
				for (int i = 0; i < layer.GetBiases().GetSize(); i++)
				{
					double b = 0.0;
					in >> b;
					layer.GetBiases()[i] = b;
				}
			}

			weight_grad.resize(layers.size());
			bias_grad.resize(layers.size());

			ClearGradients();

			in.close();
		}
	public: // gradient descent
		void Learn(std::vector<util::DataPoint<double>>& data, double learnRate)
		{
			for (util::DataPoint<double>& dataP : data)
			{
				GetGradients(dataP);
			}

			ApplyGradients(learnRate / (double)data.size());
			ClearGradients();
		}

		void SlowLearn(std::vector<util::DataPoint<double>>& data, double learnRate)
		{
			static constexpr double h = 0.0000001;

			for (util::DataPoint<double>& dataP : data)
			{
				CalculateOutputs(dataP);
			}

			double cost = COST(data);

			int l_i = 0;
			for (Layer& layer : layers)
			{
				int w_i = 0;
				for (double& w : layer.GetWeights().GetValues())
				{
					w += h;
					for (util::DataPoint<double>& dataP : data)
					{
						CalculateOutputs(dataP);
					}
					w -= h;
					double new_cost = COST(data);
					weight_grad[l_i][w_i] = (new_cost - cost) / h;
					w_i++;
				}
				int b_i = 0;
				for (double& b : layer.GetBiases().GetValues())
				{
					b += h;
					for (util::DataPoint<double>& dataP : data)
					{
						CalculateOutputs(dataP);
					}
					b -= h;
					double new_cost = COST(data);
					bias_grad[l_i][b_i] = (new_cost - cost) / h;
					b_i++;
				}
				l_i++;
			}

			ApplyGradients(learnRate);

			ClearGradients();
		}

#ifdef UNIT_TEST
	public:
#else
	private:
#endif
		void ApplyGradients(double learnRate)
		{
			int i = 0;
			for (Layer& l : layers)
			{
				util::Matrix<double>& l_wg = weight_grad[i];
				util::Matrix<double>& l_bg = bias_grad[i];

				l.GetWeights() = l.GetWeights() - l_wg * learnRate;
				l.GetBiases() = l.GetBiases() - l_bg * learnRate;
				i++;
			}

#ifdef UNIT_TEST
			out_weight_grad = weight_grad;
			out_bias_grad = bias_grad;
#endif
		}

		void ClearGradients()
		{
			int w_i = 0;
			for (util::Matrix<double>& grad : weight_grad)
			{
				grad = { {}, layers[w_i].GetWeights().GetRows(), layers[w_i].GetWeights().GetColumns() };
				w_i++;
			}
			int b_i = 0;
			for (util::Matrix<double>& grad : bias_grad)
			{
				grad = { {}, layers[b_i].GetBiases().GetRows(), layers[b_i].GetBiases().GetColumns() };
				b_i++;
			}
		}

		void UpdateGradients(int layer_i, util::Matrix<double> nodeValues)
		{
			util::Matrix<double> bias_g = nodeValues;
			util::Matrix<double> weight_g = layers[(std::size_t)layer_i - 1].GetOutputs().GetTransposed() * nodeValues;

			weight_grad[layer_i] = weight_grad[layer_i] + weight_g;
			bias_grad[layer_i] = bias_grad[layer_i] + bias_g;
		}
		
		void GetGradients(util::DataPoint<double>& dataP)
		{
			CalculateOutputs(dataP);

			util::Matrix<double> nodeValues = OutputLayerValues(dataP);
			UpdateGradients(n_layers - 1, nodeValues);

			for (int i = n_layers - 2; i > 0; i--)
			{
				nodeValues = HiddenLayerValues(i, nodeValues);
				UpdateGradients(i, nodeValues);
			}
		}

		util::Matrix<double> OutputLayerValues(util::DataPoint<double>& dataP)
		{
			util::Matrix<double> nodeValues = actf::Activation_derivative(outputActiv, layers[(std::size_t)n_layers - 1].GetWeightedInputs());
			int i = 0;
			for (double& value : nodeValues.GetValues())
			{
				value *= COST_DERIVATIVE(dataP.output[i], dataP.expected[i]); // a typo right here wasted 2 weeks of my life
				i++;
			}
			return nodeValues;
		}

		util::Matrix<double> HiddenLayerValues(int layer_i, util::Matrix<double> nodeValues)
		{
			return util::Hadamard(nodeValues * layers[(std::size_t)layer_i + 1].GetWeights().GetTransposed(), actf::Activation_derivative(hiddenActiv, layers[layer_i].GetWeightedInputs()));
		}

#ifdef UNIT_TEST
	public:
#else
	private:
#endif
		std::vector<Layer> layers;
		actf::ACTIVATION_TYPE hiddenActiv;
		actf::ACTIVATION_TYPE outputActiv;

		std::vector<util::Matrix<double>> weight_grad;
		std::vector<util::Matrix<double>> bias_grad;

		std::vector<int> layer_c;

#ifdef UNIT_TEST
		std::vector<util::Matrix<double>> out_weight_grad;
		std::vector<util::Matrix<double>> out_bias_grad;
#endif
		int n_layers;
	};
}