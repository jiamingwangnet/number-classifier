#pragma once

#include "Utility.h"
#include "Network.h"
#include <functional>

namespace util
{
	class Trainer
	{
	public:
		Trainer(int batchsize, std::vector<DataPoint<double>> trainData, std::vector<DataPoint<double>> testData)
			:
			batchsize(batchsize), trainData(trainData), testData(testData)
		{
			int tr_i = 0;
			std::vector<DataPoint<double>> batch;
			for (DataPoint<double>& dp : trainData)
			{
				batch.push_back(dp);
				tr_i++;
				if (tr_i % batchsize == 0)
				{
					batched_trainData.push_back(batch);
					batch.clear();
				}
			}
			if (batch.size() != 0)
			{
				batched_trainData.push_back(batch);
			}

			int te_i = 0;
			std::vector<DataPoint<double>> te_batch;
			for (DataPoint<double>& dp : testData)
			{
				te_batch.push_back(dp);
				te_i++;
				if (te_i % batchsize == 0)
				{
					batched_testData.push_back(te_batch);
					te_batch.clear();
				}
			}
			if (te_batch.size() != 0)
			{
				batched_testData.push_back(te_batch);
			}
		}

		void Train(net::Network& model, double learnRate)
		{
			for(int i = 0; i < batched_trainData.size(); i++)
			{
				Train(model, learnRate, i);
			}
		}

		void Train(net::Network& model, double learnRate, int batch)
		{
			/*std::vector<DataPoint<double>> data = batched_trainData[batch];
			for (DataPoint<double>& dp : data)
			{
				Offset offset{ dp.input };
				Noise noise{ dp.input };
				ProcessInput(offset, dp.input);
				ProcessInput(noise, dp.input);
			}*/
			model.Learn(batched_trainData[batch], learnRate);
		}

		void Test(net::Network& model)
		{
			model.CalculateOutputs(testData);
		}

		void Test(net::Network& model, int batch)
		{
			model.CalculateOutputs(batched_testData[batch]);
		}

		const std::vector<DataPoint<double>>& GetTestData() const
		{
			return testData;
		}

		const std::vector<std::vector<DataPoint<double>>>& GetTestDataBatches() const
		{
			return batched_testData;
		}
		
		const std::vector<std::vector<DataPoint<double>>>& GetTrainingDataBatches() const
		{
			return batched_trainData;
		}
	private:
		int batchsize;
		std::vector<std::vector<DataPoint<double>>> batched_trainData;
		std::vector<std::vector<DataPoint<double>>> batched_testData;

		std::vector<DataPoint<double>> trainData;
		std::vector<DataPoint<double>> testData;
	};
}