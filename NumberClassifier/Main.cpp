#include "MNISTReader.h"
#include "Network.h"
#include "Cost.h"
#include "Trainer.h"
#include <iostream>
#include <conio.h>

int main()
{
	std::cout << "Loading...\n";

	util::MNISTReader train_reader("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	std::vector<util::DataPoint<double>> train_data = train_reader.GetData(util::DATATYPE::TRAIN);

	util::MNISTReader test_reader("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	std::vector<util::DataPoint<double>> test_data = test_reader.GetData(util::DATATYPE::TEST);

	/*for (int i = 0; i < 100; i++) // list first 100 data points
	{
		util::DataPoint<double> d0 = data[i];

		std::cout << '\t' << d0.label << '\n';
		for (int r = 0; r < 28; r++)
		{
			for (int c = 0; c < 28; c++)
			{
				std::cout << (d0.input(r, c) > 0.0 ? '#' : ' ');
			}
			std::cout << '\n';
		}
	}*/

	std::cout << "Model Value Path: ";
	std::string valuePath;
	std::cin >> valuePath;

	
	net::Network model{ std::vector<int>{784, 256, 256, 10}, net::actf::ACTIVATION_TYPE::RELU, net::actf::ACTIVATION_TYPE::SOFTMAX };
	if (valuePath != "!")
	{
		model = net::Network{ valuePath };
	}

	util::Trainer trainer{ 100, train_data, test_data };
	std::cout << "\n----STARTED----\n";
	for (int i = 0, tr_batch = 0, te_batch = 0, epoch = 0;; tr_batch++, te_batch++)
	{
		trainer.Train(model, 0.05, tr_batch);
		trainer.Test(model, te_batch);

		std::cout << "\n-----------------------------------------------------------------------------------------\n";

		std::cout << "Epoch: " << epoch << '\n';
		std::cout << "Batch Epoch: " << i << "\n\n";

		std::cout << "Train Accuracy: " << (net::cstf::Accuracy(trainer.GetTrainingDataBatches()[tr_batch]) * 100.0) << '%' << '\n';
		std::cout << "Test Accuracy: " << (net::cstf::Accuracy(trainer.GetTestDataBatches()[te_batch]) * 100.0) << '%' << "\n\n";

		std::cout << "Train Cost: " << COST(trainer.GetTrainingDataBatches()[tr_batch]) << '\n';
		std::cout << "Test Cost: " << COST(trainer.GetTestDataBatches()[te_batch]) << '\n';

		std::cout << "-----------------------------------------------------------------------------------------\n";

		if (te_batch == trainer.GetTestDataBatches().size() - 1)
		{
			te_batch = 0;
		}
		if (tr_batch == trainer.GetTrainingDataBatches().size() - 1)
		{
			tr_batch = 0;
		}
		if (i % trainer.GetTrainingDataBatches().size() == 0 && i != 0)
		{
			epoch++;
		}
		i++;

		if (i % 500 == 0)
		{
			std::cout << "\nSaving...\n\n";
			model.Save("save.txt");
		}

		if (_kbhit()) break;
	}	

	model.Save("save.txt");

	for (;;)
	{
		int d_index = 0;

		std::string choice;
		
		do
		{
			std::cout << "Bitmap / Dataset (b/d): ";
			std::cin >> choice;
			std::cout << '\n';
		} while (!(choice == "b" || choice == "d"));

		util::DataPoint<double> data;

		if (choice == "d")
		{
			std::cout << "Input data index: ";
			std::cin >> d_index;
			std::cout << '\n';
			data = test_data[d_index];
		}
		else
		{
			std::string path;
			std::cout << "Bitmap image file (28x28): ";
			std::cin >> path;
			std::cout << '\n';
			data = util::MNISTReader::ReadBitmap(path);
		}

		std::cout << "label: " << data.label << '\n';
		for (int r = 0; r < 28; r++)
		{
			for (int c = 0; c < 28; c++)
			{
				std::cout << (data.input(0, r * 28 + c) > 0.0 ? (data.input(0, r * 28 + c) > 0.5 ? '#' : '.') : ' ');
			}
			std::cout << '\n';
		}
		model.CalculateOutputs(data);

		for (int i = 0; i < 10; i++)
		{
			std::cout << "\n" << i << ": " << data.output[i] << '\n';
		}
		std::cout << '\n';
	}

	return 0;
}