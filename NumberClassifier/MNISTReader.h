#pragma once
#define NOMINMAX

#include <vector>
#include "Utility.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <Windows.h>
#include <cassert>
#include <iostream>

namespace util
{
	static constexpr int TRAIN_DATA_ITEMS = 60000;
	static constexpr int TEST_DATA_ITEMS = 10000;

	// https://stackoverflow.com/questions/3823921/convert-big-endian-to-little-endian-when-reading-from-a-binary-file
	template <class T>
	void endswap(T* objp)
	{
		unsigned char* memp = reinterpret_cast<unsigned char*>(objp);
		std::reverse(memp, memp + sizeof(T));
	}

	struct TRAINFILELABELS
	{
		std::uint32_t magic;
		std::uint32_t items; // must be 60000 for train data and 10000 for test data
		std::uint8_t labels[TRAIN_DATA_ITEMS];
	};

	struct TRAINFILEIMAGES
	{
		std::uint32_t magic;
		std::uint32_t items; // must be 60000 for train data and 10000 for test data
		std::uint32_t rows;
		std::uint32_t columns;
		std::uint8_t imagesbytes[TRAIN_DATA_ITEMS * 28 * 28];
	};

	struct TESTFILELABELS
	{
		std::uint32_t magic;
		std::uint32_t items; // must be 60000 for train data and 10000 for test data
		std::uint8_t labels[TEST_DATA_ITEMS];
	};

	struct TESTFILEIMAGES
	{
		std::uint32_t magic;
		std::uint32_t items; // must be 60000 for train data and 10000 for test data
		std::uint32_t rows;
		std::uint32_t columns;
		std::uint8_t imagesbytes[TEST_DATA_ITEMS * 28 * 28];
	};

	enum class DATATYPE
	{
		TRAIN,
		TEST
	};

	// numbers 0 1 2 3 4 5 6 7 8 9 vertically
	inline Matrix<double> LabelToMatrix(int label)
	{
		Matrix<double> res{ {}, 1, 10 };
		res[label] = 1.0;
		return res;
	}

	class Noise
	{
	public:
		Noise(Matrix<double>& input)
			: input(input)
		{}
		Matrix<double> operator()(double level)
		{
			Matrix<double> res{ {}, input.GetRows(), input.GetColumns() };
			int i = 0;
			static constexpr double acc = 100000.0;
			for (double& v : input.GetValues())
			{
				res[i] = input[i];
				res[i] += Random<double>(std::normal_distribution<double>(0.0, 0.15)) * level;
				if (res[i] > 1.0)
				{
					res[i] = 1.0;
				}
				else if (res[i] < 0.0)
				{
					res[i] = 0.0;
				}
				i++;
			}
			return res;
		}
	private:
		Matrix<double>& input;
	};

	class Offset
	{
	public:
		Offset(Matrix<double>& input)
			: input(input)
		{}
		Matrix<double> operator()(double level)
		{
			Matrix<double> res{ {},input.GetRows(), input.GetColumns() };
			int xOffset = (int)((double)Random<int>(std::uniform_int_distribution<int>(0,8)) * level);
			int yOffset = (int)((double)Random<int>(std::uniform_int_distribution<int>(0,8)) * level);

			int x = 0;
			int y = 0;
			for (double& v : input.GetValues())
			{
				if (!((x + xOffset >= 28 || x + xOffset < 0) || (y + yOffset >= 28 || y + yOffset < 0)))
				{
					int off_x = x + xOffset;
					int off_y = y + yOffset;

					res[off_y * 28 + off_x] = input[y * 28 + x];
				}

				x++;
				if (x % 28 == 0)
				{
					x = 0;
					y++;
				}
			}

			return res;
		}
	private:
		Matrix<double>& input;
	};
	template<class Processor>
	void ProcessInput(Processor proc, Matrix<double>& input)
	{
		input = proc(Random<double>(std::uniform_real_distribution<double>(-1.0, 1.0)));
	}

	class MNISTReader
	{
	public:
		MNISTReader(std::string dataPath, std::string labelsPath)
			: dataPath(dataPath), labelsPath(labelsPath), data()
		{}

		static DataPoint<double> ReadBitmap(std::string path)
		{
			std::ifstream image;

			image.open(path, std::ios::binary);
			if (!image)
			{
				std::cout << "Image not found\n";
				return {};
			}

			uint8_t* datBuff[2] = { nullptr, nullptr };

			datBuff[0] = new uint8_t[sizeof(BITMAPFILEHEADER)];
			datBuff[1] = new uint8_t[sizeof(BITMAPINFOHEADER)];

			image.read((char*)datBuff[0], sizeof(BITMAPFILEHEADER));
			image.read((char*)datBuff[1], sizeof(BITMAPINFOHEADER));

			BITMAPFILEHEADER* bmFileHeader = (BITMAPFILEHEADER*)datBuff[0];
			BITMAPINFOHEADER* bmInfoHeader = (BITMAPINFOHEADER*)datBuff[1];

			assert(bmFileHeader->bfType == 0x4D42);
			assert(bmInfoHeader->biBitCount == 24 || bmInfoHeader->biBitCount == 32);
			assert(bmInfoHeader->biCompression == BI_RGB);

			const bool is32b = bmInfoHeader->biBitCount == 32;

			int width = bmInfoHeader->biWidth;
			int height;

			// test for reverse row order and control
			// y loop accordingly
			int yStart;
			int yEnd;
			int dy;
			if (bmInfoHeader->biHeight < 0)
			{
				height = -bmInfoHeader->biHeight;
				yStart = 0;
				yEnd = height;
				dy = 1;
			}
			else
			{
				height = bmInfoHeader->biHeight;
				yStart = height - 1;
				yEnd = -1;
				dy = -1;
			}
			image.seekg(bmFileHeader->bfOffBits);
			// padding is for the case of of 24 bit depth only
			const int padding = (4 - (width * 3) % 4) % 4;

			assert(height == 28);
			assert(width == 28);

			Matrix<double> input{ {}, 1, 28 * 28 };

			for (int y = yStart; y != yEnd; y += dy)
			{
				for (int x = 0; x < width; x++)
				{
					unsigned int color = 0;

					color += image.get();
					color += image.get();
					color += image.get();

					color /= 3;
					double c_res = (double) color / 255;
					
					input[y * 28 + x] = c_res;

					if (is32b)
					{
						image.seekg(1, std::ios::cur);
					}
				}
				if (!is32b)
				{
					image.seekg(padding, std::ios::cur);
				}
			}

			for (int i = 0; i < 2; i++)
				delete datBuff[i];

			DataPoint<double> res;
			res.input = input;
			res.label = (double)std::stoi(path.substr(0, path.find_last_of("."))); // file must be in the same directory for correct label
			res.expected = LabelToMatrix((int)res.label);

			image.close();

			return res;
		}

		std::vector<DataPoint<double>> GetData(DATATYPE type)
		{
			std::ifstream labels_f(labelsPath, std::ios::binary);
			std::ifstream data_f(dataPath, std::ios::binary);
	
			switch (type)
			{
			case util::DATATYPE::TRAIN:
			{
				TRAINFILELABELS* trainlabels = new TRAINFILELABELS;
				TRAINFILEIMAGES* trainimages = new TRAINFILEIMAGES;

				labels_f.read(reinterpret_cast<char*>(trainlabels), sizeof(*trainlabels));
				data_f.read(reinterpret_cast<char*>(trainimages), sizeof(*trainimages));

				// swap the endian
				endswap(&trainlabels->magic);
				endswap(&trainlabels->items);

				endswap(&trainimages->magic);
				endswap(&trainimages->items);
				endswap(&trainimages->rows);
				endswap(&trainimages->columns);

				for (unsigned int i = 0; i < trainlabels->items; i++)
				{
					unsigned char label = trainlabels->labels[i];
					DataPoint<double> dp;

					dp.label = (double)label;
					dp.expected = LabelToMatrix(label);
					Matrix<double> number{ {}, 1, 28 * 28 };
					
					for (unsigned int j = i * 28 * 28; j < (i + 1) * 28 * 28; j++)
					{
						number[j - (i * 28 * 28)] = (double)trainimages->imagesbytes[j] / 255.0;
					}

					dp.input = number;

					Offset offset{ dp.input };
					Noise noise{ dp.input };
					ProcessInput(offset, dp.input);
					ProcessInput(noise, dp.input);

					data.push_back(dp);
				}

				delete trainlabels;
				delete trainimages;
			}
				break;
			case util::DATATYPE::TEST:
				{
					TESTFILELABELS* testlabels = new TESTFILELABELS;
					TESTFILEIMAGES* testimages = new TESTFILEIMAGES;

					labels_f.read(reinterpret_cast<char*>(testlabels), sizeof(*testlabels));
					data_f.read(reinterpret_cast<char*>(testimages), sizeof(*testimages));

					// swap the endian
					endswap(&testlabels->magic);
					endswap(&testlabels->items);

					endswap(&testimages->magic);
					endswap(&testimages->items);
					endswap(&testimages->rows);
					endswap(&testimages->columns);

					for (unsigned int i = 0; i < testlabels->items; i++)
					{
						unsigned char label = testlabels->labels[i];
						DataPoint<double> dp;

						dp.label = (double)label;
						dp.expected = LabelToMatrix(label);
						Matrix<double> number{ {}, 1, 28 * 28 };

						for (unsigned int j = i * 28 * 28; j < (i + 1) * 28 * 28; j++)
						{
							number[j - (i * 28 * 28)] = (double)testimages->imagesbytes[j] / 255.0;
						}

						Offset offset{ number };
						Noise noise{ number };
						ProcessInput(offset, number);
						ProcessInput(noise, number);

						dp.input = number;

						data.push_back(dp);
					}

					delete testlabels;
					delete testimages;
				}
			}
			return data;
		}
	private:
		std::string dataPath;
		std::string labelsPath;
		std::vector<DataPoint<double>> data;
	};
}