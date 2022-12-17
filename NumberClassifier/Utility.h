#pragma once

#include <vector>
#include <cassert>
#include <random>

#define SELF (*this)

namespace util
{
	template<typename T>
	class Matrix
	{
	public:
		Matrix(std::vector<T> values, int rows, int columns, T init = (T)0)
			: values(values), rows(rows), columns(columns)
		{
			this->values.resize((std::size_t)rows * columns, init);
		}
		Matrix()
			: values(0), rows(0), columns(0)
		{}
	public:
		// operators
		const T& operator()(int row, int column) const
		{
			return values[row * columns + column];
		}
		T& operator()(int row, int column)
		{
			return values[row * columns + column];
		}
		T& operator[](int index)
		{
			return values[index];
		}
		const T& operator[](int index) const
		{
			return values[index];
		}

		Matrix operator*(const Matrix& rhs) const // row dot column
		{
			assert(columns == rhs.rows);
			Matrix res{{}, rows, rhs.columns};
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < rhs.columns; j++)
				{
					for (int k = 0; k < rhs.rows; k++)
					{
						res(i, j) += SELF(i, k) * rhs(k, j);
					}
				}
			}
			return res;
		}

		Matrix operator*(const T& rhs) const
		{
			Matrix res{ {}, rows, columns };
			int i = 0;
			for (const T& v : values)
			{
				res[i] = v * rhs;
				i++;
			}
			return res;
		}

		Matrix operator+(const Matrix& rhs)
		{
			assert(rows == rhs.rows && columns == rhs.columns);
			Matrix res({}, rows, columns);
			for (int i = 0; i < rows * columns; i++)
			{
				res[i] = values[i] + rhs.values[i];
			}
			return res;
		}

		Matrix operator-(const Matrix& rhs)
		{
			assert(rows == rhs.rows && columns == rhs.columns);
			Matrix res({}, rows, columns);
			for (int i = 0; i < rows * columns; i++)
			{
				res[i] = values[i] - rhs.values[i];
			}
			return res;
		}

		bool operator==(const Matrix& rhs) const
		{
			return values == rhs.values && rows == rhs.rows && columns == rhs.columns;
		}
	public:
		// utility
		Matrix GetTransposed()
		{
			Matrix res{ {}, columns, rows };
			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < columns; c++)
				{
					res(c, r) = SELF(r, c);
				}
			}
			return res;
		}
		bool SizeEqu(const Matrix& rhs)
		{
			return rows == rhs.rows && columns == rhs.columns;
		}
	public:
		// getters/settors
		std::vector<T>& GetValues() { return values; }
		int GetRows() { return rows; }
		int GetColumns() { return columns; }
		int GetSize() { return rows * columns; }
	private:
		std::vector<T> values;
		int rows;
		int columns;
	};

	template<typename T>
	struct DataPoint
	{
		DataPoint(Matrix<T> input, Matrix<T> expected) : input(input), expected(expected) {};
		DataPoint(Matrix<T> output, Matrix<T> expected, Matrix<T> input) : output(output), expected(expected), input(input) {};
		DataPoint() = default;

		Matrix<T> input;
		Matrix<T> expected;
		Matrix<T> output;
		T label;
	};

	inline std::random_device _rd;
	inline std::mt19937 _rng(/*_rd()*/36456355);
	template<typename T, typename distr>
	inline T Random(distr dist)
	{
		return dist(_rng);
	}

	template <typename T>
	inline Matrix<T> Hadamard(Matrix<T> lhs, Matrix<T> rhs)
	{
		assert(lhs.SizeEqu(rhs));
		Matrix<T> res{ {},lhs.GetRows(), lhs.GetColumns() };
		for (int i = 0; i < lhs.GetValues().size(); i++)
		{
			res[i] = lhs[i] * rhs[i];
		}
		return res;
	}
}