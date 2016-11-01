#ifndef MY_MATRIX_H
#define MY_MATRIX_H

#include <iostream>

using std::cout;
using std::endl;

class vector2d
{
public:

	float v[2];

	float operator[](unsigned int idx) { return (idx < 2 ? v[idx] : 0.0f); }
	void operator=(vector2d b){ v[0] = b.v[0]; v[1] = b.v[1]; }
};
#ifdef OLD_MATRIX
class matrix
{
public:
	matrix(){
		this->m_sizeX = 0; m_sizeY = 0;
		data = 0;
	}


	matrix(matrix *p){
		this->m_sizeX = p->m_sizeX; m_sizeY = p->m_sizeY;
		this->create();
		for (int i = 0; i < m_sizeX; i++)
		{
			for (int j = 0; j < m_sizeY; j++)
				data[i][j] = (*p)[i][j];
			//memcpy((void*)&data[i][j], (void*)&((*p)[i][j]), sizeof(float));
		}
		is_transposed = false;
	}

	matrix(matrix &p){
		this->m_sizeX = p.m_sizeX; m_sizeY = p.m_sizeY;
		this->create();
		for (int i = 0; i < m_sizeX; i++)
		{
			for (int j = 0; j < m_sizeY; j++)
				data[i][j] = p[i][j];
			//memcpy((void*)&data[i][j], (void*)&((p)[i][j]), sizeof(float));
		}
		is_transposed = false;
	}

	matrix(unsigned int n, unsigned int m)
	{
		m_sizeX = n;
		m_sizeY = m;
		create();
	}
	~matrix()
	{
		destroy();
		if (out) delete out;
	}

	void destroy()
	{
		if (data != 0)
		{
			for (int i = 0; i < m_sizeX; i++)
			{
				delete[] data[i];
			}
			delete[] data;
		}
		data = 0;
		is_transposed = false;
	}
	void create()
	{
		if (m_sizeY > 0 && m_sizeX > 0)
		{
			data = new float*[m_sizeX];
			for (int i = 0; i < m_sizeX; i++)
			{
				data[i] = new float[m_sizeY];
				for (int j = 0; j < m_sizeY; j++)
					data[i][j] = 0.0f;
			}
		}
		is_transposed = false;
	}

	void operator=(matrix &b)
	{
		this->destroy();
		this->m_sizeX = b.NumRows();
		this->m_sizeY = b.NumColumns();
		this->create();
		for (int i = 0; i < m_sizeX; i++)
		{
			for (int j = 0; j < m_sizeY; j++)
				data[i][j] = b(i, j);
			//memcpy((void*)&data[i][j], (void*)&b[i][j], sizeof(float));
		}


	}

	void operator=(matrix *b)
	{
		this->destroy();
		this->m_sizeX = b->NumRows();
		this->m_sizeY = b->NumColumns();
		this->create();
		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
				data[i][j] = (*b)(i, j);
			//memcpy((void*)&data[i][j], (void*)&((*b)[i][j]), sizeof(float));
		}
	}





	float& operator()(unsigned int i, unsigned int j)
	{
		return get(i, j);
	}

	float& get(unsigned int i, unsigned int j)
	{
		float null_return = 0.0f;
		if (i < NumRows() && j < NumColumns())
			return is_transposed ? data[j][i] : data[i][j];
		else return null_return;
	}
	// Hadamard element wise product
	matrix operator | (matrix &b)
	{
		if (this->NumColumns() == b.NumColumns() && this->NumRows() == b.NumRows())
		{
			if (out) delete out;
			out = new matrix(this->NumRows(), this->NumColumns());

			for (int i = 0; i < this->NumRows(); i++)
			{

				for (int j = 0; j < this->NumColumns(); j++)
				{
					//for (int k = 0; k < this->NumColumns(); k++)
					{
						(*out)(i, j) = get(i, j) * b(i, j);
					}
				}
			}

			return *out;
		}
		
		return matrix(0, 0);
	}

	matrix operator*(matrix &b)
	{
		if (b.NumColumns() == 1 && b.NumRows() == 1)
		{
			return (*this) * b(0, 0);
		}
		if (this->NumColumns() == b.NumRows())
		{
			if (out) delete out;
			out = new matrix(this->NumRows(), b.NumColumns());

			for (int i = 0; i < this->NumRows(); i++)
			{

				for (int j = 0; j < b.NumColumns(); j++)
				{
					for (int k = 0; k < this->NumColumns(); k++)
					{
						(*out)(i, j) += get(i, k) * b(k, j);
					}
				}
			}

			return *out;
		}
		return matrix(0, 0);
	}

	matrix operator*(float s)
	{
		if (out) delete out;
		out = new matrix(this->NumRows(), this->NumColumns());

		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
			{
				(*out)(i, j) = get(i, j) * s;
			}
		}
		return *out;
	}

	matrix operator/(float s)
	{
		if (out) delete out;
		out = new matrix(this->NumRows(), this->NumColumns());

		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
			{
				(*out)(i, j) = get(i, j) / s;
			}
		}
		return *out;
	}
	matrix operator+(float s)
	{
		if (out) delete out;
		out = new matrix(this->NumRows(), this->NumColumns());

		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
			{
				(*out)(i, j) = get(i, j) + s;
			}
		}
		return *out;
	}
	matrix operator+(matrix &b)
	{
		if (this->NumColumns() != b.NumColumns() || this->NumRows() != b.NumRows())
			return matrix(0, 0);
		else
		{
			if (out) delete out;
			out = new matrix(this->NumRows(), this->NumColumns());

			for (int i = 0; i < this->NumRows(); i++)
			{
				for (int j = 0; j < this->NumColumns(); j++)
				{
					(*out)(i, j) = this->get(i, j) + b.get(i, j);
				}
			}

			return *out;
		}
		return matrix(0, 0);
	}

	matrix operator-(matrix &b)
	{
		if (this->NumColumns() != b.NumColumns() || this->NumRows() != b.NumRows())
			return matrix(0, 0);
		else
		{
			if (out) delete out;
			out = new matrix(this->NumRows(), this->NumColumns());

			for (int i = 0; i < this->NumRows(); i++)
			{

				for (int j = 0; j < this->NumColumns(); j++)
				{
					(*out)(i, j) = this->get(i, j) - b.get(i, j);
				}
			}

			return *out;
		}
		return matrix(0, 0);
	}

	void print()
	{
		if (this->NumRows() == 0 || this->NumColumns() == 0 || this->data == 0)
		{
			cout << "Empty Matrix" << endl;
			return;
		}
		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
			{
				cout << get(i, j) << "  ";
			}
			cout << endl;
		}
	}



	float trace()
	{
		float sum = 0.0f;
		if (m_sizeX != m_sizeY) return 0.0f;

		for (int i = 0; i < this->NumRows(); i++)
		{
			for (int j = 0; j < this->NumColumns(); j++)
			{
				if (i == j) sum += get(i, j);
			}

		}
		return sum;
	}

	bool IsSquare() 
	{
		if (m_sizeX == m_sizeY)
			return true;

		return false;
	}

	bool ContainsNAN()
	{
		for (int i = 0; i < m_sizeX; i++)
		{
			for (int j = 0; j < m_sizeY; j++)
			{
				if (data[i][j] != data[i][j])
					return true;
			}
		}
		return false;
	}

	void transpose()
	{
		is_transposed = !is_transposed;
	}



	inline unsigned int NumRows() 
	{ 
		return  (is_transposed ? m_sizeY : m_sizeX); 
	}
	inline unsigned int NumColumns() 
	{ 
		return (is_transposed ? m_sizeX : m_sizeY); 
	}


	bool is_transposed = false;
	unsigned int m_sizeX = 0;
	unsigned int m_sizeY = 0;
	float **data = 0;


private:
	float* operator[](unsigned int a)
	{
		if (a < m_sizeX)
			return data[a];
		else return 0;
	}


	matrix *out = 0;


};

#endif

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

#include <complex>

using std::complex;

using std::cout;
using std::endl;




class LINALG
{
public:
	template< class T >
	static inline void SWAP(T &a, T &b)
	{
		T temp = a;
		a = b;
		b = temp;
	}




	static inline float RandomFloat(float min, float max)
	{
		float r = (float)rand() / (float)RAND_MAX;
		return min + r * (max - min);
	}

	static inline float RandomInt(int min, int max)
	{
		float r = (float)rand() / (float)RAND_MAX;
		return (int)((float)min + r * float(max - min));
	}

	template<class T>
	static inline T sgn(T x)
	{
		if (x > 0.0 + DBL_EPSILON)
			return 1.0;
		if (x < 0.0 - DBL_EPSILON)
			return -1.0;

		return x;
	}

	static inline double round_to_n_digits(double x, int n)
	{
		double scale = pow(10.0, ceil(log10(fabs(x))) + n);

		return round(x * scale) / scale;
	}

	class vector2d
	{
	public:

		float v[2];

		float operator[](unsigned int idx) { return (idx < 2 ? v[idx] : 0.0f); }
		void operator=(vector2d b){ v[0] = b.v[0]; v[1] = b.v[1]; }
	};
private:
	template < class T > class matrix;



	template<class T>
	class matrix
	{
	public:
		matrix<T>(){
			this->SX = 0; SY = 0;
			data = 0;
		}


		matrix<T>(matrix<T> *p){
			this->SX = p->SX; SY = p->SY;
			this->create();
			for (int i = 0; i < SX; i++)
			{
				for (int j = 0; j < SY; j++)
					data[i * SY + j] = (*p)(i, j);
			}
			is_transposed = false;
		}

		matrix<T>(matrix &p){
			this->SX = p.SX; SY = p.SY;
			this->create();
			for (int i = 0; i < SX; i++)
			{
				for (int j = 0; j < SY; j++)
					data[i * SY + j] = p(i, j);
			}
			is_transposed = false;
		}

		matrix<T>(unsigned int n, unsigned int m)
		{
			SX = n;
			SY = m;
			create();
		}
		~matrix<T>()
		{
			destroy();
			if (out) delete out;
		}

		void destroy()
		{
			if (data != 0)
			{
				/*for (int i = 0; i < SX; i++)
				{
				delete[] data[i];
				}*/
				delete[] data;
			}
			data = 0;
			is_transposed = false;
		}
		void create()
		{
			if (SY > 0 && SX > 0)
			{
				data = new T[SX * SY];
				for (int i = 0; i < SX; i++)
				{

					for (int j = 0; j < SY; j++)
						data[i * SY + j] = 0.0;
				}
			}
			is_transposed = false;
		}
		void create(int sizeX, int sizeY)
		{
			SX = sizeX;
			SY = sizeY;
			if (SY > 0 && SX > 0)
			{
				data = new T[SX * SY];
				for (int i = 0; i < SX; i++)
				{

					for (int j = 0; j < SY; j++)
						data[i * SY + j] = 0.0;
				}
			}
			is_transposed = false;
		}

		void operator=(matrix &b)
		{
			if (!(this->NumRows() == b.NumRows()) || !(this->NumCols() == b.NumCols()))
			{
				this->destroy();
				this->SX = b.NumRows();
				this->SY = b.NumCols();
				this->create();
			}
			for (int i = 0; i < SX; i++)
			{
				for (int j = 0; j < SY; j++)
					data[i * SY + j] = b(i, j);
			}
		}

		inline bool NotEqual(T test_value, T compare)
		{
			return ((test_value > compare + this->precision) || (test_value < compare - this->precision));
		}


		void operator=(matrix *b)
		{
			if (!(this->NumRows() == b->NumRows()) ||
				!(this->NumCols() == b->NumCols()))
			{
				this->destroy();
				this->SX = b->NumRows();
				this->SY = b->NumCols();
				this->create();
			}

			for (int i = 0; i < this->NumRows(); i++)
			{
				for (int j = 0; j < this->NumCols(); j++)
					data[i * SY + j] = (*b)(i, j);
			}
		}


		bool operator==(matrix &b)
		{

			if (!this->EqualSize(b))
			{
				return false;
			}
			for (int i = 0; i < NumRows(); i++)
			{
				for (int j = 0; j < NumCols(); j++)
				{
					if (NotEqual(get(i, j), b(i, j)))
						return false;
				}
			}
			return true;
		}

		inline bool operator!=(matrix& b)
		{
			return !((*this) == b);
		}


		inline T& operator()(unsigned int i, unsigned int j)
		{
			if (i < NumRows() && j < NumCols())
				return get(i, j);
		}

		// access without bounds ... checking ...

	private:
		inline T& get(unsigned int i, unsigned int j)
		{
			return data[i * NumCols() + j];
			/*T null_return = 0.0;
			if (i < NumRows() && j < NumCols())
			return is_transposed ? data[j*NumCols() + i] : data[i * NumCols() + j];
			else return null_return;*/
		}

	public:

		// Hadamard element wise product
		matrix& operator | (matrix &b)
		{
			if ((this->NumCols() == b.NumCols()) && (this->NumRows() == b.NumRows()))
			{
				//	if (out) delete out;
				//	out = new matrix(this->NumRows(), this->NumCols());
				if (out)
				{
					if (!((out->NumRows() == this->NumRows()) &&
						(out->NumCols() == this->NumCols())))
					{
						delete out;
						out = new matrix(this->NumRows(), this->NumCols());
					}
				}
				else
				{
					out = new matrix(this->NumRows(), this->NumCols());
				}

				for (int i = 0; i < this->NumRows(); i++)
				{

					for (int j = 0; j < this->NumCols(); j++)
					{
						//for (int k = 0; k < this->NumCols(); k++)
						{
							(*out)(i, j) = get(i, j) * b(i, j);
						}
					}
				}

				return *out;
			}

			return matrix(0, 0);
		}

		matrix& operator*(matrix &b)
		{
			/*if (b.NumCols() == 1 && b.NumRows() == 1)
			{
			//is a 1x1 matrix treated like a scalar?
			return (*this) * b(0, 0);
			}*/
			if (this->NumCols() == b.NumRows())
			{
				//if (out) delete out;
				//out = new matrix(this->NumRows(), b.NumCols());
				if (out)
				{
					if (!((out->NumRows() == this->NumRows()) &&
						(out->NumCols() == b.NumCols())))
					{
						delete out;
						out = new matrix(this->NumRows(), b.NumCols());
					}
				}
				else
				{
					out = new matrix(this->NumRows(), b.NumCols());
				}

				for (int i = 0; i < this->NumRows(); i++)
				{

					for (int j = 0; j < b.NumCols(); j++)
					{
						(*out)(i, j) = 0.0;
						for (int k = 0; k < this->NumCols(); k++)
						{
							(*out)(i, j) += get(i, k) * b(k, j);
						}
					}
				}

				return *out;
			}
			return matrix<T>(0, 0);
		}

		matrix& operator*(T s)
		{
			// prevent infinity, negative infinity, complex infinity or other NAN's from anihilating our matrix
			/*if (s != s)
			{
			cout << "Error (operator * scalar): attempt to multiply by a NAN" << endl;
			return (*this);
			}*/

			if (out)
			{

				if (!((out->NumRows() == this->NumRows()) &&
					(out->NumCols() == this->NumCols())))
				{
					delete out;
					out = new matrix(this->NumRows(), this->NumCols());
				}
			}
			else
			{
				out = new matrix(this->NumRows(), this->NumCols());
			}

			//if (out)delete out;
			//out = new matrix(this->NumRows(), this->NumCols());

			for (int i = 0; i < this->NumRows(); i++)
			{
				for (int j = 0; j < this->NumCols(); j++)
				{
					(*out)(i, j) = get(i, j) * s;
				}
			}
			return *out;
		}

		matrix& operator/(T s)
		{
			//if (out) delete out;
			//out = new matrix(this->NumRows(), this->NumCols());

			if (out)
			{
				if (!((out->NumRows() == this->NumRows()) &&
					(out->NumCols() == this->NumCols())))
				{
					delete out;
					out = new matrix(this->NumRows(), this->NumCols());
				}
			}
			else
			{
				out = new matrix(this->NumRows(), this->NumCols());
			}
			for (int i = 0; i < this->NumRows(); i++)
			{
				for (int j = 0; j < this->NumCols(); j++)
				{
					(*out)(i, j) = get(i, j) / s;
				}
			}
			return *out;
		}

		matrix& operator+(T s)
		{
			//if (out)delete out;
			//out = new matrix(this->NumRows(), this->NumCols());
			if (out)
			{
				if (!((out->NumRows() == this->NumRows()) &&
					(out->NumCols() == this->NumCols())))
				{
					delete out;
					out = new matrix(this->NumRows(), this->NumCols());
				}
			}
			else
			{
				out = new matrix(this->NumRows(), this->NumCols());
			}

			for (int i = 0; i < this->NumRows(); i++)
			{
				for (int j = 0; j < this->NumCols(); j++)
				{
					(*out)(i, j) = get(i, j) + s;
				}
			}
			return *out;
		}

		matrix& operator+(matrix &b)
		{
			if (this->NumCols() != b.NumCols() || this->NumRows() != b.NumRows())
			{
				return matrix<T>(0, 0);
			}
			else
			{
				//if (out)delete out;
				//out = new matrix(this->NumRows(), this->NumCols());
				if (out)
				{
					if (!((out->NumRows() == this->NumRows()) &&
						(out->NumCols() == this->NumCols())))
					{
						delete out;
						out = new matrix(this->NumRows(), this->NumCols());
					}
				}
				else
				{
					out = new matrix(this->NumRows(), this->NumCols());
				}

				for (int i = 0; i < this->NumRows(); i++)
				{
					for (int j = 0; j < this->NumCols(); j++)
					{
						(*out)(i, j) = this->get(i, j) + b.get(i, j);
					}
				}

				return *out;
			}
			return matrix(0, 0);
		}

		matrix& operator-(matrix &b)
		{
			if (this->NumCols() != b.NumCols() || this->NumRows() != b.NumRows())
				return matrix(0, 0);
			else
			{
				//	if (out)delete out;
				//	out = new matrix(this->NumRows(), this->NumCols());
				if (out)
				{
					if (!((out->NumRows() == this->NumRows()) &&
						(out->NumCols() == this->NumCols())))
					{
						delete out;
						out = new matrix(this->NumRows(), this->NumCols());
					}
				}
				else
				{
					out = new matrix(this->NumRows(), this->NumCols());
				}

				for (int i = 0; i < this->NumRows(); i++)
				{

					for (int j = 0; j < this->NumCols(); j++)
					{
						(*out)(i, j) = this->get(i, j) - b.get(i, j);
					}
				}

				return *out;
			}
			return matrix(0, 0);
		}

		// solution for matrix multiplication involving transposed matrices ...
		matrix& mul_transposed(matrix &b)
		{
			/*if (b.NumCols() == 1 && b.NumRows() == 1)
			{
			//is a 1x1 matrix treated like a scalar?
			return (*this) * b(0, 0);
			}*/
			if (this->NumRows() == b.NumRows())
			{
				//if (out) delete out;
				//out = new matrix(this->NumRows(), b.NumCols());
				if (out)
				{
					if (!((out->NumRows() == this->NumCols()) &&
						(out->NumCols() == b.NumCols())))
					{
						delete out;
						out = new matrix(this->NumCols(), b.NumCols());
					}
				}
				else
				{
					out = new matrix(this->NumCols(), b.NumCols());
				}

				for (int i = 0; i < this->NumCols(); i++)
				{

					for (int j = 0; j < b.NumCols(); j++)
					{
						(*out)(i, j) = 0.0;
						for (int k = 0; k < this->NumRows(); k++)
						{
							(*out)(i, j) += get(k, i) * b(k, j);
						}
					}
				}

				return *out;
			}
			return matrix<T>(0, 0);
		}

		matrix& mul_by_transpose(matrix &b)
		{
			/*if (b.NumCols() == 1 && b.NumRows() == 1)
			{
			//is a 1x1 matrix treated like a scalar?
			return (*this) * b(0, 0);
			}*/
			if (this->NumCols() == b.NumCols())
			{
				//if (out) delete out;
				//out = new matrix(this->NumRows(), b.NumCols());
				if (out)
				{
					if (!((out->NumRows() == this->NumRows()) &&
						(out->NumCols() == b.NumRows())))
					{
						delete out;
						out = new matrix(this->NumRows(), b.NumRows());
					}
				}
				else
				{
					out = new matrix(this->NumRows(), b.NumRows());
				}

				for (int i = 0; i < this->NumRows(); i++)
				{

					for (int j = 0; j < b.NumRows(); j++)
					{
						(*out)(i, j) = 0.0;
						for (int k = 0; k < this->NumCols(); k++)
						{
							(*out)(i, j) += get(i, k) * b(j, k);
						}
					}
				}

				return *out;
			}
			return matrix<T>(0, 0);
		}

		matrix& add_transposed(matrix &b)
		{
			if (this->NumRows() != b.NumCols() || this->NumCols() != b.NumRows())
			{
				return matrix<T>(0, 0);
			}
			else
			{
				//if (out)delete out;
				//out = new matrix(this->NumRows(), this->NumCols());
				if (out)
				{
					if (!((out->NumRows() == this->NumCols()) &&
						(out->NumCols() == this->NumRows())))
					{
						delete out;
						out = new matrix(this->NumCols(), this->NumRows());
					}
				}
				else
				{
					out = new matrix(this->NumCols(), this->NumRows());
				}

				for (int i = 0; i < this->NumCols(); i++)
				{
					for (int j = 0; j < this->NumRows(); j++)
					{
						(*out)(i, j) = this->get(j, i) + b.get(i, j);
					}
				}

				return *out;
			}
			return matrix(0, 0);
		}

		matrix& sub_transposed(matrix &b)
		{
			if (this->NumRows() != b.NumCols() || this->NumCols() != b.NumRows())
			{
				return matrix<T>(0, 0);
			}
			else
			{
				//if (out)delete out;
				//out = new matrix(this->NumRows(), this->NumCols());
				if (out)
				{
					if (!((out->NumRows() == this->NumCols()) &&
						(out->NumCols() == this->NumRows())))
					{
						delete out;
						out = new matrix(this->NumCols(), this->NumRows());
					}
				}
				else
				{
					out = new matrix(this->NumCols(), this->NumRows());
				}

				for (int i = 0; i < this->NumCols(); i++)
				{
					for (int j = 0; j < this->NumRows(); j++)
					{
						(*out)(i, j) = this->get(j, i) - b.get(i, j);
					}
				}

				return *out;
			}
			return matrix(0, 0);
		}

		// may need to override this for different matrix types
		void print(int precis)
		{
			if (this->NumRows() == 0 || this->NumCols() == 0 || this->data == 0)
			{
				cout << "Empty Matrix" << endl;
				return;
			}
			for (int i = 0; i < this->NumRows(); i++)
			{
				for (int j = 0; j < this->NumCols(); j++)
				{
					T f = get(i, j);
					//cout << f.5 << "  ";
					if (precis == -1)
					{
						cout << f << "  ";
					}
					if (precis == 2)
						printf("%8.2f    ", f);
					else if (precis == 3)
						printf("%9.3f ", f);
					else if (precis == 4)
						printf("%10.4f    ", f);
					else if (precis == 5)
						printf("%11.5f    ", f);
					else if (precis == 6)
						printf("%12.6f    ", f);
				}
				cout << endl;
			}
		}



		T trace()
		{
			T sum = 0.0;
			if (SX != SY) return 0.0;

			for (int i = 0; i < this->NumRows(); i++)
			{
				sum += get(i, i);
			}
			return sum;
		}



		bool ContainsNAN()
		{
			for (int i = 0; i < SX; i++)
			{
				for (int j = 0; j < SY; j++)
				{
					if (get(i, j) != get(i, j))
						return true;
				}
			}
			return false;
		}

		void transpose()
		{
			if (this->IsSquare())
			{
				for (int i = 0; i < SX; i++)
				{
					for (int j = i; j < SY; j++)
					{
						SWAP<T>(get(i, j), get(j, i));
						//if (get(i, j) != get(i, j))
						//	return true;
					}
				}
			}
			else if (this->NumCols() == 1 || this->NumRows() == 1) 
			{
				// special case of transposition of a vector
				// this is faster because the matrix in memory is the same 
				// after reallocation, only the indexes need change.
				int temp = SX;
				SX = SY;
				SY = temp;
			}
			else
			{
				matrix<T> Y(this->NumCols(), this->NumRows());

				for (int i = 0; i < SX; i++)
				{
					for (int j = 0; j < SY; j++)
					{
						Y(j, i) = get(i, j);
					}
				}
				this->destroy();
				this->SX = Y.NumRows();
				this->SY = Y.NumCols();
				this->create();

				for (int i = 0; i < SX; i++)
				{
					for (int j = 0; j < SY; j++)
					{
						get(i, j) = Y(i, j);
					}
				}

				is_transposed = !is_transposed;
			}
		}



		inline unsigned int NumRows()
		{
			return SX;
			//return  (is_transposed ? SY : SX);
		}
		inline unsigned int NumCols()
		{
			return SY;
			//return (is_transposed ? SX : SY);
		}

		inline bool IsSquare()
		{
			return SX == SY;
		}

		bool IsSymmetric()
		{
			if (!this->IsSquare())
			{
				cout << "Error (IsSymmetric): matrix must be square " << endl;
				return false;
			}

			for (int i = 0; i < this->NumRows(); i++)
			{
				for (int j = 0; j <= i; j++)
				{
					// if get(i,j) is a NAN then this will accidently return false
					if (i != j)
						if (NotEqual(get(i, j), get(j, i)))
							return false;
				}
			}

			return true;
		}

		bool IsSkewSymmetric()
		{
			if (!this->IsSquare())
			{
				cout << "Error (IsSymmetric): matrix must be square " << endl;
				return false;
			}

			for (int i = 0; i < this->NumRows(); i++)
			{
				for (int j = 0; j <= i; j++)
				{
					if (i != j)
						if (NotEqual(get(i, j), -get(j, i)))
							return false;
				}
			}

			return true;
		}

		bool IsPositiveDefinite()
		{
			if (!this->IsSquare())
			{
				cout << "Error: Positive Definite Matrixces must be square" << endl;
				return false;
			}

			matrix X(this->NumRows(), 1);

			for (int i = 0; i < this->NumRows(); i++)
				X(i, 0) = RandomFloat(1, 5);

			matrix XT = X;
			XT.transpose();

			matrix res = XT * (*this) * X;

			for (int i = 0; i < res.NumCols(); i++)
			{
				for (int j = 0; j < res.NumRows(); j++)
				{
					if (res(j, i) <= 0) return false;
				}
			}
			return true;
		}

		//============================================================================
		// For solution of the system Ax = b, 
		// relies on the condition that this matrix A is square and 
		// that b has the same number of rows as A
		//
		// inputs b, outputs x, 
		//
		// For more information please consult Krezig: Advanced Engineering Mathematics, sec 19.1
		//
		//============================================================================
		matrix& Gauss_Elimination(matrix &b)
		{
			if (!this->IsSquare())
			{
				cout << "Error in Gauss calculation: System Matrix must be square." << endl;
				return b;
			}

			if (b.NumRows() != this->NumRows())
			{
				cout << "Error in Gauss calculation: b must have the same number of rows as A" << endl;
				return b;
			}

			unsigned int n = b.NumRows();

			for (int k = 0; k < n - 1; k++)
			{
				bool bSolutionExists = false;
				unsigned int j = k + 1;
				for (j = k + 1; j < n; j++)
				{
					if (get(j, k) != 0)
					{
						bSolutionExists = true;
						break;
					}
				}

				if (!bSolutionExists)
				{
					cout << "No Unique Solution Exists" << endl;
					return b; // no solution
				}

				for (int i = 0; i < n; i++)
				{
					SWAP(get(j, i), get(k, i));

					SWAP(b(j, 0), b(k, 0));
				}
				for (j = k + 1; j < n; j++)
				{
					T mjk = get(j, k) / get(k, k);
					for (int p = k; p < n; p++)
					{
						get(j, p) = get(j, p) - mjk * get(k, p);
					}
					b(j, 0) = b(j, 0) - mjk * b(k, 0);
				}
			}
			if (get(n - 1, n - 1) == 0)
			{
				cout << "No Unique Solution Exists" << endl;
				return b; // no solution
			}

			if (out) delete out;
			out = new matrix(n, 1);

			(*out)(n - 1, 0) = b(n - 1, 0) / get(n - 1, n - 1);

			for (int i = n - 2; i > -1; i--)
			{
				T the_sum = 0;
				for (int j = i + 1; j < n; j++)
					the_sum += get(i, j)*(*out)(j, 0);

				(*out)(i, 0) = (1 / get(i, i)) * (b(i, 0) - the_sum);
			}

			return *out;
		}


		//============================================================================
		// For solution of the system Ax = b, 
		// relies on the condition that this matrix A is square and 
		// that b has the same number of rows as A
		//
		// inputs b, outputs x, 
		//
		// For more information please consult Krezig: Advanced Engineering Mathematics, sec 19.1
		//
		//============================================================================
		matrix& Gauss_Elimination_Save_Original(matrix &b)
		{
			if (!this->IsSquare())
			{
				cout << "Error in Gauss calculation: System Matrix must be square." << endl;
				return b;
			}

			if (b.NumRows() != this->NumRows())
			{
				cout << "Error in Gauss calculation: b must have the same number of rows as A" << endl;
				return b;
			}

			matrix X = (*this);

			unsigned int n = b.NumRows();

			for (int k = 0; k < n - 1; k++)
			{
				bool bSolutionExists = false;
				unsigned int j = k + 1;
				for (j = k + 1; j < n; j++)
				{
					if (X(j, k) != 0)
					{
						bSolutionExists = true;
						break;
					}
				}

				if (!bSolutionExists)
				{
					cout << "No Unique Solution Exists" << endl;
					return b; // no solution
				}

				for (int i = 0; i < n; i++)
				{
					SWAP(X(j, i), X(k, i));

					SWAP(b(j, 0), b(k, 0));
				}
				for (j = k + 1; j < n; j++)
				{
					T mjk = X(j, k) / X(k, k);
					for (int p = k; p < n; p++)
					{
						X(j, p) = X(j, p) - mjk * X(k, p);
					}
					b(j, 0) = b(j, 0) - mjk * b(k, 0);
				}
			}
			if (X(n - 1, n - 1) == 0)
			{
				cout << "No Unique Solution Exists" << endl;
				return b; // no solution
			}

			if (out) delete out;
			out = new matrix(n, 1);

			(*out)(n - 1, 0) = b(n - 1, 0) / X(n - 1, n - 1);

			for (int i = n - 2; i > -1; i--)
			{
				T the_sum = 0;
				for (int j = i + 1; j < n; j++)
					the_sum += X(i, j)*(*out)(j, 0);

				(*out)(i, 0) = (1 / X(i, i)) * (b(i, 0) - the_sum);
			}

			return *out;
		}

		// generalization of the reduction to triangular form above
		bool ReduceToUpperTriangularForm(T &sign)
		{
			if (!this->IsSquare())
			{
				cout << "Error (ReduceToLowerTriangularForm) : System Matrix must be square." << endl;
				return false;
			}

			unsigned int n = this->NumRows();

			sign = 1;

			// this part is almost exactly the same as the section from the Gauss method 
			for (int k = 0; k < n - 1; k++)
			{
				bool bSolutionExists = false;
				unsigned int j = n - 1;
				for (j = k + 1; j < n; j++)
				{
					if (get(j, k) != 0)
					{
						bSolutionExists = true;
						break;
					}
				}

				if (!bSolutionExists) return false;

				for (int i = 0; i < n; i++)
				{
					SWAP(get(j, i), get(k, i));
					sign = -sign;
				}

				for (j = k + 1; j < n; j++)
				{
					T mjk = get(j, k) / get(k, k);
					for (int p = k; p < n; p++)
					{
						get(j, p) = get(j, p) - mjk * get(k, p);
					}
				}
			}
		}



		T Det_2x2(int r, int c)
		{
			if ((r + 1 < NumRows()) &&
				(c + 1 < NumCols()))
			{
				return get(r, c) * get(r + 1, c + 1) - get(r + 1, c)*get(r, c + 1);
			}
			return 0.0;
		}
	private:






		// C(n,k) = C(5,4) = 5*4*3*2*1 / ((4*3*2*1) * (5-4)!) = 5 
		inline T Det_5x5_internal(int r1, int r2, int r3, int r4, int r5, int c)
		{
			T c1 = get(r1, c);
			T c2 = -get(r2, c);
			T c3 = get(r3, c);
			T c4 = -get(r4, c);
			T c5 = get(r5, c);

			return c1 * Det_4x4_internal(r2, r3, r4, r5, c + 1) +
				c2 * Det_4x4_internal(r1, r3, r4, r5, c + 1) +
				c3 * Det_4x4_internal(r1, r2, r4, r5, c + 1) +
				c4 * Det_4x4_internal(r1, r2, r3, r5, c + 1) +
				c5 * Det_4x4_internal(r1, r2, r3, r4, c + 1);
		}

		inline T Det_4x4_internal(int r1, int r2, int r3, int r4, int c)
		{
			T c1 = get(r1, c);
			T c2 = -get(r2, c);
			T c3 = get(r3, c);
			T c4 = -get(r4, c);

			return c1 * Det_3x3_internal(r2, r3, r4, c + 1) +
				c2 * Det_3x3_internal(r1, r3, r4, c + 1) +
				c3 * Det_3x3_internal(r1, r2, r4, c + 1) +
				c4 * Det_3x3_internal(r1, r2, r3, c + 1);

		}
		inline T Det_3x3_internal(int r1, int r2, int r3, int c)
		{
			T c1 = get(r1, c);
			T c2 = -get(r2, c);
			T c3 = get(r3, c);

			return c1 * Det_2x2_internal(r2, r3, c + 1) +
				c2 * Det_2x2_internal(r1, r3, c + 1) +
				c3* Det_2x2_internal(r1, r2, c + 1);


		}
		inline T Det_2x2_internal(int r1, int r2, int c)
		{
			return get(r1, c) * get(r2, c + 1) - get(r2, c) * get(r1, c + 1);
		}
	public:
		T DiagonalEntryProduct()
		{
			if (!this->IsSquare())
			{
				cout << "Error (DiagonalEntryProduct): error matrix should be square" << endl;
				return 0.0;
			}
			T prod = get(0, 0);
			for (int r = 1; r < this->NumRows(); r++)
				prod *= get(r, r);

			return prod;
		}



		// interesting way of evaluating the determinant
		T Determinant()
		{
			//	if (SX == 2 && SY == 2) return Det_2x2(0, 0);
			//	if (SX == 3 && SY == 3) return Det_3x3(0, 0);
			matrix temp = (*this);
			T sign = 1;
			if (temp.ReduceToUpperTriangularForm(sign))
				return sign*temp.DiagonalEntryProduct();

			return 0;
		}

		//============================================================================
		// Matrix Inversion via Gauss-Jordan elimination
		// The matrix should be square, returns the inverse
		//
		//============================================================================
		matrix& Gauss_Jordan()
		{
			if (!this->IsSquare())
			{
				cout << "Error in Gauss calculation: System Matrix must be square." << endl;
				return (*this);
			}


			unsigned int n = this->NumRows();

			/*if (out)
			{
			if (!((out->NumRows() == this->NumRows()) &&
			(out->NumCols() == this->NumCols())))
			{
			delete out;
			out = new matrix(this->NumRows(), this->NumCols());
			}
			}
			else
			{
			out = new matrix(this->NumRows(), this->NumCols());
			}*/
			if (out) delete out;
			out = new matrix<T>(this->NumRows(), this->NumCols());

			out->Identity();


			// this part is almost exactly the same as the section from the Gauss method 
			for (int k = 0; k < n - 1; k++)
			{
				bool bSolutionExists = false;
				unsigned int j = n - 1;
				for (j = k + 1; j < n; j++)
				{
					if (get(j, k) != 0)
					{
						bSolutionExists = true;
						break;
					}
				}

				if (!bSolutionExists)
				{
					cout << "Error (Gauss_Jordan): No Unique Solution Exists." << endl;
					return (*this);
				}

				for (int i = 0; i < n; i++)
				{
					SWAP(get(j, i), get(k, i));

					SWAP((*out)(j, i), (*out)(k, i));
				}
				for (j = k + 1; j < n; j++)
				{
					T mjk = get(j, k) / get(k, k);

					for (int p = k; p < n; p++)
					{

						get(j, p) = get(j, p) - mjk * get(k, p);

						(*out)(j, p) = (*out)(j, p) - mjk * (*out)(k, p);
					}
					//b(j, 0) = b(j, 0) - mjk * b(k, 0);
				}


			}

			// make sure all the elements on the diagonal are 1 by dividing all 
			// rows by the elements in the diagonals of that row
			// need this for the next loop
			for (int k = 0; k < n; k++)
			{
				T val = get(k, k);
				for (int c = 0; c < n; c++)
				{
					get(k, c) = get(k, c) / val;
					(*out)(k, c) = (*out)(k, c) / val;
				}
			}

			// starting at the n-1 th row, we have
			// a matrix in echelon form with ones on the main diagonal
			// the 
			for (int r = n - 2; r >-1; r--) // n-2 here because we index from zero instead of one
			{


				// the next element after the diagonal is Row+1
				for (int c = r + 1; c < n; c++)
				{
					// get the value in the row / column position, we will make the element in (r,c) into zero
					// by multiplying by the *next* row, elemnts before the diagonal are already zero, and 
					// the next elements in this row will be handled next ...
					// ...
					T val = get(r, c);
					for (int p = 0; p < n; p++)
					{
						// we use c to use a mulitiple of c's row, subtracting a multiple of the 1's on the diagonal
						get(r, p) = get(r, p) - get(c, p)*val;

						// all operations are mirrored on the other *output* matrix, or the inverse
						(*out)(r, p) = (*out)(r, p) - (*out)(c, p)*val;
					}


				}

			}


			return *out;

		}

		//============================================================================
		// Solve a matrix system of the form Ly = b by back substitution
		// L, is Lower triangular, y is the solution vector, b is the output
		// inputs L, y, b
		//============================================================================
		matrix& Solve_Lower_TriangularSystem(matrix& L, matrix & y, matrix& b)
		{
			if (!L.IsSquare())
			{
				cout << "Error (Solve_Lower_TriangularSystem): Lower Triangular Matrix should be square" << endl;
				return y;
			}

			if (!y.NumRows() == L.NumRows()) {
				cout << "Error (Solve_Lower_TriangularSystem): vector y should have the same number of rows as L" << endl;
				return y;
			}
			if (!b.NumRows() == y.NumRows())
			{
				cout << "Error (Solve_Lower_TriangularSystem): vector y should have the same number of rows as b" << endl;
				return y;
			}

			y(0, 0) = b(0, 0) / L(0, 0);

			for (int i = 1; i < L.NumRows(); i++)
			{
				T sum0 = 0.0;
				for (int s = 0; s < i; s++)
				{
					sum0 += L(i, s) * y(s, 0);
				}
				y(i, 0) = (1 / L(i, i)) * (b(i, 0) - sum0);
			}

			return y;
		}

		//============================================================================
		// Solve a system of the form Ux = y by back substitution
		// U is upper triangular, x is the solution vector, y is the output
		// inputs U,x,y
		//============================================================================
		matrix& Solve_Upper_TriangularSystem(matrix& U, matrix & x, matrix& y)
		{
			if (!U.IsSquare()) {
				cout << "Error (Solve_Upper_TriangularSystem): matrix should be square" << endl;
				return y;
			}
			if (!y.NumRows() == U.NumRows())
			{
				cout << "Error (Solve_Upper_TriangularSystem): vector y should have same number of rows as matrix" << endl;
				return y;
			}
			if (!x.NumRows() == y.NumRows()) {
				cout << "Error (Solve_Upper_TriangularSystem): x should be same shape as y" << endl;
				return y;
			}

			int n = x.NumRows();

			x(n - 1, 0) = y(n - 1, 0) / U(n - 1, n - 1);

			for (int i = n - 2; i > -1; i--)
			{
				T sum0 = 0;
				for (int j = i + 1; j < n; j++)
					sum0 += U(i, j)*x(j, 0);

				x(i, 0) = (1 / U(i, i)) * (y(i, 0) - sum0);
			}

			return x;
		}

		//============================================================================
		// Solves a system Ax = b via the Cholesky method
		// this matrix must be symmetric and positive definite
		// inputs b, output vector
		// returns x, solution vector
		//============================================================================
		matrix& Cholesky(matrix& b)
		{
			if (!this->IsSymmetric()) {
				cout << "Error (Cholesky): Matrix should be symetric" << endl;
				return b;
			}
			if (!this->IsPositiveDefinite())
			{
				cout << "Error (Cholesky): Matrix should be positive definite" << endl;
				return b;
			}

			int n = this->NumRows();
			matrix M(n, n);

			M(0, 0) = sqrt(get(0, 0));


			for (int j = 1; j < n; j++)
			{
				M(j, 0) = get(j, 0) / M(0, 0);
			}

			for (int k = 1; k < this->NumCols(); k++)
			{
				for (int j = k; j < n; j++)
				{
					if (j == k)
					{
						T sum1 = 0;
						for (int s = 0; s < j; s++) sum1 += M(j, s) * M(j, s);

						M(j, j) = sqrt(get(j, j) - sum1);
					}
					else
					{
						T sum2 = 0;
						for (int s = 0; s < k; s++) sum2 += M(j, s) * M(k, s);

						M(j, k) = (1 / M(k, k)) * (get(j, k) - sum2);
					}
				}
			}

			matrix y(n, 1);
			Solve_Lower_TriangularSystem(M, y, b);
			M.transpose();

			if (out) delete out;
			out = new matrix(n, 1);

			Solve_Upper_TriangularSystem(M, *out, y);

			return *out;
		}

		inline bool EqualSize(matrix& b){ return (this->NumRows() == b.NumRows()) && (this->NumCols() == b.NumCols()); }

		void Identity()
		{
			if (!this->IsSquare())
			{
				cout << "Error (Identity): Identity Matrix must be square" << endl;
				return;
			}
			for (int i = 0; i < this->NumCols(); i++)
			{
				for (int j = 0; j < this->NumRows(); j++)
				{
					if (i == j)
					{
						get(j, i) = 1.0;
					}
					else
					{
						get(j, i) = 0.0;
					}

				}
			}
		}

		// TODO: optimize this...
		// thanks to
		// http://mathworld.wolfram.com/MatrixInverse.html
		// and 
		// http://www.gamedev.net/page/resources/_/technical/math-and-physics/matrix-inversion-using-lu-decomposition-r3637
		// for this solution 
		matrix& Invert_Crout()
		{
			int n = this->NumCols();

			matrix A_inv(n, n);

			matrix b(n, 1);

			for (int c = 0; c < n; c++)
			{
				//b.ToZero();
				b(c, 0) = 1.0;

				matrix sol = Solve_System_Crout(b);

				for (int r = 0; r < n; r++)
				{
					A_inv(r, c) = sol(r, 0);
				}

				b(c, 0) = 0.0;
			}

			(*out) = A_inv;

			return (*out);

		}


		/* UNTESTED */
		matrix& Invert_Gauss()
		{
			if (!this->IsSquare())
			{
				cout << "Error (Invert_Gauss): matrix should be square" << endl;
				return matrix(0, 0);
			}


			int n = this->NumCols();
			matrix sol(n, 1);
			matrix I_Col(n, 1);
			matrix X(n, n);

			for (int i = 0; i < n; i++)
			{
				I_Col(i, 0) = 1.0;
				sol = Gauss_Elimination_Save_Original(I_Col);
				I_Col(i, 0) = 0.0;

				for (int j = 0; j < n; j++)
					X(j, i) = sol(j, 0);
			}

			if (out) delete out;
			out = X;

			/* UNTESTED */
			return (*out);
		}

		// note for this method, inv can be arbitrary iff every eigenvalue of (I-AX(0)) is of absolute value < 1
		// the method is mostly used for improving an innaccurate inverse obtained by another method.

		matrix& Newtons_Iteration_for_Inverse(matrix& inv, int num_iterations)
		{
			if (!inv.IsSquare() || !this->IsSquare())
			{
				cout << "Error (Newtons_Iteration_for_Inverse): Matrices should be square" << endl;
			}

			int n = inv.NumCols();

			// need an identity matrix
			matrix I(n, n);
			I.Identity();

			for (int i = 0; i < num_iterations; i++)
			{
				// incrementally update the inverse ...
				// could possibly test for convergence here using a norm
				inv = inv * (I * 2 - (*this) * inv);
			}

			return inv;
		}

		//============================================================================
		// The LU decomposition using doolittles method, this function
		// does not solve a system, it simple decompses this matrix into two
		//matrices such that A = LU
		//============================================================================
		int LU_Decomposition_Doolittle(matrix& L, matrix& U)
		{
			if (!L.EqualSize(U))
			{
				cout << "Error (LU_Decomposition_Doolittle): L and U must be equal size" << endl;
				return 0;
			}
			if (!this->EqualSize(L))
			{
				cout << "Error (LU_Decomposition_Doolittle):  L and U must be the same size as the source matrix" << endl;
				return 0;
			}
			if (!this->IsSquare())
			{
				cout << "Error (LU_Decomposition_Doolittle):  source matrix must be square" << endl;
				return 0;
			}



			int n = this->NumCols();

			L.Identity();

			for (int k = 0; k < n; k++)
			{
				U(0, k) = get(0, k);

				L(k, 0) = get(k, 0) / U(0, 0); // loop 1 with rows indexed with k, ignore cols

				L(k, k) = 1.0; // loop 1 as L(k,k)
			}

			for (int j = 1; j < n; j++)
			{
				for (int k = 0; k < n; k++)
				{
					if (k >= j)
					{
						T sum0 = 0.0;
						for (int s = 0; s < j; s++) sum0 += L(j, s) * U(s, k);

						U(j, k) = get(j, k) - sum0;
					}
					else
					{
						T sum1 = 0.0;
						for (int s = 0; s < k; s++) sum1 += L(j, s) * U(s, k);

						L(j, k) = (1 / U(k, k))*(get(j, k) - sum1);
					}

				}
			}

			return 1;
		}

		//============================================================================
		// Solves the system via doolittles method, for the equation Ax = b
		// it first factorizes A into LU such that A=LU, then solves the 
		// lower triangular system Ly = b and then the upper triangular system Ux=y
		// returns x
		//
		//============================================================================
		matrix& Solve_System_Doolittle(matrix& b)
		{

			if (!this->IsSquare())
			{
				cout << "Error (Solve_System_Doolittle):  source matrix must be square" << endl;
				return b;
			}
			if (!(b.NumRows() == this->NumRows()))
			{
				cout << "Error (Solve_System_Doolittle):  matrix b must have same number of rows as source matrix" << endl;
				return b;
			}


			int n = this->NumCols();

			matrix L(n, n);
			matrix U(n, n);


			this->LU_Decomposition_Doolittle(L, U);

			matrix y(n, 1);
			Solve_Lower_TriangularSystem(L, y, b);

			if (out) delete out;
			out = new matrix(n, 1);

			Solve_Upper_TriangularSystem(U, *out, y);

			return *out;
		}


		//============================================================================
		// Very similar to the Doolittle method, solves A = LU for L and U
		// inputs require L and U to be of the same dimensionality as A
		//============================================================================
		int LU_Decomposition_Crout(matrix& L, matrix& U)
		{
			if (!L.EqualSize(U))
			{
				cout << "Error (LU_Decomposition_Doolittle): L and U must be equal size" << endl;
				return 0;
			}
			if (!this->EqualSize(L))
			{
				cout << "Error (LU_Decomposition_Doolittle):  L and U must be the same size as the source matrix" << endl;
				return 0;
			}
			if (!this->IsSquare())
			{
				cout << "Error (LU_Decomposition_Doolittle):  source matrix must be square" << endl;
				return 0;
			}



			int n = this->NumCols();

			U.Identity();

			for (int j = 0; j < n; j++)
			{
				L(j, 0) = get(j, 0);

				U(0, j) = get(0, j) / L(0, 0);

				U(j, j) = 1.0;

			}

			for (int k = 1; k < n; k++)
			{
				for (int j = 0; j < n; j++)
				{
					if (j >= k)
					{
						T sum0 = 0.0;
						for (int s = 0; s < k; s++) sum0 += L(j, s) * U(s, k);

						L(j, k) = get(j, k) - sum0;
					}
					else
					{
						T sum1 = 0.0;
						for (int s = 0; s < j; s++) sum1 += L(j, s) * U(s, k);

						U(j, k) = (1 / L(j, j))*(get(j, k) - sum1);
					}
				}
			}

			return 1;
		}

		//============================================================================
		// Solves the system via Crout method, for the equation Ax = b
		// it first factorizes A into LU such that A=LU, then solves the
		// lower triangular system Ly = b and then the upper triangular system Ux=y
		// returns x
		//
		//============================================================================
		matrix& Solve_System_Crout(matrix& b)
		{

			if (!this->IsSquare())
			{
				cout << "Error (Solve_System_Doolittle):  source matrix must be square" << endl;
				return b;
			}
			if (!(b.NumRows() == this->NumRows()))
			{
				cout << "Error (Solve_System_Doolittle):  matrix b must have same number of rows as source matrix" << endl;
				return b;
			}


			int n = this->NumCols();

			matrix L(n, n);
			matrix U(n, n);


			this->LU_Decomposition_Crout(L, U);

			matrix y(n, 1);
			Solve_Lower_TriangularSystem(L, y, b);

			if (out) delete out;
			out = new matrix(n, 1);

			Solve_Upper_TriangularSystem(U, *out, y);

			return *out;
		}

		//============================================================================
		// Solves the system via Doolittles or Crouts method, for the equation Ax = b
		// If the matrices L and U satisfy A = LU (where A is the system matix)
		// it solves the lower triangular system Ly = b and then the upper triangular system Ux=y
		// returns x
		//
		//============================================================================
		matrix& Solve_System_LU(matrix& L, matrix& U, matrix& b)
		{

			if (!this->IsSquare())
			{
				cout << "Error (Solve_System):  source matrix must be square" << endl;
				return b;
			}
			if (!L.EqualSize(U))
			{
				cout << "Error (Solve_System): L and U must be equal size" << endl;
				return b;
			}
			if (!this->EqualSize(L))
			{
				cout << "Error (Solve_System):  L and U must be the same size as the source matrix" << endl;
				return b;
			}

			int n = this->NumCols();

			//this->LU_Decomposition_Crout(L, U);

			matrix y(n, 1);
			Solve_Lower_TriangularSystem(L, y, b);

			if (out) delete out;
			out = new matrix(n, 1);

			Solve_Upper_TriangularSystem(U, *out, y);

			return *out;
		}

		//============================================================================
		// For solution of the system Ax = b, relies on the condition
		// that this matrix A is square and that b has the same number of rows as A
		//
		// inputs b, initial estimation x0, tolerance (eps > 0) small number, MAX_ITERATIONS for number of loops,
		// outputs x or reports failure and outputs ,
		//
		// For more information please consult Krezig: Advanced Engineering Mathematics, 
		//
		//============================================================================
		matrix & Gauss_Seidel(matrix& b, matrix& x0, float tolerance, unsigned int MAX_ITERATIONS)
		{
			if (x0.NumRows() != this->NumRows())
			{
				cout << "Gauss Seidel error: x0 must have the same number of rows as this matrix" << endl;
				return x0;
			}

			if (b.NumRows() != this->NumRows())
			{
				cout << "Gauss Seidel error: b must have the same number of rows as this matrix" << endl;
				return x0;
			}

			int n = this->NumRows();

			if (out) delete out;
			out = new matrix(n, 1);

			for (int m = 0; m < MAX_ITERATIONS; m++)
			{
				for (int j = 0; j < n; j++)
				{
					T sum1 = 0.0;
					for (int k = 0; k < j; k++)
					{
						sum1 += get(j, k) * (*out)(k, 0);
					}
					T sum2 = 0.0;
					for (int k = j + 1; k < n; k++)
					{
						sum2 += get(j, k) * x0(k, 0);
					}

					(*out)(j, 0) = (-1 / get(j, j))*(sum1 + sum2 - b(j, 0));
				}
				float max_magnitude = 0.0;
				for (int j = 0; j < n; j++)
				{
					float magnitude = abs((*out)(j, 0) - x0(j, 0));
					if (magnitude > max_magnitude) max_magnitude = magnitude;

					x0(j, 0) = (*out)(j, 0);
				}
				if (max_magnitude < tolerance) return *out;

			}
			cout << "No output satisfying the tolerance condition obtained after N iteration steps." << endl;
			return x0;
		}



		void Householder_Tridiagonalize()
		{
			/*if (!this->IsSymmetric())
			{
			cout << "Error (Householder_Tridiagonalize): matrix must be symetric" << endl;
			}*/
			int n = this->NumRows();

			matrix<T>  V(n, 1);
			matrix<T>  VT(1, n);
			matrix<T>  Inn(n, n);
			Inn.Identity();

			matrix<T> H_ = (*this);
			matrix<T> P;

			T S = 0.0;
			for (int c = 0; c < n - 2; c++)
			{
				//H = (*this);

				S = 0.0;
				for (int r = c + 1; r < n; r++)
				{
					S += H_(r, c)*H_(r, c);
				}

				S = sqrt(S);
				for (int r = c + 1; r < n; r++)
				{

					if (r == c + 1)
					{
						V(r, 0) = sqrt(0.5*(1.0 + abs(H_(r, c)) / S));

					}
					else if (r > c + 1)
					{
						V(r, 0) = H_(r, c) * sgn(H_(c + 1, c)) / (2.0 * V(c + 1, 0)*S);
					}

					VT(0, r) = V(r, 0);
				}

				// anotH_er copy 
				P = Inn - V * VT * 2;

				H_ = H_ * P;
				H_ = P * H_;

				//(*this) = P * (*this) * P;

				// zero the vectors again
				V.ToZero();
				VT.ToZero();

			}

			(*this) = H_;

		}


		matrix& Eigenvalues_PowerMethod(int max_iterations)
		{
			if (!this->IsSquare())
			{
				cout << "Error (Eigenvalues_PowerMethod): matrix must be square" << endl;
				return *this;
			}

			int n = this->NumCols();

			if (out) delete out;
			out = new matrix(n, 1);

			matrix V(n, 1);
			for (int i = 0; i < n; i++)
				V(i, 0) = 1;

			for (int i = 0; i < n; i++)
			{
				if (i == 0)
				{
					T last_value = 0;
					for (int k = 0; k < max_iterations; k++)
					{
						V = (*this) * V;

						T max_value = 0;
						for (int j = 0; j < n; j++)
							if (V(j, 0) > max_value)
								max_value = V(j, 0);

						for (int j = 0; j < n; j++)
							V(j, 0) = V(j, 0) / max_value;

						if (max_value == last_value)
						{
							out(0, 0) = max_value;
							break;
						}
					}
				}

				/*	else if ( i b  )
				{

				}*/
			}
			return *out;
		}



		// note : this needs to be rewritten. 
		// this should be rewritten for complex types too.
		// also a tolerance should be built in to all tests for equality.
		bool IsOrthogonal()
		{
			if (!this->IsSquare())
			{
				cout << "Error (IsOrthogonal): Matrix should be square" << endl;
				return false;
			}
			matrix<T> AT = (*this);

			AT.transpose();

			matrix<T> I = AT * (*this);

			for (int i = 0; i < this->NumRows(); i++)
				for (int j = 0; j < this->NumRows(); j++)
					if ((i == j && NotEqual(I(i, j), 1.0)) ||
						(i != j && NotEqual(I(i, j), 0.0)))
						return false;

			/*matrix<T> I2 = (*this) * AT;

			for (int i = 0; i < this->NumRows(); i++)
			for (int j = 0; j < this->NumRows(); j++)
			if ((i == j && NotEqual( I2(i, j), 1.0)) ||
			(i != j && NotEqual( I2(i, j), 0.0)))
			return false;
			*/

			return true;

		}

		void Overwrite_Submatrix(matrix<T> b, int r, int c)
		{
			if (b.NumCols() > this->NumCols() || b.NumRows() > this->NumRows())
			{
				cout << "Error (Overwrite_Submatrix): sub matrix dimensions exceed destination dimensions" << endl;
				return;
			}

			if ((r + b.NumRows() > this->NumRows()) || (c + b.NumCols() > this->NumCols()))
			{
				cout << "Error (Overwrite_Submatrix): sub matrix size plus dimensions exceed destination dimensions" << endl;
				return;
			}

			for (int i = 0; i < b.NumRows(); i++)
			{
				for (int j = 0; j < b.NumCols(); j++)
				{
					get(r + i, c + j) = b(i, j);
				}
			}

		}

		void Overwrite_Submatrix_transposed(matrix<T> b, int r, int c)
		{
			if (b.NumCols() > this->NumCols() || b.NumRows() > this->NumRows())
			{
				cout << "Error (Overwrite_Submatrix): sub matrix dimensions exceed destination dimensions" << endl;
				return;
			}

			if ((r + b.NumCols() > this->NumRows()) || (c + b.NumRows() > this->NumCols()))
			{
				cout << "Error (Overwrite_Submatrix): sub matrix size plus dimensions exceed destination dimensions" << endl;
				return;
			}

			for (int i = 0; i < b.NumRows(); i++)
			{
				for (int j = 0; j < b.NumCols(); j++)
				{
					get(r + j, c + i) = b(i, j);
				}
			}

		}

		void Eigenvalues_2x2(int r, int c, complex<T> &L1, complex<T> &L2)
		{
			if (r + 1 >= this->NumRows() || c + 1 >= this->NumCols())
			{
				cout << "Error (Eigenvalues_2x2): index out of bounds" << endl;
				return;// false;
			}

			T trace = get(r, c) + get(r + 1, c + 1);
			T det = get(r, c)*get(r + 1, c + 1) - get(r + 1, c)*get(r, c + 1);

			L1 = complex<T>(trace * 0.5, 0) + std::sqrt(complex<T>(trace*trace / 4.0 - det));
			L2 = complex<T>(trace * 0.5, 0) - std::sqrt(complex<T>(trace*trace / 4.0 - det));
		}



		T Det_3x3(int r, int c)
		{
			if (r + 2 >= NumRows() || c + 2 >= NumCols())
			{
				cout << "Error (Det_3x3): Out of bounds error" << endl;
				return 0.0;
			}

			T c1 = get(r, c);
			T c2 = -get(r, c + 1);
			T c3 = get(r, c + 2);

			return  c1 * (get(r + 1, c + 1) * get(r + 2, c + 2) - get(r + 1, c + 2)*get(r + 2, c + 1)) +
				c2 * (get(r + 1, c) * get(r + 2, c + 2) - get(r + 1, c + 2) * get(r + 2, c)) +
				c3 * (get(r + 1, c) * get(r + 2, c + 1) - get(r + 2, c) * get(r + 1, c + 1));

		}


		// anayltic solution from wikipedias
		void EigenValues_3x3(complex<T> &L1, complex<T> &L2, complex<T> &L3)
		{
			if (!this->IsSymmetric())
			{
				cout << "Error (EigenValues3x3): Matrix must be symmetric" << endl;
				return;
			}

#define A (*this)

			T p1 = A(0, 1) *A(0, 1) + A(0, 2) *A(0, 2) + A(1, 2) *A(1, 2);

			if (p1 == 0)
			{
				L1 = A(0, 0);
				L2 = A(1, 1);
				L3 = A(2, 2);
			}
			else
			{
				T q = trace() / 3.0;

				T p2 = (A(0, 0) - q) *(A(0, 0) - q) + (A(1, 1) - q)*(A(1, 1) - q) + (A(2, 2) - q)*(A(2, 2) - q) + 2 * p1;

				T p = std::sqrt(p2 / 6.0);

				matrix<T> I_3(3, 3);
				I_3.Identity();

				matrix<T> B = (A - I_3*q)*(1 / p);

				T r = B.Det_3x3(0, 0) / 2.0;

				T phi;

				if (r <= -1)
				{
					phi = M_PI / 3.0;
				}
				else if (r >= 1)
				{
					phi = 0;
				}

				else
				{
					phi = std::acos(r) / 3.0;
				}

				L1 = q + 2 * p * std::cos(phi);
				L2 = q + 2 * p * std::cos(phi + (2 * M_PI / 3));
				L3 = 3 * q - L1 - L2;

			}
#undef A
		}

		void QR_algorithm(matrix<T>& eigen_values)
		{
			if (!this->IsSquare())
			{
				cout << "Error (QR_algorithm): Matrix must be square" << endl;
				return;
			}

			int n = this->NumCols();

			matrix<T> C_n(n, n);
			C_n.Identity();

			matrix<T> R_n(n, n);
			R_n.Identity();

			matrix<T> I_b(2, 2);
			I_b.Identity();

			matrix<T> *C = new matrix<T>[n - 1]; // an array of n 4x4 matrices
			for (int i = 0; i < n - 1; i++)
				C[i] = I_b;



			// to compute R0 = C

			for (int loop = 0; loop < 100; loop++)
			{
				for (int j = 0; j < n - 1; j++)
				{
					T b11 = get(j, j);
					T b21 = get(j + 1, j);

					T tan_theta = (b21 / b11);

					T cos_theta = 1 / sqrt(1 + tan_theta*tan_theta);
					T sin_theta = tan_theta / sqrt(1 + tan_theta*tan_theta);

					C[j](0, 0) = cos_theta;      C[j](0, 1) = sin_theta;
					C[j](1, 0) = -sin_theta;      C[j](1, 1) = cos_theta;


					C_n.Overwrite_Submatrix(C[j], j, j);

					(*this) = C_n * (*this);

					C_n.Overwrite_Submatrix(I_b, j, j); // set back to identity for next C_j
				}
				//C_n.Identity();

				for (int j = 0; j < n - 1; j++)
				{
					C_n.Overwrite_Submatrix_transposed(C[j], j, j);

					(*this) = (*this) * C_n;

					C_n.Overwrite_Submatrix(I_b, j, j); // set back to identity for next C_j
				}

			}

			// set the eigen values and exit
			bool flag_last = false;
			for (int j = 0; j < n - 1; j++)
			{
				T sub_diag2 = get(j + 1, j);
				//precision = FLT_EPSILON;
				if (sub_diag2 > precision || sub_diag2 < -precision /*FLT_EPSILON*/)
				{
					complex<T> L1, L2;
					Eigenvalues_2x2(j, j, L1, L2);

					eigen_values(j, 0) = L1.real();
					eigen_values(j, 1) = L1.imag();

					eigen_values(j + 1, 0) = L2.real();
					eigen_values(j + 1, 1) = L2.imag();

					flag_last = true;
				}
				else
				{
					if (!flag_last)
						eigen_values(j, 0) = get(j, j);

					flag_last = false;
				}
			}

			if (flag_last == false)
			{
				eigen_values(n - 1, 0) = get(n - 1, n - 1);
			}

			delete[] C;

		}




		void ToZero()
		{
			for (int i = 0; i < this->NumRows(); i++)
				for (int j = 0; j < this->NumCols(); j++)
					get(i, j) = 0;
		}

		void ClipToZero(T eps)
		{
			for (int i = 0; i < this->NumRows(); i++)
				for (int j = 0; j < this->NumCols(); j++)
					if ((get(i, j) > 0 && get(i, j) < eps) ||
						(get(i, j) < 0 && get(i, j) > -eps))
						get(i, j) = 0;
		}


		void Round_to_N_digits(int N)
		{
			for (int i = 0; i < this->NumRows(); i++)
				for (int j = 0; j < this->NumCols(); j++)
					get(i, j) = round_to_n_digits(get(i, j), N);
		}




		inline int max_row_of_column(int row_start, int c)
		{
			if (row_start > this->NumRows())
			{
				cout << "Error (max_row_of_column): row start exceeds bounds" << endl;
				return -1;
			}

			if (c >= this->NumCols())
			{
				cout << "Error (max_row_of_column): column index exceeds bounds" << endl;
				return -1;
			}


			T max_val = get(row_start, c);
			int max_int = row_start;
			for (int r = row_start + 1; r < NumRows(); r++)
			{
				if (get(r, c) > max_val)
				{
					max_val = get(r, c);
					max_int = r;
				}
			}
			return max_int;
		}

		inline int max_off_diagonal_elem_in_row(int r)
		{
			if (r >= this->NumRows())
			{
				cout << "Error (max_row_of_column): row start exceeds bounds" << endl;
				return -1;
			}

			T max_val = 0.0;
			int max_int = 0;
			for (int c = 0; c < NumCols(); c++)
			{
				if (r != c)
				{
					T val = abs(get(r, c));
					if (val > max_val)
					{
						max_val = val;
						max_int = c;
					}
				}
			}
			return max_int;
		}

		inline void SwapRow(int r1, int r2)
		{
			if (r1 >= this->NumRows() || r2 >= this->NumRows())
			{
				cout << "Error (SwapRow): index out of bounds" << endl;
				return;
			}

			for (int c = 0; c < this->NumCols(); c++)
			{
				SWAP<T>(get(r1, c), get(r2, c));
			}
		}

		inline void SwapColumn(int c1, int c2)
		{
			if (c1 >= this->NumCols() || c2 >= this->NumCols())
			{
				cout << "Error (SwapColumn): index out of bounds" << endl;
				return;
			}

			for (int r = 0; r < this->NumRows(); r++)
			{
				SWAP<T>(get(r, c1), get(r, c2));
			}
		}

		inline void CopyVector_from_SubMatrix_to_SubMatrix(matrix<T>& Source, int r1, int r2, int n)
		{
			if (n > this->NumCols() || n > Source.NumCols())
			{
				cout << "Error (CopyVector_from_SubMatrix_to_SubMatrix): columns overflow" << endl;
				return;
			}
			for (int i = 0; i < n; i++)
			{
				get(r1, i) = Source(r2, i);
			}
		}




		T Frobenius_Norm()
		{
			T sum = 0.0;
			for (int r = 0; r < NumRows(); r++)
			{
				for (int c = 0; c < NumCols(); c++)
				{
					sum += get(r, c)*get(r, c);
				}
			}

			return sqrt(sum);
		}



		matrix<T>(T p[6][6])
		{
			SX = 6; SY = 6;
			this->create();
			this->CopyData((T*)p);
		}

		matrix<T>(T p[5][5])
		{
			SX = 5; SY = 5;
			this->create();
			this->CopyData((T*)p);
		}

		matrix<T>(T p[4][4])
		{
			SX = 4;  SY = 4;
			this->create();

			this->CopyData((T*)p);
		}

		matrix<T>(T p[3][3])
		{
			SX = 3;  SY = 3;
			this->create();
			this->CopyData((T*)p);
		}

		matrix<T>(T p[2][2])
		{
			SX = 2; SY = 2;
			this->create();
			this->CopyData((T*)p);
		}


		//http://stackoverflow.com/questions/19840213/how-to-read-values-from-a-2d-initializer-list-and-put-them-in-a-2d-vector
		matrix<T>(const std::initializer_list<std::initializer_list<T>>& list)
		{

			this->SX = list.size();
			this->SY = (*list.begin()).size();
			this->create();

			int r = 0;
			for (const auto& l : list) {
				int c = 0;
				for (const auto d : l) {
					(*this)(r, c++) = d; // rather than get() because of the bounds check
				}
				++r;
			}
		}

		//	virtual void Set_Zero_Epsilon() = 0;

		T precision = FLT_EPSILON; // defaults to float eps
	private:

		void CopyData(T *p)
		{
			memcpy(data, p, SX*SY*sizeof(T));
		}



		bool is_transposed = false;
		unsigned int SX = 0;
		unsigned int SY = 0;

		matrix<T> *out = 0;

		T* data = 0;

	};


public:

	typedef matrix < float >  matrixf;
	typedef  matrix < double > matrixd;





};



inline LINALG::matrixf operator*(float s, LINALG::matrixf a)
{
	return a * s;
}

#endif
