#ifndef UTILS_H
#define UTIlS_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>



using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::istringstream;
using std::copy;
using std::istream_iterator;
using std::back_inserter;
using std::ifstream;
using std::ios;
using std::runtime_error;

inline float RandomFloat(float min, float max)
{
	float r = (float)rand() / (float)RAND_MAX;
	return min + r * (max - min);
}

inline float RandomInt(int min, int max)
{
	float r = (float)rand() / (float)RAND_MAX;
	return (int)((float)min + r * float(max - min));
}


inline int string_to_number(string s)
{
	return atoi(s.c_str());
}

inline float string_to_float(string s)
{
	return atof(s.c_str());
}

inline bool isNumberC(const std::string& s)
{
	char* p;
	strtod(s.c_str(), &p);
	return *p == 0;
}

inline double string_to_double(string s)
{
	char* p;
	double val = strtod(s.c_str(), &p);
	if (*p == 0) return val;

	return 0.0;
}

class CSV
{
public:

	vector< vector< float >> iris_data;
	vector< string > found_tags;

	void test()
	{
		vector<vector<string>> output;
		LoadCSV("iris.csv", output);


		static bool first_time = true;

		if (first_time)
		{
			for (int i = 0; i < output.size(); i++)
			{
				for (int j = 0; j < output[i].size(); j++)
				{
					cout << output[i][j] << ":";

				}
				cout << endl;
			}
		}



		RestoreCSV_Iris_Numbers(output, iris_data, found_tags);
		if (first_time)
		{
			for (int i = 0; i < iris_data.size(); i++)
			{
				for (int j = 0; j < iris_data[i].size(); j++)
				{
					cout << iris_data[i][j] << ":";

				}
				cout << endl;
			}

		}

		cout << endl <<"printing found tags" << endl;
		for (int p = 0; p < found_tags.size(); p++)
			cout << found_tags[p] << endl;

		first_time = false;
	}
	void RestoreCSV_Iris_Numbers(vector< vector< string> > &v1, vector <vector< float> > &vout, vector<string> &found_tags)
	{

		for (int i = 0; i < v1.size(); i++)
		{
			vout.push_back(vector<float>());
			for (int j = 0; j < v1[i].size(); j++)
			{
				if (isNumberC(v1[i][j]))
				{
					vout[i].push_back((float)string_to_double(v1[i][j]));
				}
				else
				{
					bool tag_found = false;
					for (int p = 0; p < found_tags.size(); p++)
					{
						if (v1[i][j] == found_tags[p])
						{
							tag_found = true;
							vout[i].push_back((float)p);
						}
					}
					if (!tag_found)
					{
						found_tags.push_back(v1[i][j]);
						vout[i].push_back(found_tags.size() - 1);
					}
				}
			}
		}
	}

	void LoadCSV(string filename, vector<vector<string>> &output)
	{
		ifstream file_stream;
		file_stream.open(filename, std::ifstream::in);

		// split input by newline
		vector<string> tokenized_string;
		copy(istream_iterator<string>(file_stream),
			istream_iterator<string>(),
			back_inserter<vector<string> >(tokenized_string));

		// split lines by commas and store in output
		for (int j = 0; j < tokenized_string.size(); j++) {
			istringstream ss(tokenized_string[j]);
			vector<string> result;
			while (ss.good()) {
				string substr;
				getline(ss, substr, ',');
				result.push_back(substr);
			}
			output.push_back(result);
		}
		file_stream.close();
	}

};


typedef unsigned char uchar;

class MNIST
{
public:

	/*
	functions from Stack Overflow
	http://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

	These return raw points to arrays of type uchar (typedef above)
	The array of images will be somthing like
	uchar MNIST_images[number_of_images][image_size];
	although in code it will be 
	uchar **MNIST_images;
	The labels will be stored as
	uchar *MNIST_Labels;
	and will be accessed
	MNIST_Labels[num_labels];
	*/
	uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size, int &rows, int &cols) {
		auto reverseInt = [](int i) {
			unsigned char c1, c2, c3, c4;
			c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
			return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
		};

		typedef unsigned char uchar;

		ifstream file(full_path, ios::binary);

		if (file.is_open()) {
			int magic_number = 0, n_rows = 0, n_cols = 0;

			file.read((char *)&magic_number, sizeof(magic_number));
			magic_number = reverseInt(magic_number);

			if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

			file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
			file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
			file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

			image_size = n_rows * n_cols;
			rows = n_rows;
			cols = n_cols;

			uchar** _dataset = new uchar*[number_of_images];
			for (int i = 0; i < number_of_images; i++) {
				_dataset[i] = new uchar[image_size];
				file.read((char *)_dataset[i], image_size);
			}
			return _dataset;
		}
		else {
			throw runtime_error("Cannot open file `" + full_path + "`!");
		}
	}

	uchar* read_mnist_labels(string full_path, int& number_of_labels) {
		auto reverseInt = [](int i) {
			unsigned char c1, c2, c3, c4;
			c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
			return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
		};

		typedef unsigned char uchar;

		ifstream file(full_path, ios::binary);

		if (file.is_open()) {
			int magic_number = 0;
			file.read((char *)&magic_number, sizeof(magic_number));
			magic_number = reverseInt(magic_number);

			if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

			file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

			uchar* _dataset = new uchar[number_of_labels];
			for (int i = 0; i < number_of_labels; i++) {
				file.read((char*)&_dataset[i], 1);
			}
			return _dataset;
		}
		else {
			throw runtime_error("Unable to open file `" + full_path + "`!");
		}
	}
};
#endif
