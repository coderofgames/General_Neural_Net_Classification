// General_Neural_Net_Classification.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "my_matrix.h"
#include "NeuralNetworks.h"
#include "Timer.h"

#include <iostream>
#include <vector>
#include <algorithm>

void Compute_IRIS_data_version_3_(int num_iterations, vector<vector<float>> &data, vector<int> &training_set, vector<int> &test_set, vector<int> &indexes, double &time_)
{
	Timer timer;

	int input_data_size = 1;
	int num_inputs = 2;
	int num_hidden = 9;
	int num_hidden2 = 3;
	int num_hidden3 = 8;
	int num_outputs = 3;

	// ==========================================
	matrix input_matrix(1, num_inputs);

	float alpha = 0.5f;
	float beta = 0.2;// 5f; // this does not appear to be working on either of these problems

	float sum_squared_errors = 0.0f;

	using namespace Vanilla_Layer::Linked_Layer_Loop_Eval;

	vector<Layer*> layers;
	
	layers.push_back(new Linear_Layer(num_inputs));

	//layers.push_back(new Leaky_ReLU_Layer(num_hidden));
	//layers.push_back(new ReLU_Layer(num_hidden));
	layers.push_back(new Sigmoid_Layer(num_hidden));
	//layer_sizes.push_back(num_hidden2);
	//layer_sizes.push_back(num_hidden3);
	layers.push_back(new Softmax_Layer(num_outputs));

	NeuralNetwork *neuralNet =
		new NeuralNetwork(layers);

	neuralNet->Initialize_Weights();

	//neuralNet->input_layer[3]->alpha = 0.1;
	neuralNet->input_layer[2]->alpha_ = 0.3;
	neuralNet->input_layer[1]->alpha_ = 0.5;
	neuralNet->input_layer[0]->alpha_ = 0.7;

	cout << endl;
	cout << "Training, please wait ..." << endl;

	timer.Start();

	float tolerance = 0.15; // the new hyperparameter

	matrix expected(1, 3);
	matrix output(1, 3);

	for (int mm = 0; mm < num_iterations; mm++)
	{

		for (int q = 0; q < training_set.size(); q++)
		{
			// index remap the shuffled set to the original data
			input_matrix(0, 0) = data[training_set[q]][0];
			input_matrix(0, 1) = data[training_set[q]][1];

			// formulate the correct output vector

			for (int p = 0; p < 3; p++)
			{
				if ((int)data[training_set[q]][2] == p)
				{
					expected(0, p) = 1.0;
				}
				else
				{
					expected(0, p) = 0.0;
				}
			}

			// feed forward

			output = neuralNet->FeedForward(input_matrix);

			// back propagate output
			neuralNet->BackPropagateErrors(expected, output);

			// weight deltas
			neuralNet->ComputeDeltas(alpha, beta);

			// update weights
			neuralNet->UpdateWeights();

		}

	}

	timer.Update();
	timer.Stop();

	double time_taken = timer.GetTimeDelta();
	cout << "Finished training, Total calculation performed in " << time_taken << " seconds" << endl;

	time_ += time_taken;

	sum_squared_errors = 0.0f; // used here to count the number of correct guesses

	tolerance += 0.2;

	matrix test_output(1, 3);
	for (int q = 0; q < test_set.size(); q++)
	{
		input_matrix(0, 0) = data[test_set[q]][0];
		input_matrix(0, 1) = data[test_set[q]][1];

		// test the neural network

		test_output = neuralNet->FeedForward(input_matrix);

		if (test_output.NumCols() != 3) cout << "should have more columns" << endl;

		int actual_type = (int)data[test_set[q]][2];

		int found_type = 0;

		if ((test_output(0, 0) > (1 - tolerance)) && (test_output(0, 1) < tolerance) && (test_output(0, 2) < tolerance))
		{
			found_type = 0;
		}
		if ((test_output(0, 0) < tolerance) && (test_output(0, 1) > (1 - tolerance)) && (test_output(0, 2) < tolerance))
		{
			found_type = 1;
		}
		if ((test_output(0, 0) < tolerance) && (test_output(0, 1) < tolerance) && (test_output(0, 2)> (1 - tolerance)))
		{
			found_type = 2;
		}

		//	cout << "Test Sample: " << q << ", Found Type: " << found_type << ", Actual Type: " << actual_type << endl;

		if (found_type == actual_type) sum_squared_errors += 1.0f;
	}

	cout << "Finished Test, Total classified correctly of " << test_set.size() << " tested: " << (int)sum_squared_errors << endl;

}


int main(int argc, char* argv[])
{

	// load the spiral dataset
	std::ifstream data_output;


	data_output.open("some.csv", std::ifstream::in);
	// 
	vector<string> tokenized_string;
	copy(istream_iterator<string>(data_output),
		istream_iterator<string>(),
		back_inserter<vector<string> >(tokenized_string));

	vector<vector<string>> output;
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

		for (int c = 0; c < result.size(); c++)
			cout << result[c] << ", ";
		cout << endl;
	}

	data_output.close();

	vector< vector <float> > X;
	vector <int>  Y;

	for (int i = 0; i < output.size(); i++)
	{

		vector<float> v;
		v.push_back(atof(output[i][0].c_str()));
		v.push_back(atof(output[i][1].c_str()));
		v.push_back(atof(output[i][2].c_str()));

		X.push_back(v);

	}



	vector<int> indexes_spiral;
	for (int i = 0; i < X.size(); i++)
	{

		indexes_spiral.push_back(i);
	}

	data_output.close();

	// shuffle the indexes to randomize the order of the data

	for (int i = 0; i < 5; i++)
		std::random_shuffle(indexes_spiral.begin(), indexes_spiral.end());


	// create a vector of indexes for training
	vector<int> training_set_spiral;
	// create a vector of indexes for testing
	vector< int > test_set_spiral;

	// store the first half of the indexes in the training set
	// and the second half of the indexes in the test set
	for (int i = 0; i < indexes_spiral.size(); i++)
	{
		if (i < 200)
		{
			training_set_spiral.push_back(indexes_spiral[i]);
		}
		else
		{
			test_set_spiral.push_back(indexes_spiral[i]);
		}
	}

	double total_time = 0;
	cout << "===========================================================================================" << endl;
	cout << "Testing Linked_Layer_Loop_Eval neural net object" << endl;

	Compute_IRIS_data_version_3_
		(1500, X, training_set_spiral, test_set_spiral,
		indexes_spiral, total_time);

	cout << endl << endl;
	cout << "Finished testing linked layer with iterative evaluation neural net object" << endl;
	return 0;
}

