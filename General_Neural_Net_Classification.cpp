// General_Neural_Net_Classification.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "my_matrix.h"
#include "NeuralNetworks.h"
#include "Timer.h"

#include <iostream>
#include <vector>
#include <algorithm>

void Load_SPIRAL_data_(vector< vector <float> > &X, vector<int> &training_set_spiral, vector<int> &test_set_spiral)
{
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
	
}

void Compute_SPIRAL_data_version_3_(int num_iterations, vector<vector<float>> &data, vector<int> &training_set, vector<int> &test_set, vector<int> &indexes, double &time_)
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



using namespace Vanilla_Layer::Linked_Layer_Loop_Eval;
NeuralNetwork * Compute_WAVE_data_version_1_(int num_iterations, double &time_)
{
	Timer timer;

	int input_data_size = 1;
	int num_inputs = 256;
	int num_hidden = 64;
	int num_hidden2 = 3;
	int num_hidden3 = 8;
	int num_outputs = 20;

	// ==========================================
	matrix input_matrix(1, num_inputs);

	matrix input_array[20];
	for (int i = 0; i < num_outputs; i++)
	{
		input_array[i].create(1, num_inputs);
	}

	float Time_Period = 256.0f; // normalizing the time period
	float t = 0.0f;
	
	float frequency_min = 1;
	float frequency_max = 750;
	float frequency_step = 60;
	float freq = frequency_min;
	float omega = 0.0f;
	float theta = 0.0f;
	float wave = 0.0f;
	for (int j = 0; j < num_outputs; j++)
	{ 
		freq = frequency_min +  float(j); 
		omega = 2.0f * M_PI * freq / Time_Period;
		//cout << j << ":= ";
		for (int t = 0; t < num_inputs; t++)
		{
			theta = omega*(float)t;
			wave = std::sinf(theta)+ 1;
			input_array[j](0, t) = wave ;
			 //cout << wave << ", ";
		}
		//cout << endl;
	}


	float alpha = 0.5f;
	float beta = 0.2;// 5f; // this does not appear to be working on either of these problems

	float sum_squared_errors = 0.0f;

	

	vector<Layer*> layers;

	layers.push_back(new Linear_Layer(num_inputs));

	//layers.push_back(new Leaky_ReLU_Layer(num_hidden));
//	layers.push_back(new ReLU_Layer(num_hidden));
	layers.push_back(new Sigmoid_Layer(num_hidden));
	//layer_sizes.push_back(num_hidden2);
	//layer_sizes.push_back(num_hidden3);
	layers.push_back(new Sigmoid_Layer(num_outputs));

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

	matrix expected(1, num_outputs);
	matrix output(1, num_outputs);
	matrix error(1, num_outputs);

	for (int mm = 0; mm < num_iterations; mm++)
	{

		for (int q = 0; q < num_outputs; q++)
		{
			// index remap the shuffled set to the original data
			for (int t = 0; t < num_inputs; t++)
				input_matrix(0, t) = input_array[q](0, t);

			// formulate the correct output vector
			expected(0, q) = 1.0f;
		
			
			// feed forward

			output = neuralNet->FeedForward(input_matrix);

			// back propagate output
			neuralNet->BackPropagateErrors(expected, output);

			// weight deltas
			neuralNet->ComputeDeltas(alpha, beta);

			// update weights
			neuralNet->UpdateWeights();

		}
		expected.ToZero();

	}
	
	timer.Update();
	timer.Stop();

	expected.ToZero();


	double time_taken = timer.GetTimeDelta();
	cout << "Finished training, Total calculation performed in " << time_taken << " seconds" << endl;

	time_ += time_taken;

	sum_squared_errors = 0.0f; // used here to count the number of correct guesses

	//tolerance += 0.2;

	matrix test_output(1, num_outputs);


	for (int q = 0; q < num_outputs; q++)
	{
		// index remap the shuffled set to the original data
		for (int t = 0; t < num_inputs; t++)
			input_matrix(0, t) = input_array[q](0, t);

		// formulate the correct output vector
		expected(0, q) = 1.0f;
		if (q > 0)
			expected(0, q - 1) = 0.0f;

		// test the neural network

		test_output = neuralNet->FeedForward(input_matrix);

		//if (test_output.NumCols() != 3) cout << "should have more columns" << endl;
		error = expected - test_output;
		//test_output.print(2);
		cout << endl;
		int actual_type = q;

		int found_type = q;

		bool b_found_example = false;
		
		float max_out = 0.0f;
			for (int p = 0; p < num_outputs; p++)
			{
				//if (test_output(0, p) > 0.2)
				//cout << error(0, p) << ", " << p << endl;

				if (error(0, p) > -0.5)
				{
					b_found_example = true;
				}
				if (error(0, p)  > -0.5)
					found_type = p;
			}

			if (b_found_example)
				if (found_type == actual_type)
					sum_squared_errors += 1.0f;



		//	cout << "Test Sample: " << q << ", Found Type: " << found_type << ", Actual Type: " << actual_type << endl;

		
	}

	cout << "Finished Test, Total classified correctly of " << num_outputs << " tested: " << (int)sum_squared_errors << endl;
	return neuralNet;
}

void Test_WAVE_NN_version_1_(NeuralNetwork *neuralNet, double &time_)
{
	Timer timer;

	
	int num_inputs = 256;
	int num_hidden = 64;

	int num_outputs = 20;

	// ==========================================
	matrix input_matrix(1, num_inputs);


	matrix input_array[4];
	for (int i = 0; i < 4; i++)
	{
		input_array[i].create(1, num_inputs);
	}

	float Time_Period = 256.0f; // normalizing the time period
	float t = 0.0f;

	float frequency_min = 1;
	float frequency_max = 750;
	float frequency_step = 60;
	float freq = frequency_min;
	float freq2 = 0.0f;
	float omega = 0.0f;
	float theta = 0.0f;
	float omega2 = 0.0f;
	float theta2 = 0.0f;
	float wave = 0.0f;

	matrix expected_array[4];
	
	for (int i = 0; i < 4; i++)
	{
		expected_array[i].create(1, num_outputs);
	}

	for (int j = 1; j < 5; j++)
	{
		freq = frequency_min + float(j);
		freq2 = frequency_min + 3 * float(j);
		expected_array[j-1](0,j) = 1.0f;
		expected_array[j-1](0, 3 * j) = 1.0f;

		omega = 2.0f * M_PI * freq / Time_Period;
		omega2 = 2.0f * M_PI * freq2 / Time_Period;

		for (int t = 0; t < num_inputs; t++)
		{
			theta = omega*(float)t;
			theta2 = omega2*(float)t;
			wave =  ((std::sinf(theta) + std::sinf(theta2)) +2)/2;
			input_array[j-1](0, t) = wave ;
		}
	}


	float alpha = 0.5f;
	float beta = 0.2;// 5f; // this does not appear to be working on either of these problems

	float sum_squared_errors = 0.0f;

	
	timer.Start();

	float tolerance = 0.2; // the new hyperparameter

	
	matrix output(1, num_outputs);


	sum_squared_errors = 0.0f; // used here to count the number of correct guesses



	matrix test_output(1, num_outputs);

	matrix errors(1, num_outputs);

	for (int q = 0; q < 4; q++)
	{
		// index remap the shuffled set to the original data
		for (int t = 0; t < num_inputs; t++)
			input_matrix(0, t) = input_array[q](0, t);

	
		// test the neural network

		test_output = neuralNet->FeedForward(input_matrix);

		errors = expected_array[q] - test_output;

		//if (test_output.NumCols() != 3) cout << "should have more columns" << endl;

		int actual_type = q;

		int found_type = q;

		

	
		cout << q << ":  ";
		bool b_found_example = false;

		float max_out = 0.0f;
		cout << " Test Wave: " << q << endl;
		for (int p = 0; p < num_outputs; p++)
		{
			
			//if (test_output(0, p) > 0.2)
			

			if (errors(0, p) > -0.5)
			{
				b_found_example = true;
				cout << " Frequency: " << p << endl;
			}
			if (errors(0, p)  > -0.5)
				found_type = p;
		}

		if (b_found_example)
			if (found_type == actual_type)
				sum_squared_errors += 1.0f;
		





		//	cout << "Test Sample: " << q << ", Found Type: " << found_type << ", Actual Type: " << actual_type << endl;


	}
	timer.Update();
	timer.Stop();

	double time_taken = timer.GetTimeDelta();
	cout << "Finished testing, Total calculation performed in " << time_taken << " seconds" << endl;

	cout << "Finished Test, Total classified correctly of " << 4 << " tested: " << (int)sum_squared_errors << endl;

}
int main(int argc, char* argv[])
{

	// load the spiral dataset

	double total_time = 0;
	cout << "===========================================================================================" << endl;
	cout << "Testing Linked_Layer_Loop_Eval neural net object" << endl;
	cout << "This example recognizes input waves (sin)" << endl;

//	Compute_IRIS_data_version_3_
//		(1500, X, training_set_spiral, test_set_spiral,
//		indexes_spiral, total_time);

	cout << endl << endl;
	cout << "Finished testing linked layer with iterative evaluation neural net object" << endl;

	NeuralNetwork *nn = Compute_WAVE_data_version_1_
		(2,  total_time);

	cout << endl << endl; 
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "Testing waves" << endl;
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << endl << endl;
	Test_WAVE_NN_version_1_(nn, total_time);

	return 0;
}

