#ifndef NEURAL_NETWORKS_H
#define NEURAL_NETWORKS_H


#include "my_matrix.h"




#include <math.h>
//#undef matrix;
#ifndef max
#define max(a,b) a>b?a:b
#endif

typedef LINALG::matrixf matrix;

namespace
{
	float tan_hyperbolic(float x)
	{
		float ex = std::exp(x);
		float emx = std::exp(-x);
		return (ex - emx) / (ex + emx);
	}
	float tan_hyperbolic_deriv(float y)
	{
		return (1 + y)* (1 - y);
	}


	matrix& tan_hyperbolic(matrix &out, matrix &m)
	{
		if (out.NumCols() != m.NumCols() || out.NumRows() != m.NumRows())
		{
			out.destroy();
			out.create(m.NumRows(), m.NumCols());
		}
		for (int i = 0; i < m.NumRows(); i++)
			for (int j = 0; j < m.NumCols(); j++)
				out(i, j) = tan_hyperbolic(m(i, j));

		return out;
	}

	matrix& tan_hyperbolic_derivative(matrix& out, matrix &m)
	{
		if (out.NumCols() != m.NumCols() || out.NumRows() != m.NumRows())
		{
			out.destroy();
			out.create(m.NumRows(), m.NumCols());
		}
		for (int i = 0; i < m.NumRows(); i++)
			for (int j = 0; j < m.NumCols(); j++)
				out(i, j) = tan_hyperbolic_deriv(m(i, j));

		return out;
	}

	float sigmoid(float x)
	{
		return 1.f / (1.f + std::exp(-x));
	}

	float sigmoid_deriv(float x)
	{
		return x * (1 - x);
	}

	matrix& sigmoid(matrix &out, matrix &m)
	{
		if (out.NumCols() != m.NumCols() || out.NumRows() != m.NumRows())
		{
			out.destroy();
			out.create(m.NumRows(), m.NumCols());
		}
		for (int i = 0; i < m.NumRows(); i++)
			for (int j = 0; j < m.NumCols(); j++)
				out(i, j) = 1.f / (1.f + std::exp(-m(i, j)));

		return out;
	}

	matrix& sigmoid_deriv(matrix& out, matrix &m)
	{
		if (out.NumCols() != m.NumCols() || out.NumRows() != m.NumRows())
		{
			out.destroy();
			out.create(m.NumRows(), m.NumCols());
		}
		for (int i = 0; i < m.NumRows(); i++)
			for (int j = 0; j < m.NumCols(); j++)
				out(i, j) = m(i, j) * (1 - m(i, j));

		return out;
	}


	matrix& pow(matrix& source, float exponent)
	{
		for (int i = 0; i < source.NumRows(); i++)
		{
			for (int j = 0; j < source.NumCols(); j++)
			{
				source(i, j) = std::pow(source(i, j), exponent);
			}
		}
		return source;
	}

	matrix& ln(matrix& source)
	{
		for (int i = 0; i < source.NumRows(); i++)
		{
			for (int j = 0; j < source.NumCols(); j++)
			{
				source(i, j) = log(source(i, j));
			}
		}
		return source;
	}

	matrix& exp(matrix& source)
	{
		for (int i = 0; i < source.NumRows(); i++)
		{
			for (int j = 0; j < source.NumCols(); j++)
			{
				source(i, j) = std::exp(source(i, j));
			}
		}
		return source;
	}

	float SUM(matrix& source)
	{
		float sum = 0.0f;
		for (int i = 0; i < source.NumRows(); i++)
		{
			for (int j = 0; j < source.NumCols(); j++)
			{
				sum += source(i, j);// = exp(source(i, j));
			}
		}
		return sum;
	}









	matrix& QuadraticCostFunction(matrix& out, matrix& desired_outputs, matrix& outputs)
	{
		out = pow((desired_outputs - outputs), 2) *0.5;

		return out;
	}
}


namespace Vanilla_Layer
{



	namespace Linked_Layer
	{
		class Layer;

		class Link_Tensor
		{
		public:
			Link_Tensor(Layer* Prev, Layer* Next, int num_inputs, int num_outputs)
			{
				weights_.create(num_inputs, num_outputs);
				delta_weights_.create(num_inputs, num_outputs);

				prev = Prev;
				next = Next;
			}

			void init_random_sample_weights_iris()
			{
				for (int i = 0; i < weights_.NumRows(); i++)
				{
					for (int j = 0; j < weights_.NumCols(); j++)
					{
						weights_(i, j) = RandomFloat(0, 1) *0.01;
					}
				}
			}

			void Compute_Weight_Deltas(matrix &deltas_, float alpha, float beta);

			void UpdateWeights();

			Layer * prev;
			Layer * next;

			matrix weights_;
			matrix delta_weights_;
		};



		class Layer
		{
		public:

			float alpha = 0.5;

			Link_Tensor *connection_in;
			Link_Tensor *connection_out;

			matrix neurons_;
			matrix thetas_;


			matrix delta_thetas_;
			 matrix last_deltas;

			matrix deltas_;

			Layer(){}

			Layer(unsigned int num_elements, unsigned int num_inputs)
			{
				neurons_.create(1, num_elements);
				last_deltas.create(1, num_elements);
				thetas_.create(1, num_elements);
				deltas_.create(1, num_elements);
				delta_thetas_.create(1, num_elements);
			}

			Layer(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
			{
				connection_in = in;
				connection_out = out;

				neurons_.create(1, num_elements);

				thetas_.create(1, num_elements);

				delta_thetas_.create(1, num_elements);
			}

			void init_random_sample_weights_iris()
			{


				for (int i = 0; i < thetas_.NumRows(); i++)
				{
					for (int j = 0; j < thetas_.NumCols(); j++)
					{
						thetas_(i, j) = RandomFloat(0, 1) / 5;
					}
				}
			}

			matrix& FeedForward(matrix &input_matrix)
			{
				neurons_ = input_matrix * connection_in->weights_ - thetas_;

				sigmoid(neurons_, neurons_);

				if (connection_out && connection_out->next)
					return connection_out->next->FeedForward(neurons_);

				return neurons_;
#ifdef VERBOSE
				PrintNeurons();
#endif
			}

			void BackPropogate()
			{
				
				


				sigmoid_deriv(deltas_, neurons_);

				

				//if ( connection_out )

				connection_out->weights_.transpose();
				deltas_ = deltas_ | (connection_out->next->deltas_ * connection_out->weights_);
				connection_out->weights_.transpose();
				
				/*if (last_deltas.NumCols() != 0)
				{
				
					deltas_ = deltas_ +  (deltas_ - last_deltas )* this->alpha;
					
				}

				last_deltas = deltas_;
				*/


				if (connection_in && connection_in->prev)
					connection_in->prev->BackPropogate();

#ifdef VERBOSE
				PrintDeltas();
#endif

			}

			void BackPropogate_output(matrix& output_error)
			{
				deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

				if (connection_in && connection_in->prev)
					connection_in->prev->BackPropogate();
#ifdef VERBOSE
				PrintDeltas();
#endif
			}

			void ComputeWeightDeltas(float alpha, float beta)
			{
				if (connection_in)
					connection_in->Compute_Weight_Deltas(deltas_, this->alpha, beta);

				delta_thetas_ = delta_thetas_ * beta + deltas_* (-1.0f) * this->alpha;


#ifdef VERBOSE
				PrintDeltaWeights();
#endif
			}


			void UpdateWeights()
			{
				if (connection_in)
					connection_in->UpdateWeights();

				thetas_ = thetas_ + delta_thetas_;

#ifdef VERBOSE
				PrintWeights();
#endif
			}

			void PrintNeurons()
			{
				cout << "Neurons" << endl;
				neurons_.print(3);
				cout << endl;
			}

			void PrintDeltas()
			{
				cout << "Deltas" << endl;
				deltas_.print(3);
			}
			void PrintDeltaWeights()
			{
				cout << "Delta Weights" << endl;
				connection_in->delta_weights_.print(3);
			}
			void PrintDeltaThetas()
			{
				cout << "Delta Thetas" << endl;
				delta_thetas_.print(3);
			}

			void PrintWeights()
			{
				cout << "Weights" << endl;
				connection_in->weights_.print(3);
			}
		};


		void Link_Tensor::Compute_Weight_Deltas(matrix &deltas_, float alpha, float beta)
		{
			delta_weights_.transpose();

			deltas_.transpose();

			delta_weights_ = delta_weights_ * beta + deltas_* this->prev->neurons_ * alpha;

			delta_weights_.transpose();

			deltas_.transpose();


			if (prev)
				prev->ComputeWeightDeltas(alpha, beta);


		}

		void Link_Tensor::UpdateWeights()
		{
			weights_ = weights_ + delta_weights_;// 

			if (prev)
				prev->UpdateWeights();
		}

		class NeuralNetwork
		{
		public:
			NeuralNetwork(){
				input_layer = 0;
			}

			NeuralNetwork(vector<int> &layer_sizes){


				num_layers = layer_sizes.size();
				if (num_layers > 0)
				{
					input_layer = new Layer*[num_layers];

					input_layer[0] = new Layer(layer_sizes[0], 1);
					input_layer[0]->connection_in = 0;
					for (int i = 1; i < num_layers; i++)
					{
						input_layer[i] = new Layer(layer_sizes[i], layer_sizes[i - 1]);

						input_layer[i]->connection_in =
							new Link_Tensor(input_layer[i - 1],  // prev
							input_layer[i],      // next (this one)
							layer_sizes[i - 1], // num_inputs (nodes in prev)
							layer_sizes[i]);    // num_outputs (nodes in this)

						input_layer[i - 1]->connection_out = input_layer[i]->connection_in; // the last layers output connection is the input to this

					}

					input_layer[num_layers - 1]->connection_out = 0; // or something ...

					// initialize the weights
					for (int i = 0; i < num_layers; i++)
					{
						if (input_layer[i]->connection_in)
							input_layer[i]->connection_in->init_random_sample_weights_iris();
					}

				}
			}

			~NeuralNetwork()
			{
				// initialize the weights
				for (int i = 0; i < num_layers; i++)
				{
					if (input_layer[i]->connection_in)
						delete input_layer[i]->connection_in;
					delete input_layer[i];
				}
				delete[] input_layer;
			}


			matrix& FeedForward(matrix& input)
			{
				input_layer[0]->neurons_ = input;
				return input_layer[1]->FeedForward(input);
			}

			void BackPropagateErrors(matrix& errors)
			{
				input_layer[num_layers - 1]->BackPropogate_output(errors);
			}

			void ComputeDeltas(float alpha, float beta)
			{
				input_layer[num_layers - 1]->ComputeWeightDeltas(alpha, beta);
			}

			void UpdateWeights()
			{
				input_layer[num_layers - 1]->UpdateWeights();
			}

			int num_layers = 0;
			Layer **input_layer;

		};
	}



	namespace Linked_Layer_Loop_Eval
	{
		class Layer;
		class Sigmoid_Layer ;
		class Softmax_Layer;
		class ReLU_Layer;

		class Link_Tensor
		{
		public:
			Link_Tensor(Layer* Prev, Layer* Next, int num_inputs, int num_outputs)
			{
				weights_.create(num_inputs, num_outputs);
				delta_weights_.create(num_inputs, num_outputs);

				drop_out_connection_.create(num_inputs, num_outputs);

				prev = Prev;
				next = Next;
			}

			void init_random_sample_weights_sigmoid()
			{
				for (int i = 0; i < weights_.NumRows(); i++)
				{
					for (int j = 0; j < weights_.NumCols(); j++)
					{
						weights_(i, j) = RandomFloat(0, 1) / 5;
					}
				}
			}

			void init_random_sample_weights_NearZero()
			{
				for (int i = 0; i < weights_.NumRows(); i++)
				{
					for (int j = 0; j < weights_.NumCols(); j++)
					{
						weights_(i, j) = RandomFloat(0, 1) *0.01;
					}
				}
			}

			void init_random_sample_weights_ToZero()
			{
				for (int i = 0; i < weights_.NumRows(); i++)
				{
					for (int j = 0; j < weights_.NumCols(); j++)
					{
						weights_(i, j) = 0.0;
					}
				}
			}

			void Compute_Weight_Deltas(matrix &deltas_, float alpha, float beta);

			void UpdateWeights();

			void Apply_Dropout()
			{
				stored_weights_ = weights_;
				weights_ = weights_ | drop_out_connection_;
			}

			void Restore_Dropout()
			{
				for (int i = 0; i < weights_.NumRows(); i++)
				{
					for (int j = 0; j < weights_.NumCols(); j++)
					{
						if (drop_out_connection_(i, j) == 0.0)
						{
							weights_(i, j) = stored_weights_(i, j);
						}
					}
				}
			}

			Layer * prev;
			Layer * next;

			matrix stored_weights_;
			matrix weights_;
			matrix delta_weights_;

			matrix drop_out_connection_;
		};

		enum LAYER_TYPE
		{
			Linear_Layer_type=0,
			Softmax_Layer_type,
			SoftPlus_Layer_type,
			ELU_Layer_type,
			PReLU_Layer_type,
			Leaky_ReLU_Layer_type,
			ReLU_Layer_type,
			Sigmoid_Layer_type,
		};

		class Layer
		{
		public:
			Link_Tensor *connection_in;
			Link_Tensor *connection_out;

			matrix neurons_;
			matrix thetas_;

			matrix delta_thetas_;

			matrix deltas_;

			int m_num_elements = 0;

			float alpha_ = 0.5;

			float eps = 0.0;

			LAYER_TYPE m_type;

			Layer(){}

			void Create(unsigned int num_elements, unsigned int num_inputs)
			{
				m_num_elements = num_elements;
				neurons_.create(1, num_elements);
				thetas_.create(1, num_elements);
				delta_thetas_.create(1, num_elements);
			}

			void Create(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
			{
				m_num_elements = num_elements;

				connection_in = in;
				connection_out = out;

				neurons_.create(1, num_elements);
				thetas_.create(1, num_elements);
				delta_thetas_.create(1, num_elements);
			}

			void SetConnections(Link_Tensor *in, Link_Tensor *out)
			{
				connection_in = in;
				connection_out = out;
			}

			Layer(unsigned int num_elements)
			{
				m_num_elements = num_elements;

				neurons_.create(1, num_elements);
				thetas_.create(1, num_elements);
				delta_thetas_.create(1, num_elements);
			}

			Layer(unsigned int num_elements, unsigned int num_inputs)
			{
				Create(num_elements, num_inputs);
			}

			Layer(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
			{
				Create(num_elements, num_inputs, in, out);
			}



			virtual void Activation_Function(matrix& out, matrix& in) = 0;

			virtual void Activation_Function_Deriv(matrix& out, matrix& in) = 0;





			void FeedForward(matrix &input_matrix)
			{
				neurons_ = input_matrix * connection_in->weights_ - thetas_;

				Activation_Function(neurons_, neurons_);

#ifdef VERBOSE
				PrintNeurons();
#endif
			}

			void BackPropogate()
			{
				Activation_Function_Deriv(deltas_, neurons_);

				if (connection_out)
				{
					// Claiming that these may pass through unchanged was false
					// the multiplication of the next deltas by the output weights is
					// needed to get the correct matrix size.
					// now I'm using the derivative of the ReLU 
					connection_out->weights_.transpose();
					deltas_ = deltas_ | (connection_out->next->deltas_ * connection_out->weights_);
					connection_out->weights_.transpose();
				}

#ifdef VERBOSE
				PrintDeltas();
#endif

			}
		


			void BackPropogate_output(matrix& expected, matrix& out)
			{
				// predicted - actual
				deltas_ = expected - out; 

#ifdef VERBOSE
				PrintDeltas();
#endif
			}


			void BackPropogate_output(matrix& output_error)
			{
				deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

#ifdef VERBOSE
				PrintDeltas();
#endif
			}

			void ComputeWeightDeltas(float alpha, float beta)
			{
				if (connection_in)
					connection_in->Compute_Weight_Deltas(deltas_, alpha, beta);

				delta_thetas_ = delta_thetas_ * beta + deltas_* (-1.0f) * this->alpha_;


#ifdef VERBOSE
				PrintDeltaWeights();
#endif
			}


			void UpdateWeights()
			{
				if (connection_in)
					connection_in->UpdateWeights();

				thetas_ = thetas_ + delta_thetas_;

#ifdef VERBOSE
				PrintWeights();
#endif
			}

			void PrintNeurons()
			{
				cout << "Neurons" << endl;
				neurons_.print(3);
				cout << endl;
			}

			void PrintDeltas()
			{
				cout << "Deltas" << endl;
				deltas_.print(3);
			}
			void PrintDeltaWeights()
			{
				cout << "Delta Weights" << endl;
				if( connection_in )connection_in->delta_weights_.print(3);
			}
			void PrintDeltaThetas()
			{
				cout << "Delta Thetas" << endl;
				delta_thetas_.print(3);
			}

			void PrintWeights()
			{
				cout << "Weights" << endl;
				if (connection_in)connection_in->weights_.print(3);
			}
		};

		// notes from: http://cs231n.stanford.edu/slides/winter1516_lecture5.pdf
		// Squashes the numbers in the range 0-1 
		// Simulates the firing rate of a neuron
		// 
		// saturated neurons can kill the gradients
		// sigmoid outputs are not zero centered
		// => all the gradients are positive or all the gradients are negative
		class Sigmoid_Layer : public Layer
		{
		public:
			Sigmoid_Layer(unsigned int num_elements) : Layer(num_elements)
			{
				m_type = LAYER_TYPE::Sigmoid_Layer_type;
			}
			Sigmoid_Layer(unsigned int num_elements, unsigned int num_inputs) : Layer(num_elements, num_inputs)
			{
				m_type = LAYER_TYPE::Sigmoid_Layer_type;
			}

			Sigmoid_Layer(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
				: Layer(num_elements, num_inputs, in, out)
			{
				m_type = LAYER_TYPE::Sigmoid_Layer_type;
			}

			void Activation_Function(matrix& out, matrix& in)
			{
				sigmoid(out, in);
			}

			void Activation_Function_Deriv(matrix& out, matrix& in)
			{
				sigmoid_deriv(out, in);
			}

		};
	

		// notes from: http://cs231n.stanford.edu/slides/winter1516_lecture5.pdf
		// Does not saturate in positive region
		// converges faster than sigmoid or tanh
		// 
		// ReLU outputs are not zero centered
		// => dead ReLU units will never activate
		class ReLU_Layer : public Layer
		{
		public:



			ReLU_Layer(unsigned int num_elements) : Layer(num_elements)
			{
				m_type = LAYER_TYPE::ReLU_Layer_type;
			}
			ReLU_Layer(unsigned int num_elements, unsigned int num_inputs) : Layer(num_elements, num_inputs)
			{
				m_type = LAYER_TYPE::ReLU_Layer_type;
			}

			ReLU_Layer(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
				: Layer(num_elements, num_inputs, in, out)
			{
				m_type = LAYER_TYPE::ReLU_Layer_type;
			}


			void Activation_Function(matrix& out, matrix& in)
			{
				out = in;

				for (int i = 0; i < out.NumRows(); i++)
				{
					for (int j = 0; j < out.NumCols(); j++)
					{
						if (out(i, j) < eps)
							out(i, j) = eps;
					}
				}
			}

			void Activation_Function_Deriv(matrix& out, matrix& in)
			{
				out = in;
				for (int i = 0; i < out.NumRows(); i++)
					for (int j = 0; j < out.NumCols(); j++)
						out(i, j) = out(i, j) > eps ? 1.0 : 0.0;
			}

		};


		// notes from: http://cs231n.stanford.edu/slides/winter1516_lecture5.pdf
		// Does not saturate
		// computationally efficient
		// converges faster than sigmoid or tanh (6x)
		// does not die
		// 
		// ReLU outputs are not zero centered

		class Leaky_ReLU_Layer : public Layer
		{
		public:
			Leaky_ReLU_Layer(unsigned int num_elements) : Layer(num_elements)
			{
				m_type = LAYER_TYPE::Leaky_ReLU_Layer_type;
			}
			Leaky_ReLU_Layer(unsigned int num_elements, unsigned int num_inputs) : Layer(num_elements, num_inputs)
			{
				m_type = LAYER_TYPE::Leaky_ReLU_Layer_type;
			}

			Leaky_ReLU_Layer(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
				: Layer(num_elements, num_inputs, in, out)
			{
				m_type = LAYER_TYPE::Leaky_ReLU_Layer_type;
			}


			void Activation_Function(matrix& out, matrix& in)
			{
				out = in;

				for (int i = 0; i < out.NumRows(); i++)
				{
					for (int j = 0; j < out.NumCols(); j++)
					{
						out(i, j) = max(0.1 * out(i, j), out(i, j));
					}
				}
			}

			void Activation_Function_Deriv(matrix& out, matrix& in)
			{
				out = in;

				for (int i = 0; i < out.NumRows(); i++)
				{
					for (int j = 0; j < out.NumCols(); j++)
					{
						if (out(i, j) < 0.0)
						{
							out(i, j) = 0.1;
						}
						else
						{
							out(i, j) = 1.0;
						}
					}
				}
			}

		};

		// notes from: http://cs231n.stanford.edu/slides/winter1516_lecture5.pdf
		// Does not saturate
		// computationally efficient
		// converges faster than sigmoid or tanh (6x)
		// does not die
		// back propogates into alpha
		// ReLU outputs are not zero centered

		class PReLU_Layer : public Layer
		{
		public:
			PReLU_Layer(unsigned int num_elements) : Layer(num_elements)
			{
				m_type = LAYER_TYPE::PReLU_Layer_type;
			}
			PReLU_Layer(unsigned int num_elements, unsigned int num_inputs) : Layer(num_elements, num_inputs)
			{
				m_type = LAYER_TYPE::PReLU_Layer_type;
			}

			PReLU_Layer(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
				: Layer(num_elements, num_inputs, in, out)
			{
				m_type = LAYER_TYPE::PReLU_Layer_type;
			}


			void Activation_Function(matrix& out, matrix& in)
			{
				out = in;

				for (int i = 0; i < out.NumRows(); i++)
				{
					for (int j = 0; j < out.NumCols(); j++)
					{
						out(i, j) = max(this->eps * out(i, j), out(i, j));
					}
				}
			}

			void Activation_Function_Deriv(matrix& out, matrix& in)
			{
				out = in;

				for (int i = 0; i < out.NumRows(); i++)
				{
					for (int j = 0; j < out.NumCols(); j++)
					{
						if (out(i, j) < 0.0)
						{
							out(i, j) = this->eps;
						}
						else
						{
							out(i, j) = 1.0;
						}
					}
				}
			}

		};




		// notes from: http://cs231n.stanford.edu/slides/winter1516_lecture5.pdf
		// All benefits of Leaky_ReLU
		// does not die
		// closer to zero mean outputs
		//
		// computation requires exp


		class ELU_Layer : public Layer
		{
		public:
			ELU_Layer(unsigned int num_elements) : Layer(num_elements)
			{
				m_type = LAYER_TYPE::ELU_Layer_type;
			}
			ELU_Layer(unsigned int num_elements, unsigned int num_inputs) : Layer(num_elements, num_inputs)
			{
				m_type = LAYER_TYPE::ELU_Layer_type;
			}

			ELU_Layer(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
				: Layer(num_elements, num_inputs, in, out)
			{
				m_type = LAYER_TYPE::ELU_Layer_type;
			}


			void Activation_Function(matrix& out, matrix& in)
			{
				out = in;

				for (int i = 0; i < out.NumRows(); i++)
				{
					for (int j = 0; j < out.NumCols(); j++)
					{
						if (out(i, j) < 0) 
							out(i, j) = this->eps * (exp(out(i, j)) - 1.0);
					}
				}
			}

			void Activation_Function_Deriv(matrix& out, matrix& in)
			{
				out = in;
				for (int i = 0; i < out.NumRows(); i++)
				{
					for (int j = 0; j < out.NumCols(); j++)
					{
						if (out(i, j) < 0.0)
						{
							out(i, j) = out(i, j) + this->alpha_;
						}
						else
						{
							out(i, j) = 1.0;
						}
					}
				}
			}
		};


		class SoftPlus_Layer : public Layer
		{
		public:
			SoftPlus_Layer(unsigned int num_elements) : Layer(num_elements)
			{
				m_type = LAYER_TYPE::SoftPlus_Layer_type;
			}
			SoftPlus_Layer(unsigned int num_elements, unsigned int num_inputs) : Layer(num_elements, num_inputs)
			{
				m_type = LAYER_TYPE::SoftPlus_Layer_type;
			}

			SoftPlus_Layer(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
				: Layer(num_elements, num_inputs, in, out)
			{
				m_type = LAYER_TYPE::SoftPlus_Layer_type;
			}


			void Activation_Function(matrix& out, matrix& in)
			{
				out = in;

				for (int i = 0; i < out.NumRows(); i++)
				{
					for (int j = 0; j < out.NumCols(); j++)
					{
						out(i, j) = log(1 + exp(out(i, j)));
					}
				}
			}

			void Activation_Function_Deriv(matrix& out, matrix& in)
			{
				sigmoid(out, in);
			}

		};
	
		class Softmax_Layer : public Layer
		{
		public:

			Softmax_Layer(unsigned int num_elements) : Layer(num_elements)
			{
				m_type = LAYER_TYPE::Softmax_Layer_type;
			}
			Softmax_Layer(unsigned int num_elements, unsigned int num_inputs) : Layer(num_elements, num_inputs)
			{
				m_type = LAYER_TYPE::Softmax_Layer_type;
			}

			Softmax_Layer(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
				: Layer(num_elements, num_inputs, in, out)
			{
				m_type = LAYER_TYPE::Softmax_Layer_type;
			}


			void Activation_Function(matrix& out, matrix& in)
			{
				out = in;
				out = exp(out);
				float sum = SUM(out);
				out = out / sum;
			}

			void Activation_Function_Deriv(matrix& out, matrix& in)
			{
				out =  in;
			}


			void BackPropogate_output(matrix& expected, matrix& out)
			{
				// predicted - actual
				deltas_ = expected - out;

#ifdef VERBOSE
				PrintDeltas();
#endif
			}


		};
		

		class Linear_Layer : public Layer
		{
		public:

			Linear_Layer(unsigned int num_elements) : Layer(num_elements)
			{
				m_type = LAYER_TYPE::Linear_Layer_type;
			}
			Linear_Layer(unsigned int num_elements, unsigned int num_inputs) : Layer(num_elements, num_inputs)
			{
				m_type = LAYER_TYPE::Linear_Layer_type;
			}

			Linear_Layer(unsigned int num_elements, unsigned int num_inputs, Link_Tensor *in, Link_Tensor *out)
				: Layer(num_elements, num_inputs, in, out)
			{
				m_type = LAYER_TYPE::Linear_Layer_type;
			}


			void Activation_Function(matrix& out, matrix& in)
			{
				out = in;
				
			}

			void Activation_Function_Deriv(matrix& out, matrix& in)
			{
				out = in;
				for (int i = 0; i < out.NumRows(); i++)
				{
					for (int j = 0; j < out.NumCols(); j++)
					{
						out(i, j) = 1.0;
					}
				}
			}


		};



		void Link_Tensor::Compute_Weight_Deltas(matrix &deltas_, float alpha, float beta)
		{
			delta_weights_.transpose();

			deltas_.transpose();

			delta_weights_ = delta_weights_ * beta + deltas_* this->prev->neurons_ * alpha;

			delta_weights_.transpose();

			deltas_.transpose();
		}

		void Link_Tensor::UpdateWeights()
		{
			weights_ = weights_ + delta_weights_;// 
		}



		class NeuralNetwork
		{
		public:
			NeuralNetwork(){
				input_layer = 0;
			}

			NeuralNetwork(vector<int> &layer_sizes){


				num_layers = layer_sizes.size();
				if (num_layers > 0)
				{
					input_layer = new Layer*[num_layers];

					input_layer[0] = new Sigmoid_Layer(layer_sizes[0], 1);
					input_layer[0]->connection_in = 0;
					for (int i = 1; i < num_layers-1; i++)
					{
						input_layer[i] = new Sigmoid_Layer(layer_sizes[i], layer_sizes[i - 1]);

						input_layer[i]->connection_in =
							new Link_Tensor(input_layer[i - 1],  // prev
							input_layer[i],      // next (this one)
							layer_sizes[i - 1], // num_inputs (nodes in prev)
							layer_sizes[i]);    // num_outputs (nodes in this)

						input_layer[i - 1]->connection_out = input_layer[i]->connection_in; // the last layers output connection is the input to this

					}
					int i = num_layers - 1;
					input_layer[i] = new Softmax_Layer(layer_sizes[i], layer_sizes[i - 1]);

					input_layer[i]->connection_in =
						new Link_Tensor(input_layer[i - 1],  // prev
						input_layer[i],      // next (this one)
						layer_sizes[i - 1], // num_inputs (nodes in prev)
						layer_sizes[i]);    // num_outputs (nodes in this)

					input_layer[i - 1]->connection_out = input_layer[i]->connection_in; // the last layers output connection is the input to this




					input_layer[num_layers - 1]->connection_out = 0; // or something ...

					// initialize the weights
					for (int i = 0; i < num_layers; i++)
					{
						if (input_layer[i]->connection_in)
							input_layer[i]->connection_in->init_random_sample_weights_sigmoid();
					}

				}
			}

			NeuralNetwork(vector<Layer*> &layers){


				num_layers = layers.size();
				if (num_layers > 0)
				{
					input_layer = new Layer*[num_layers];

					input_layer[0] = layers[0];
					input_layer[0]->connection_in = 0;
					for (int i = 1; i < num_layers; i++)
					{
						input_layer[i] = layers[i];

						input_layer[i]->connection_in = new Link_Tensor(
							input_layer[i - 1],  // prev
							input_layer[i],      // next (this one)
							input_layer[i - 1]->m_num_elements, // num_inputs (nodes in prev)
							input_layer[i]->m_num_elements);    // num_outputs (nodes in this)

						input_layer[i - 1]->connection_out = input_layer[i]->connection_in; // the last layers output connection is the input to this

					}
	
					input_layer[num_layers - 1]->connection_out = 0; // or something ...



				}
			}

			~NeuralNetwork()
			{
				// initialize the weights
				for (int i = 0; i < num_layers; i++)
				{
					if (input_layer[i]->connection_in)
						delete input_layer[i]->connection_in;
					delete input_layer[i];
				} 
				delete[] input_layer;
			}


			matrix& FeedForward(matrix& input)
			{
				input_layer[0]->neurons_ = input;

				for (int i = 1; i < num_layers - 1; i++)
				{
					input_layer[i]->FeedForward(input_layer[i - 1]->neurons_);
				}
				input_layer[num_layers - 1]->FeedForward(input_layer[num_layers - 2]->neurons_);

				return input_layer[num_layers - 1]->neurons_;

			}

			void BackPropagateErrors(matrix& expected, matrix& output)
			{
				input_layer[num_layers - 1]->BackPropogate_output(expected, output);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->BackPropogate();
				}

			}

			void BackPropagateErrors(matrix& errors)
			{
				input_layer[num_layers - 1]->BackPropogate_output(errors);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->BackPropogate();
				}

			}

			void ComputeDeltas(float alpha, float beta)
			{
				input_layer[num_layers - 1]->ComputeWeightDeltas(alpha, beta);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->ComputeWeightDeltas(alpha, beta);
				}
			}

			void UpdateWeights()
			{
				input_layer[num_layers - 1]->UpdateWeights();

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->UpdateWeights();
				}
			}

			// Note: Need to establish procedure for creating weights based on input data.
			//
			//
			void Initialize_Weights()
			{
				// initialize the weights
				for (int i = 0; i < num_layers; i++)
				{
					if (input_layer[i]->connection_in)
					{
						/*
								Linear_Layer
								Softmax_Layer
								SoftPlus_Layer
								ELU_Layer
								PReLU_Layer
								Leaky_ReLU_Layer
								ReLU_Layer
								Sigmoid_Layer
						*/
						switch (input_layer[i]->m_type)
						{
							case LAYER_TYPE::ELU_Layer_type:
							{
								input_layer[i]->connection_in->init_random_sample_weights_NearZero();
								break;
							}
							case LAYER_TYPE::Leaky_ReLU_Layer_type:
							{
								input_layer[i]->connection_in->init_random_sample_weights_NearZero();
								break;
							}
							case LAYER_TYPE::Linear_Layer_type:
							{
								input_layer[i]->connection_in->init_random_sample_weights_NearZero();
								break;
							}
							case LAYER_TYPE::PReLU_Layer_type:
							{
								input_layer[i]->connection_in->init_random_sample_weights_NearZero();
								break;
							}
							case LAYER_TYPE::ReLU_Layer_type:
							{
								input_layer[i]->connection_in->init_random_sample_weights_ToZero();
								break;
							}
							case LAYER_TYPE::Sigmoid_Layer_type:
							{
								input_layer[i]->connection_in->init_random_sample_weights_sigmoid();
								break;
							}
							case LAYER_TYPE::Softmax_Layer_type:
							{
								input_layer[i]->connection_in->init_random_sample_weights_sigmoid();
								break;
							}
							case LAYER_TYPE::SoftPlus_Layer_type:
							{
								input_layer[i]->connection_in->init_random_sample_weights_sigmoid();
								break;
							}
						}
					

						
					}
				}
			}

			// maybe change for a vector here
			int num_layers = 0;
			Layer **input_layer;

		};
	}




}







#endif