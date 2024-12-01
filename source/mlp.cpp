
#include <iostream>
#include <fstream>
#include <string_view>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <chrono>
#include <omp.h>
#include "optimize.h"
#include "settings.h"
#include "mlp.h"

#define W_0 30.0f


MLP::MLP(const int num_hidden_layers, const int hidden_layer_size) : num_hidden_layers(num_hidden_layers), hidden_layer_size(hidden_layer_size), NHL(num_hidden_layers), size(hidden_layer_size) {
	// Initialize MLP
	// -------------- W ------------------------------- b -----------------
	num_parameters = (3*size + NHL*size*size + size) + (size*(NHL + 1) + 1);
	activations_size = 3 + (NHL + 1)*size;
	zs_size = (NHL + 1)*size + 1;
	// shapes
	set_shape(3, 1, hidden_layer_size, num_hidden_layers);
	std::random_device rd;
	std::mt19937 gen(rd());
	// w ~ U(-sqrt(6 / in), sqrt(6 / in))
	w_b.resize(num_parameters, 0.01f);
	int pos = 0;
	for (int i = 0; i < num_hidden_layers + 2; ++i) {
		std::uniform_real_distribution<float> dis1(-std::sqrt(6.0f / (shapes[i])), std::sqrt(6.0f / (shapes[i])));
		for (int j = 0; j < shapes[i]*shapes[i+1]; ++j) {
			w_b[pos + j] = dis1(gen)/ W_0;
		}
		for (int j = 0; j < shapes[i+1]; ++j) {
			w_b[pos + shapes[i] * shapes[i+1] + j] = dis1(gen);
		}
		int pos1 = pos;
		pos += (shapes[i] + 1)*shapes[i+1];
	}

}

void MLP::set_shape(size_t input_shape, size_t output_shape, size_t hidden_layer_shape, size_t num_hidden_layers) {
	shapes.reserve(num_hidden_layers + 2);
	shapes.push_back(3); // input shapes
	for (int i = 0; i < num_hidden_layers + 1; ++i) {
		shapes.push_back(size);
	}
	shapes.push_back(1); // output shapes
}


void MLP::showLayers(){
	std::cout << "Layer " << 0 << ": ";
	std::cout << 3 << " -> " << hidden_layer_size << std::endl;
	for (int i = 0; i < num_hidden_layers; ++i) {
		std::cout << "Layer " << i + 1 << ": ";
		std::cout << hidden_layer_size << " -> " << hidden_layer_size << std::endl;
	}
	std::cout << "Layer " << num_hidden_layers + 1 << ": ";
	std::cout << hidden_layer_size << " -> " << 1 << std::endl;
}


// Read points
void MLP::readPoints(const std::string_view filename, int& N, std::vector<float>& points, std::vector<float>& dists) {

	std::ifstream file(filename.data(), std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	file.read(reinterpret_cast<char*>(&N), sizeof(int));
	points.resize(N * 3), dists.resize(N);
	file.read(reinterpret_cast<char*>(points.data()), sizeof(float) * 3 * N);
	file.read(reinterpret_cast<char*>(dists.data()), sizeof(float) * N);
	if (!file) {
		std::cerr << "Error reading data from test file: " << filename << std::endl;
		exit(1);
	}
	std::cout << "Read " << N << " points from " << filename << std::endl;

	file.close();
}

// Read weights and biases
bool MLP::readWeights(const std::string_view filename) {

	std::ifstream file(filename.data(), std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		return false;
	}

	file.seekg(0, std::ios::end);
	std::streampos fileSize = file.tellg();
	file.seekg(0, std::ios::beg);

	std::size_t numFloats = fileSize / sizeof(float);

	if (num_parameters != numFloats) {
		std::cerr << "Error: Failed to read weights from file: " << filename << std::endl;
		std::cerr << "Number of weights and biases do not match the hidden layers` shapes: " << num_hidden_layers << " x " << hidden_layer_size << std::endl;
		std::cerr << "Required number of weights and biases: " << num_parameters << ", got " << numFloats << " from " << filename << std::endl;
		return false;
	}

	file.read(reinterpret_cast<char*>(w_b.data()), numFloats * sizeof(float));
	file.close();
	std::cout << "Weights and biases read from file: " << filename << std::endl;
	return true;
}

// Test
float MLP::test(const std::string_view filename, int N) {
	std::vector<float> poses;
	std::vector<float> dists;
	int M;
	readPoints(filename, M, poses, dists);
	float output = test(poses, dists, N);
	return output;
}

constexpr int test_iterations = 10000;
float MLP::test(std::vector<float> &points, std::vector<float> &dists, int N) {
	std::chrono::high_resolution_clock::time_point start, end;
	std::chrono::duration<double> elapsed;
	if (N <= 0 || N > points.size() / 3) N = points.size() / 3;
	float output = 0.0f;
	start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+:output)
	for (int i = 0; i < test_iterations; ++i) {
		float input[3];
		input[0] = points[3 * i];
		input[1] = points[3 * i + 1];
		input[2] = points[3 * i + 2];
		float out_dist = forward_light(input);

		output += (out_dist - dists[i]) * (out_dist - dists[i]);
		std::cout << "dist: " << dists[i] << " out_dist: " << out_dist << std::endl;

	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	std::cout << "Test: " << output << " Time: " << elapsed.count() << std::endl;
	return output;
}
	
float MLP::forward(const float input[3], float* activations, float* zs) {
	
	activations[0] = input[0];
	activations[1] = input[1];
	activations[2] = input[2];

	// zs = 0;
	for (int i = 0; i < (NHL + 1)*hidden_layer_size + 1; ++i) {
		zs[i] = 0.0f;
	}
	//return 1.0f;
	// ===== Layer 0 =====
	// z = Wx + b
	for (int i = 0; i < hidden_layer_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			zs[i] += w_b[i * 3 + j] * activations[j];
		}
	}
	// + b
	for (int i = 0; i < hidden_layer_size; ++i) {
		zs[i] += w_b[hidden_layer_size * 3 + i];
	}
	// sin
	for (int i = 0; i < hidden_layer_size; ++i) {
		activations[3 + i] = sin(W_0 * zs[i]);
	}

	// ===== Hidden layers =====
	int w_b_pos = 4 * hidden_layer_size;
	int zs_pos = hidden_layer_size;
	int activations_pos = 3;
	for (int l = 0; l < num_hidden_layers; ++l) {
		// z = Wx + b
		for (int i = 0; i < hidden_layer_size; ++i) {
			for (int j = 0; j < hidden_layer_size; ++j) {
				zs[zs_pos + i] += activations[activations_pos + j] * w_b[w_b_pos + i * hidden_layer_size + j];
			}
		}
		// + b
		for (int i = 0; i < hidden_layer_size; ++i) {
			zs[zs_pos + i] += w_b[w_b_pos + hidden_layer_size * hidden_layer_size + i];
		}
		// sin
		activations_pos += hidden_layer_size;
		for (int i = 0; i < hidden_layer_size; ++i) {
			activations[activations_pos + i] = sin(W_0 * zs[zs_pos + i]);
		}
		zs_pos += hidden_layer_size;
		w_b_pos += (hidden_layer_size + 1) * hidden_layer_size;
	}
		
	// ===== Output layer =====
	// z = Wx + b
	for (int j = 0; j < hidden_layer_size; ++j) {
		zs[(num_hidden_layers + 1)*hidden_layer_size] += activations[activations_pos + j] * w_b[w_b_pos + j];
	}
	// + b
	zs[(num_hidden_layers + 1)*hidden_layer_size] += w_b[w_b_pos + hidden_layer_size];

	return zs[(num_hidden_layers + 1)*hidden_layer_size];
	//return zs[0];
}

float MLP::forward_light(const float input[3]) {
	
	float activations[hidden_layer_size * 2];

	// zero out;
	for (int i = 0; i < 2*hidden_layer_size; ++i) {
		activations[i] = 0.0f;
	}

	activations[0] = input[0];
	activations[1] = input[1];
	activations[2] = input[2];
	int half = 0;

	// ===== Layer 0 =====
	// z = Wx + b
	for (int i = 0; i < hidden_layer_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			activations[hidden_layer_size * ((half + 1)%2) + i] += w_b[i * 3 + j] * activations[hidden_layer_size * ((half)%2) + j];
		}
	}
	
	// + b
	for (int i = 0; i < hidden_layer_size; ++i) {
		activations[hidden_layer_size * ((half + 1)%2) + i] += w_b[hidden_layer_size * 3 + i];
	}

	// sin
	for (int i = 0; i < hidden_layer_size; ++i) {
		activations[hidden_layer_size * ((half + 1)%2) + i] = sin(W_0 * activations[hidden_layer_size * ((half + 1)%2) + i]);
	}
	

	// ===== Hidden layers =====
	int w_b_pos = 4 * hidden_layer_size;
	//int zs_pos = hidden_layer_size;
	//int activations_pos = 3;
	for (int l = 0; l < num_hidden_layers; ++l) {
		//zero
		for (int i = 0; i < hidden_layer_size; ++i) {
			activations[(hidden_layer_size * (half % 2)) + i] = 0.0f;
		}
		// z = Wx + b
		for (int i = 0; i < hidden_layer_size; ++i) {
			for (int j = 0; j < hidden_layer_size; ++j) {
				activations[(hidden_layer_size * (half % 2)) + i] += activations[(hidden_layer_size * ((half + 1) % 2)) + j] * w_b[w_b_pos + i * hidden_layer_size + j];
			}
		}
		// + b
		for (int i = 0; i < hidden_layer_size; ++i) {
			activations[(hidden_layer_size * (half % 2)) + i] += w_b[w_b_pos + hidden_layer_size * hidden_layer_size + i];
		}
		// sin
		for (int i = 0; i < hidden_layer_size; ++i) {
			activations[(hidden_layer_size * (half % 2)) + i] = sin(W_0 * activations[(hidden_layer_size * (half % 2)) + i]);
		}
		w_b_pos += (hidden_layer_size + 1) * hidden_layer_size;
		half = (half + 1) % 2;
	}

	// ===== Output layer =====
	//zero
	activations[(hidden_layer_size * (half % 2))] = 0.0f;
	// z = Wx + b
	for (int j = 0; j < hidden_layer_size; ++j) {
		activations[hidden_layer_size * (half % 2)] += activations[hidden_layer_size*((half + 1) % 2) + j] * w_b[w_b_pos + j];
	}
	// + b
	activations[hidden_layer_size * (half % 2)] += w_b[w_b_pos + hidden_layer_size];
	return activations[hidden_layer_size * (half % 2)];
}

// Backward propagation
void MLP::backward(const float target, float* activations, float* zs, float* gradient, float* error, int offset, int batch_size) {
	int activations_index = (3 + num_hidden_layers * hidden_layer_size) + (activations_size * offset);
	int zs_index = (num_hidden_layers * hidden_layer_size) + (zs_size * offset);
	int error_index = (2 * hidden_layer_size * offset);
	int pos_in_grad = num_parameters + (num_parameters * offset);
	pos_in_grad -= (hidden_layer_size + 1);
	
	// Last layer
	float delta_last = (zs[(zs_size * offset) + zs_size - 1] - target) / batch_size;

	// Last layer W
	for (int j = 0; j < hidden_layer_size; ++j) {
		// dW_l = a_l-1 * delta_l
		gradient[pos_in_grad + j] += activations[activations_index + j] * delta_last;
	}
	// Last layer B
	gradient[pos_in_grad + hidden_layer_size] += delta_last;
	error[error_index + ((num_hidden_layers + 1)%2)*hidden_layer_size] = delta_last; //// ----0---
	// skip to *b of l-1
	pos_in_grad -= (hidden_layer_size);
	activations_index -= hidden_layer_size;
	
	int p_w_b_next = num_parameters - (hidden_layer_size + 1); // *W_last
	for (int l = num_hidden_layers; l >= 0; --l) {
		// zero
		for (int j = 0; j < hidden_layer_size; ++j) {
			error[error_index + (l%2)*hidden_layer_size + j] = 0.0f;
		}
		// delta_l = (W_l+1)^T * delta_l+1 * (W_0 * cos(W_0 * z_l))
		for (int i = 0; i < shapes[l + 1]; ++i) { // (L + 1) Input shape
			for (int j = 0; j < shapes[l + 2]; ++j) { // (L + 1) Output shape
				// delta_l <-> error <-> gradient_biases
				int p_w_l_1 = p_w_b_next + j*shapes[l+1] + i;
				//------------ pos ------------------- i ----------------------------------- delta^l+1 -----b^l------------------w^l+1
				error[error_index + i +(l%2)*hidden_layer_size] += w_b[p_w_l_1] * error[error_index + ((l+1) % 2)*hidden_layer_size + j] * (W_0 * std::cos(W_0 * zs[zs_index + i]));
			}
			gradient[pos_in_grad + i] += error[error_index + i +(l%2)*hidden_layer_size];
		}

		int delta_l_index = pos_in_grad;
		pos_in_grad -= shapes[l] * shapes[l + 1];
		for (int i = 0; i < shapes[l+1]; ++i) {
			for (int j = 0; j < shapes[l]; ++j) {
				//-------------------------------------------------------------------------------- delta_l ---------------
				gradient[pos_in_grad + i*shapes[l] + j] += activations[activations_index + j] * error[error_index + i +(l%2)*hidden_layer_size];
			}
		}
		if(l > 0){
			activations_index -= shapes[l-1];
			// -> *b^l-1
			pos_in_grad -= shapes[l];
		}
		zs_index -= shapes[l+1];
		p_w_b_next -= shapes[l+1] * (shapes[l] + 1);

	}
}

void MLP::train(std::vector<float>& points, std::vector<float>& dists, int num_iters, float alpha, int Num_samples, int batch_size, float stop_threshold) {
	if(points.size() < 3 || dists.size() < 1) {
		std::cerr << "Points and dists must have at least 3 elements" << std::endl;
		exit(1);
	}
	//int N = points.size() / 3;
	float* data = points.data();
	float* labels = dists.data();
	int N = Num_samples;
	if(batch_size < 1 || batch_size >= N) batch_size = N;
	size_t count = N;
	size_t input_size = 3;
	size_t output_size = 1;
	
	std::vector<float> in_batch(input_size*batch_size);
    std::vector<float> out_batch(output_size*batch_size);
	if(stop_threshold == -1.0f) {
		stop_threshold = 1e-13f;
	}

	int max_threads = omp_get_max_threads();
	int num_threads = max_threads;
	std::vector<float> activations(activations_size * num_threads, 0.0f);
	std::vector<float> zs(zs_size * num_threads, 0.0f);
	std::vector<float> error (hidden_layer_size * 2 * num_threads, 0.0f);
	int init_pos = 0;
	float loss = 0.0f;

	//Optimizer
	Optimizer<float, OptimizerType::Adam> adam(alpha, momentum);
	adam.setSize(num_parameters, num_parameters * num_threads);
	std::vector<float>& gradient = adam.m_gradient;

	float input[3] = {points[3 * init_pos], points[3 * init_pos + 1], points[3 * init_pos + 2]};
	float fw = forward(input, activations.data(), zs.data());
	std::cout << "Initial loss: " << (fw - dists[init_pos]) * (fw - dists[init_pos]) * 0.5f << std::endl;
	
	int points_position = 0;
	for (int i = 0; i < num_iters; ++i) {

		for (int j=0;j<batch_size;j++){
			unsigned b_id = rand()%count;
			memcpy(in_batch.data() + j*input_size, data + b_id*input_size, sizeof(float)*input_size);
			memcpy(out_batch.data() + j*output_size, labels + b_id*output_size, sizeof(float)*output_size);
		}

		adam.zero_grad();
		loss = 0.0f;
		std::vector<int> thread_iterations(num_threads, 0);
		std::vector<float> thread_loss(num_threads, 0.0f);
		float out_0;
#pragma omp parallel for num_threads(num_threads)
		for (int j = 0; j < batch_size; ++j) {
			int tid = omp_get_thread_num();
			int position = thread_iterations[tid]*num_threads + tid;
			if (position >= batch_size) {
				continue;
			}
			float inputs[3] = {in_batch[3 * position], in_batch[3 * position + 1], in_batch[3 * position + 2]};
			float output = forward(inputs, activations.data() + tid*activations_size, zs.data() + tid*zs_size);
			if(position == 0) {
				out_0 = output;
			}

			float loss_tmp = 0.5f * (output - out_batch[position]) * (output - out_batch[position]);

			thread_loss[tid] += loss_tmp;
			float target = out_batch[position];
			this->backward(target, activations.data(), zs.data(), gradient.data(), error.data(), tid, batch_size);
			thread_iterations[tid]++;		
		}
		
		//// reduction sequential
		//for (int t = 1; t < num_threads; ++t) {
		//	for (int j = 0; j < num_parameters; ++j) {
		//		gradient[j] += gradient[num_parameters * t + j];
		//	}
		//}

		// reduction parallel
		int step = 1;
		while (step < num_threads) {
			#pragma omp parallel for
			for (int i = 0; i < num_threads; i += 2 * step) {
				if (i + step < num_threads) {
					for (int j = 0; j < num_parameters; ++j) {
						gradient[i * num_parameters + j] += gradient[(i + step) * num_parameters + j];
					}
				}
			}
			step *= 2;
			#pragma omp barrier
		}

		// loss reduction
		for (int i = 0; i < num_threads; ++i) {
			loss += thread_loss[i];
		}

		// Regularization
        if (i % weight_decay_interval == 0) {
            adam.set_learning_rate(adam.get_learning_rate() * weight_decay);
        }
		//// update weights
		adam.update(w_b.data());

		loss /= batch_size;
		if(i % 50 == 0) {
			std::cout << "Iteration " << i << ": loss = " << loss << std::endl;
		}
		if(loss < stop_threshold) {
			std::cout << "Converged after " << i << " iterations, loss = " << loss << std::endl;
			break;
		}
	}
}

// Save weights and biases
void MLP::saveWeights(const std::string_view filename) {
	std::ofstream file(filename.data(), std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		exit(1);
	}

	file.write(reinterpret_cast<const char*>(this->w_b.data()), (num_parameters) * sizeof(float));
	file.close();
	std::cout << "Weights saved to file:      " << filename << std::endl;
	std::cout << "Total number of parameters: " << num_parameters << std::endl;
	std::cout << "Number of hidden layers:    " << num_hidden_layers << std::endl;
	std::cout << "Layer size:                 " << hidden_layer_size << std::endl;
}

