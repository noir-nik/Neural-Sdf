#pragma once
#include <string>
#include <vector>

#define W_0 30.0f

class MLP {
public:
	MLP(const int num_hidden_layers, const int hidden_layer_size);
	void set_shape(size_t input_shape, size_t output_shape, size_t hidden_layer_shape, size_t num_hidden_layers);
	void showLayers();
	void readPoints(const std::string_view filename, int& N, std::vector<float>& points, std::vector<float>& dists);
	bool readWeights(const std::string_view filename);
	float test(const std::string_view filename, int N = 0);
	float test(std::vector<float> &points, std::vector<float> &dists, int N = 0);
	float forward(const float input[3], float* activations, float* zs);
	float forward_light(const float input[3]);
	void backward(const float target, float* activations, float* zs, float* gradient, float* error, int offset, int batch_size = 1);
	void train(std::vector<float>& points, std::vector<float>& dists, int num_iters, float alpha, int Num_samples, int batch_size = -1, float stop_threshold = -1.0f);
	void saveWeights(const std::string_view filename);

    public:
    std::vector<float> w_b;
    std::vector<int> shapes;
    int num_hidden_layers, NHL;
    int hidden_layer_size, size;
    int num_parameters;
    int activations_size = 3 + (NHL + 1)*size;
    int zs_size = (NHL + 1)*size + 1;
};

