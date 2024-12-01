
#include <iostream>
#include "fragment.h"
#include "application.h"
#include <chrono>

void renderCPU(MLP* mlp, int width, int height, std::vector<uint32_t>& pixelData) {
	model_frag = mlp;
	std::chrono::high_resolution_clock::time_point start, end;
	std::chrono::duration<double> elapsed;
	start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float4 color = fragment(float2(j, i), float2(width, height));
			pixelData[i * width + j] = RealColorToUint32(color);
		}
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	std::cout << "Render time: " << elapsed.count() << std::endl;
}

void renderGPU(MLP* mlp, int width, int height, std::vector<uint32_t>& pixelData) {
	ComputeApplication app;
	
	app.mlp.w_b = mlp->w_b;
	app.mlp.num_hidden_layers = mlp->num_hidden_layers;
	app.mlp.hidden_layer_size = mlp->hidden_layer_size;
	std::chrono::high_resolution_clock::time_point start, end;
	std::chrono::duration<double> elapsed;
	start = std::chrono::high_resolution_clock::now();
	try {
		app.run();
	} catch (const std::runtime_error& e) {
		printf("%s\n", e.what());
		
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	std::cout << "Gpu Render time: " << elapsed.count() << std::endl;
}