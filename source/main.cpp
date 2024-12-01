#include <signal.h>
#include <chrono>
#include <filesystem>

#include <iostream>
#include <getopt.h>
#include <string_view>


#include "settings.h"
#include "fragment.h"
#include "save_bmp.h"


#include "mlp.h"
#include "mesh.h"
namespace fs = std::filesystem;

const char* out_weights_default = OUTPUT_FOLDER"/out_weights.bin";
std::string_view out_weights = out_weights_default;
std::string_view input_file;
std::string_view weights_file;

int num_layers = 0;
int layer_size = 0;

enum class Mode { Train, Render };

std::shared_ptr<MLP> mlp = nullptr;
int main(int argc, char *argv[]) {

	auto print_usage = [](const char* msg) {
		std::cerr << msg << std::endl;
		std::cerr << "Usage: ./neural_sdf --train -i <input.obj> -n <layers> -s <layer_size> -o <out_weights>" << std::endl;
		std::cerr << "Usage: ./neural_sdf --render -m <path_to_weights> -n <layers> -s <layer_size>" << std::endl;
	};

	if (argc < 2 || (std::string_view(argv[1]) != "--train" && std::string_view(argv[1]) != "--render")) {
		print_usage("Invalid command");
		return -1;
	}

	Mode mode = std::string_view(argv[1]) == "--train" ? Mode::Train : Mode::Render;
	int opt;
	auto get_num_from_argv = [](const char* opt, const char* optarg) {
		unsigned long val;
		char* endptr;
		errno = 0;
		val = strtoul(optarg, &endptr, 10);
		if (errno == ERANGE || endptr == optarg || *endptr != '\0') {
			std::cerr << "Error: Invalid number: " << optarg << " for option: " << opt << std::endl;
			exit(-1);
		}
		return val;
	};

	std::string_view opt_str = mode == Mode::Train ? "i:n:s:o:" : "m:n:s:";
	while ((opt = getopt(argc - 1, argv+1, opt_str.data())) != -1) {
		switch (opt) {
		case 'i':
			input_file = optarg;
			break;
		case 'n':
			num_layers = get_num_from_argv("-n", optarg);
			break;
		case 's':
			layer_size = get_num_from_argv("-s", optarg);
			break;
		case 'o':
			out_weights = optarg;
			break;
		case 'm':
			weights_file = optarg;
			break;
		default:
			print_usage("Invalid option");
			return -1;
		}
	}
	
	if (mode == Mode::Train) {
		if (num_layers == 0 || layer_size == 0) {
			print_usage("Invalid number of layers or layer size");
			return -1;
		}
		// Save weights if signal is received
		signal(SIGINT, [](int signal) {
			if (mlp != nullptr) {
				if (fs::path(out_weights).has_parent_path() && !fs::exists(fs::path(out_weights).parent_path())) {
					fs::create_directories(fs::path(out_weights).parent_path());
				}
				mlp->saveWeights(out_weights);
			}
			exit(0);
		});

		std::chrono::high_resolution_clock::time_point start, end;
		std::chrono::duration<double> elapsed;
		std::vector<float> points, dists;
		int N;
		MeshObj::Mesh mesh;

		std::string filename_name = fs::path(input_file).filename().string();
		if (input_file.substr(input_file.find_last_of(".") + 1) != "obj") {
			std::cerr << "Error: Input file must be an .obj file." << std::endl;
			return -1;
		}
		mlp = std::make_shared<MLP>(num_layers, layer_size);

		// Load mesh
		loadMesh(input_file, mesh);
		std::cout << "Mesh loaded: " << filename_name << std::endl;
		std::cout << "Generating point cloud from: " << filename_name << std::endl;
		generatePointCloud(mesh, points, dists, {-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}, point_cloud_resolution);
		N = points.size() / 3;
		std::cout << "Generated " << N << " points from " << filename_name << std::endl;
		std::cout << "Training..." << std::endl;
		start = std::chrono::high_resolution_clock::now();
		mlp->train(points, dists, num_iterations, learning_rate, N, batch_size, stop_threshold);
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
		std::cout << "Training time: " << elapsed.count() << "s" << std::endl;
		if (fs::path(out_weights).has_parent_path() && !fs::exists(fs::path(out_weights).parent_path())) {
			fs::create_directories(fs::path(out_weights).parent_path());
		}
		mlp->saveWeights(out_weights);
	}

	if (mode == Mode::Render) {
		if (weights_file.empty()) {
			std::cerr << "Error: No weights file provided." << std::endl;
			return -1;
		}

		if (num_layers == 0 || layer_size == 0) {
			print_usage("Invalid number of layers or layer size");
			return -1;
		}
		std::vector<uint32_t> pixelData(SCREEN_WIDTH * SCREEN_HEIGHT, 0);
		mlp = std::make_shared<MLP>(num_layers, layer_size);
		if (!mlp->readWeights(weights_file)) return -1;
		std::cout << "Cpu rendering starts..." << std::endl;
		renderCPU(mlp.get(), SCREEN_WIDTH, SCREEN_HEIGHT, pixelData);
		std::string out_name = OUTPUT_FOLDER"/out_cpu.bmp";
		generateBitmapImage(pixelData.data(), SCREEN_HEIGHT, SCREEN_WIDTH, out_name.c_str(), true);
		renderGPU(mlp.get(), SCREEN_WIDTH, SCREEN_HEIGHT, pixelData); 
		std::cout << "Images saved to: " << OUTPUT_FOLDER"/" << std::endl;
	}
	return 0;
}
