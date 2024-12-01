#ifndef SETTINGS_H
#define SETTINGS_H

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 128

#define OUTPUT_FOLDER "output"

constexpr int point_cloud_resolution = 32; // 32^3 total points

constexpr int num_iterations = 15000;
constexpr int batch_size = 512;

constexpr float learning_rate = 2e-5f;
constexpr float momentum = 0.85f;
constexpr float stop_threshold = 0.0002f;
constexpr float weight_decay = 0.7f;
constexpr int weight_decay_interval = 1000;

#endif