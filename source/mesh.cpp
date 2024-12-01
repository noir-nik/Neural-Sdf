#include <iostream>
#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>
#include <limits>
#include <cmath>
#include <random>
#include <algorithm> 
#include "mesh.h"

namespace MeshObj {
// Function to read mesh data from a wavefront file
inline float distance(const float3 a, const float3 b) { return length(a - b); }

void loadMesh(const std::string_view filename, Mesh& mesh) {
	std::ifstream file(filename.data());
	if (!file.is_open()) {
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		exit(1);
	}

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string token;
		iss >> token;

		if (token == "v") { // float3 data
			float3 vertex;
			iss >> vertex.x >> vertex.y >> vertex.z;
			mesh.vertices.push_back(vertex);
		} else 
		if (token == "vn") { // Face normal data
			float3 normal;
			iss >> normal.x >> normal.y >> normal.z;
			mesh.faceNormals.push_back(normal);
		} else 

		if (token == "f") { // Face data
			std::istringstream iss(line.substr(2)); // Remove the "f "
			std::string part;
			Face face;
			int num = 0;
			while (iss >> part) {
				int index = -1, texture_index = -1, normal_index = -1;
				std::istringstream sub_iss(part);
				std::string index_str;
				// index
				std::getline(sub_iss, index_str, '/');
				index = std::stoi(index_str); // Indices are 1-based in obj files
				// texture_index
				if (std::getline(sub_iss, index_str, '/') && index_str != "") {
					texture_index = std::stoi(index_str);
				}
				// normal_index
				if (std::getline(sub_iss, index_str, '/') && index_str != "") {
					normal_index = std::stoi(index_str);
				}
				face.v[num] = index;
				face.vn[num] = normal_index;
				num++;
				if (num == 3) break;
			}
			mesh.faces.push_back(face);
		}
	}

	file.close();
}

float distance_to_edge(const float3& point, const float3& v1, const float3& v2) {
	float edge_length = distance(v1, v2);
	if (edge_length == 0) {
		return distance(point, v1);
	}

	// Compute the projection of the point onto the edge
	float t = ((point.x - v1.x) * (v2.x - v1.x) + (point.y - v1.y) * (v2.y - v1.y) + (point.z - v1.z) * (v2.z - v1.z)) / (edge_length * edge_length);
	if (t <= 0) {
		return distance(point, v1); // Closest to v1
	} else if (t >= 1) {
		return distance(point, v2); // Closest to v2
	} else {
		// Closest to the projection point on the edge
		float3 projection;
		projection.x = v1.x + t * (v2.x - v1.x);
		projection.y = v1.y + t * (v2.y - v1.y);
		projection.z = v1.z + t * (v2.z - v1.z);
		return distance(point, projection);
	}
}

// Function to calculate the distance from a point to a triangle
std::pair<float, bool> distanceToTriangle(const float3& point, const float3& v1, const float3& v2, const float3& v3) {

	// Calculate the normal vector of the triangle
	float3 normal;
	normal.x = (v2.y - v1.y) * (v3.z - v1.z) - (v2.z - v1.z) * (v3.y - v1.y);
	normal.y = (v2.z - v1.z) * (v3.x - v1.x) - (v2.x - v1.x) * (v3.z - v1.z);
	normal.z = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x);

	// precompute
	float normal_len_inverse = 1.0f / std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
	normal.x *= normal_len_inverse;
	normal.y *= normal_len_inverse;
	normal.z *= normal_len_inverse;
	
	// Calculate the distance from the point to the plane of the triangle
	float distanceToPlane = ((point.x - v1.x) * normal.x +
							 (point.y - v1.y) * normal.y +
							 (point.z - v1.z) * normal.z) /
							std::sqrt(normal.x * normal.x +
									  normal.y * normal.y +
									  normal.z * normal.z);
                                      
	// Project the point onto the plane of the triangle
	float3 projectedPoint;
	projectedPoint.x = point.x - distanceToPlane * normal.x;
	projectedPoint.y = point.y - distanceToPlane * normal.y;
	projectedPoint.z = point.z - distanceToPlane * normal.z;

	float3 vv0 = v2 - v1, vv1 = v3 - v1, vv2 = projectedPoint - v1;
	float d00 = dot(vv0, vv0);
	float d01 = dot(vv0, vv1);
	float d11 = dot(vv1, vv1);
	float d20 = dot(vv2, vv0);
	float d21 = dot(vv2, vv1);
	float denom = d00 * d11 - d01 * d01;
	float v, w, u;
	v = (d11 * d20 - d01 * d21) / denom;
	w = (d00 * d21 - d01 * d20) / denom;
	u = 1.0f - v - w;
	float& alpha = v, beta = w, gamma = u;

	bool outside_mesh = distanceToPlane > 0.000001 ? true : false;
	if (alpha >= 0 && beta >= 0 && gamma >= 0){
		return std::make_pair(std::abs(distanceToPlane), outside_mesh);
	}

	// Distance from point to edge
	// Calculate the distance from the point to each vertex of the triangle
	float d1 = distance(point, v1);
	float d2 = distance(point, v2);
	float d3 = distance(point, v3);
	std::pair<float, float3> pv1 = std::make_pair(d1, v1);
	std::pair<float, float3> pv2 = std::make_pair(d2, v2);
	std::pair<float, float3> pv3 = std::make_pair(d3, v3);

	// Sort the vertices by distance
   if (pv1.first > pv2.first) {
	   std::swap(pv1, pv2);
   } 
   if (pv1.first > pv3.first) {
	   std::swap(pv1, pv3);
   } 
   if (pv2.first > pv3.first) {
	   std::swap(pv2, pv3);
   }
	float d = distance_to_edge(point, pv1.second, pv2.second);
	return std::make_pair(d, outside_mesh);
}

// Function to find the minimal distance from a point to the mesh
float minimalDistanceToMesh(const Mesh& mesh, const float3& point) {
	std::pair<float, bool> dist, minDistance(std::numeric_limits<float>::max(), true);
	for (const auto& face : mesh.faces) {
		const float3& v1 = mesh.vertices[face.v[0] - 1]; // float3 indices are 1-based
		const float3& v2 = mesh.vertices[face.v[1] - 1];
		const float3& v3 = mesh.vertices[face.v[2] - 1];

		dist = distanceToTriangle(point, v1, v2, v3);
		//std::cout << "Dist: " << dist.first << std::endl;
		if (dist.first < minDistance.first)
			minDistance = dist;
	}
	//std::cout << minDistance.second << std::endl;
	if(minDistance.second) {
		//std::cout << "Outside mesh" << std::endl;
		return minDistance.first;
	} else {
		//std::cout << "Inside mesh" << std::endl;
		return minDistance.first * -1;
	}
}

void generatePointCloud(Mesh& mesh, std::vector<float>& points, std::vector<float>& dists, float3 bbmin, float3 bbmax, int resolution) {
	float step = (bbmax.x - bbmin.x) / resolution;
	dists.resize(resolution * resolution * resolution);
	points.resize(resolution * resolution * resolution * 3);

	for (int i = 0; i < resolution; i++) {
		for (int j = 0; j < resolution; j++) {
			for (int k = 0; k < resolution; k++) {
				float3 point = {bbmin.x + i * step, bbmin.y + j * step, bbmin.z + k * step};
				points[i * resolution * resolution * 3 + j * resolution * 3 + k * 3] = point.x;
				points[i * resolution * resolution * 3 + j * resolution * 3 + k * 3 + 1] = point.y;
				points[i * resolution * resolution * 3 + j * resolution * 3 + k * 3 + 2] = point.z;
				dists[i * resolution * resolution + j * resolution + k] = minimalDistanceToMesh(mesh, point);
			}
		}
	}
	
}

void savePointCloud(const std::string_view filename, std::vector<float>& points, std::vector<float>& dists) {
	// Save binary file
	std::ofstream file(filename.data(), std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	// Write N number as int
	int N = points.size() / 3;
	file.write(reinterpret_cast<const char*>(&N), sizeof(int));
	file.write(reinterpret_cast<const char*>(points.data()), sizeof(float) * points.size());
	file.write(reinterpret_cast<const char*>(dists.data()), sizeof(float) * dists.size());
	file.close();
	std::cout << "Points and distances saved to file: " << filename << std::endl;
}

void shuffleAligned(std::vector<float>& points, std::vector<float>& dists) {
	// Create an index array to shuffle
	std::vector<int> indices(points.size() / 3);
	for (int i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}

	// Shuffle the indices
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(indices.begin(), indices.end(), g);

	// Use the shuffled indices to shuffle both points and dists vectors
	std::vector<float> shuffledPoints(points.size());
	std::vector<float> shuffledDists(dists.size());

	for (int i = 0; i < indices.size(); ++i) {
		int idx = indices[i] * 3;
		int distIdx = indices[i];
		std::copy(points.begin() + idx, points.begin() + idx + 3, shuffledPoints.begin() + i * 3);
		shuffledDists[i] = dists[distIdx];
	}
	points = shuffledPoints;
	dists = shuffledDists;
}

// save as text to new file
void save_text_points(const std::string_view filename, const std::vector<float>& points, const std::vector<float>& dists){
	std::ofstream text_points(filename.data());
	if (!text_points.is_open()) {
		std::cerr << "Error saving text file: " << filename << std::endl;
		exit(1);
	}
	for (int i = 0; i < points.size() / 3; ++i) {
		text_points << points[i * 3] << " " << points[i * 3 + 1] << " " << points[i * 3 + 2] << " " << dists[i] << std::endl;
	}
	text_points.close();
}

} // namespace MeshObj