#pragma once
#include <vector>
#include <string>
#include "Litemath.h"

namespace MeshObj {
using namespace LiteMath;

struct Face {
	int3 v;
	int3 vn;
	Face() : v(0), vn(0) {}
	
};

struct Mesh {
	std::vector<float3> vertices;
	std::vector<float3> faceNormals;
	std::vector<Face> faces;
};

void loadMesh(const std::string_view filename, MeshObj::Mesh& mesh);
std::pair<float, bool> distanceToTriangle(const float3& point, const float3& v1, const float3& v2, const float3& v3);
float minimalDistanceToMesh(const MeshObj::Mesh& mesh, const float3& point);
void generatePointCloud(MeshObj::Mesh& mesh, std::vector<float>& points, std::vector<float>& dists, float3 bbmin, float3 bbmax, int resolution);
void savePointCloud(const std::string_view filename, std::vector<float>& points, std::vector<float>& dists);
void shuffleAligned(std::vector<float>& points, std::vector<float>& dists);
void save_text_points(const std::string_view filename, const std::vector<float>& points, const std::vector<float>& dists);

};