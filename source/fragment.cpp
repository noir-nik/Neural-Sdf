#include "fragment.h"
#include "mlp.h"

# define fov 90.f
# define itfov 1.0/(tan(3.141592654*fov/(180.*2.)))

#define CAMERA_FOV      	itfov

#define EPS     	0.005f
#define MAX_DIST     	10.f
#define MAX_STEPS		110

MLP* model_frag = nullptr;

float2 RayBoxIntersection(float3 ray_pos, float3 ray_dir, float3 boxMin, float3 boxMax) {
	ray_dir.x = 1.0f / ray_dir.x;
	ray_dir.y = 1.0f / ray_dir.y;
	ray_dir.z = 1.0f / ray_dir.z;

	float lo = ray_dir.x * (boxMin.x - ray_pos.x);
	float hi = ray_dir.x * (boxMax.x - ray_pos.x);

	float tmin = min(lo, hi);

	float tmax = max(lo, hi);

	float lo1 = ray_dir.y * (boxMin.y - ray_pos.y);
	float hi1 = ray_dir.y * (boxMax.y - ray_pos.y);

	tmin = max(tmin, min(lo1, hi1));
	tmax = min(tmax, max(lo1, hi1));

	float lo2 = ray_dir.z * (boxMin.z - ray_pos.z);
	float hi2 = ray_dir.z * (boxMax.z - ray_pos.z);

	tmin = max(tmin, min(lo2, hi2));
	tmax = min(tmax, max(lo2, hi2));

	return float2(tmin, tmax);
}


float3x3 myLookAt(float3 cameraPos, float3 lookAtPoint, float3 up) {
    float3 cd = normalize(lookAtPoint - cameraPos); // camera direction
    float3 cr = normalize(cross(up, cd)); // camera right
    float3 cu = normalize(cross(cd, cr)); // camera up

    return make_float3x3_by_columns(-1.0f * cr, cu,-1.0f *  cd);
}

float map(float3 p) {
	float input[3] = {p.x, p.y, p.z};
	float model_dist = model_frag->forward_light(input);
	return model_dist;
}

float rayMarch(float3 ro, float3 rd, float2 tNearAndFar ) {
	float hit;
	float pos = tNearAndFar.x;
	for(int i = 0; i < MAX_STEPS; i++) {
		float3 p = ro + pos * rd;
		hit = map(p);
		pos += hit;
		if(abs(hit) < EPS)
			break;
		if(pos > tNearAndFar.y) {
			break;
		}
	}
	return pos;
}


float3 getNormal(float3 p) {
	float2 e = float2(EPS * 5.0f, 0.0f);
	float3 normal = float3(map(p)) - float3(map(p - float3{e.x, e.y, e.y}), map(p - float3{e.y, e.x, e.y}), map(p - float3{e.y, e.y, e.x}));
	return normalize(normal);
}

float3 getLight(float3 p, float3 rd, float3 color) {
	// Light source (position)
	float3 lightSource = float3(20.0f, 40.0f, 30.0f);

	// Vector from point to light - L
	float3 L = normalize(lightSource - p);
	// Point p normal - N
	float3 N = getNormal(p);
	// Vector from p to camera - V
	float3 V = -1.0f * rd;
	// Reflected light - R
	float3 R = reflect(-1.0f * L, N);

	float3 diffuse = color * clamp(dot(L, N), 0.0f, 1.0f);
	float3 ambient = color * 0.05f;
	float3 fresnel = 0.10f * color * powf(1.0f + dot(rd, N), 3.0f);
	return diffuse + ambient + fresnel;
}

float4 fragment(float2 gl_FragCoord, float2 iResolution){
	float2 uv = ((gl_FragCoord + 0.5f) - 0.5f * iResolution) / iResolution.y;
	float3 up = float3(0.0f, 1.0f, 0.0f);
	float3 lookAtPoint = float3(-0.07, 0.16, 0.);
	float3 ro = float3(0.f, 0.5f, 3.f);

	float3x3 lookAt1 = myLookAt(ro, lookAtPoint, up);
	float3 rd = lookAt1 * normalize(float3(uv.x, uv.y, -CAMERA_FOV));

	float3 col = float3(0.0f, 0.0f, 0.0f);
	float2 tNearAndFar = RayBoxIntersection(ro, rd, float3(-1.f, -1.f, -1.f), float3(1.f, 1.f, 1.f));
	if(tNearAndFar.x < tNearAndFar.y && tNearAndFar.x > 0.0f){
		float dist = rayMarch(ro, rd, tNearAndFar);
		if(dist < tNearAndFar.y) {
			col = getLight(ro + rd * dist, rd, float3(0.6706, 0.7216, 0.8549));
		} else
			col = float3(0.0f, 0.0f, 0.0f);
	} 

	float4 FragColor = float4(powf(col.x, 1.0f / 2.2f), powf(col.y, 1.0f / 2.2f), powf(col.z, 1.0f / 2.2f), 1.0f);
	return FragColor;
}
