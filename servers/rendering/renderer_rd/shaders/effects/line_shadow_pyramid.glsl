#[compute]

#version 450

#VERSION_DEFINES

// Builds one level of the min-depth pyramid that line light shadows traverse. Stored
// depth is 1 - distance / far, so the nearest surface is the largest value.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#ifdef MODE_FROM_ATLAS
layout(set = 0, binding = 0) uniform sampler2D source;
#else
layout(r32f, set = 0, binding = 0) uniform restrict readonly image2D source;
#endif

layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D dest;

layout(push_constant, std430) uniform Params {
	ivec2 offset;
	ivec2 size;
}
params;

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pos, params.size))) {
		return;
	}
	pos += params.offset;

	float nearest = 0.0;
#ifdef MODE_FROM_ATLAS
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			nearest = max(nearest, texelFetch(source, pos * 4 + ivec2(x, y), 0).r);
		}
	}
#else
	for (int y = 0; y < 2; y++) {
		for (int x = 0; x < 2; x++) {
			nearest = max(nearest, imageLoad(source, pos * 2 + ivec2(x, y)).r);
		}
	}
#endif
	imageStore(dest, pos, vec4(nearest));
}
