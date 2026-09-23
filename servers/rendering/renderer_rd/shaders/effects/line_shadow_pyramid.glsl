#[compute]

#version 450

#VERSION_DEFINES

// Builds one level of the min-depth pyramid that line light shadows traverse. Stored
// depth is 1 - distance / far, so the nearest surface is the largest value.
// MODE_ROWS instead builds per-row block data (see line_light_cells_inc.glsl).

#define M_PI 3.14159265359

#include "../line_light_cells_inc.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#if defined(MODE_FROM_ATLAS) || defined(MODE_ROWS)
layout(set = 0, binding = 0) uniform sampler2D source;
#else
layout(r32f, set = 0, binding = 0) uniform restrict readonly image2D source;
#endif

layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D dest;

#ifdef MODE_ROWS
layout(r32f, set = 2, binding = 0) uniform restrict image2D level_texels;
#endif

layout(push_constant, std430) uniform Params {
	ivec2 offset;
	ivec2 size;
	int level;
	int atlas_size;
	int pad[2];
}
params;

#ifdef MODE_ROWS

// Whether texels a to b are one joined surface, straight to within
// LINE_SHADOW_STRAIGHT_TEXELS. Also returns the most bent texel and the unjoined pairs.
bool _straight(ivec2 corner, int a, int b, float a0, float delta, out int bend, out float worst, out int gap, out int gaps) {
	float sa = texelFetch(source, corner + ivec2(a, 0), 0).r;
	float sb = texelFetch(source, corner + ivec2(b, 0), 0).r;
	vec2 pa = (1.0 - sa) * vec2(sin(a0 + float(a) * delta), cos(a0 + float(a) * delta));
	vec2 chord = (1.0 - sb) * vec2(sin(a0 + float(b) * delta), cos(a0 + float(b) * delta)) - pa;
	chord /= max(length(chord), 1e-9);
	worst = 0.0;
	bend = a;
	gap = a;
	gaps = 0;
	float prev = sa;
	for (int x = a + 1; x <= b; x++) {
		float s = texelFetch(source, corner + ivec2(x, 0), 0).r;
		vec2 p = (1.0 - s) * vec2(sin(a0 + float(x) * delta), cos(a0 + float(x) * delta)) - pa;
		float off = abs(p.x * chord.y - p.y * chord.x) / max((1.0 - s) * delta, 1e-9);
		if (!line_shadow_joined(prev, s, LINE_SHADOW_CONTINUITY * delta)) {
			gaps++;
			gap = x - 1;
		}
		if (off > worst) {
			worst = off;
			bend = x;
		}
		prev = s;
	}
	return sa > 0.0 && gaps == 0 && worst <= LINE_SHADOW_STRAIGHT_TEXELS;
}

void main() {
	ivec2 id = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(id, params.size))) {
		return;
	}
	int slot = params.size.y;
	int block = 4 << params.level;
	ivec2 corner = params.offset + ivec2(id.x * block, id.y);
	float delta = M_PI / float(slot);
	float a0 = (float(id.x * block) + 0.5) * delta;

	// One straight piece, or two split after texel k at the gap or the bend.
	int bend;
	float worst;
	int gap;
	int gaps;
	uint pieces = 0u;
	if (_straight(corner, 0, block - 1, a0, delta, bend, worst, gap, gaps)) {
		pieces = 1u;
	} else {
		int k = gaps == 1 ? gap : (gaps == 0 ? bend : -1);
		int unused_bend;
		int unused_gap;
		int unused_gaps;
		float worst_b;
		if (k >= 0 && k < block - 1 && _straight(corner, 0, k, a0, delta, unused_bend, worst, unused_gap, unused_gaps) && _straight(corner, k + 1, block - 1, a0, delta, unused_bend, worst_b, unused_gap, unused_gaps)) {
			pieces = uint(k + 2);
			worst = max(worst, worst_b);
		}
	}
	imageStore(dest, line_shadow_row_texel(params.offset.x, corner.y, slot, params.level, id.x, params.atlas_size), vec4(pieces > 0u ? float((pieces << 4u) | uint(ceil(worst * 4.0))) : 0.0));
	if (pieces > 0u) {
		// Every row of the texel that sets it writes the same value.
		ivec2 texel = corner >> (params.level + 2);
		imageStore(level_texels, texel, vec4(-abs(imageLoad(level_texels, texel).r)));
	}
}

#else

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

#endif
