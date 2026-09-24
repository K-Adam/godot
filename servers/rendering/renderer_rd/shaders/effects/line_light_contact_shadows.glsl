#[compute]

#version 450

#VERSION_DEFINES

// Line light contact shadows: each pixel marches a short ray towards each quarter of
// its cells and stores the hits as bits; MODE_FILTER averages them over neighbours on
// the same surface. The forward pass adds them to the line light's shadow mask.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "../light_data_inc.glsl"
#include "../line_light_cells_inc.glsl"

layout(set = 0, binding = 0) uniform sampler2D depth_buffer;
layout(set = 0, binding = 3) uniform sampler2D normal_roughness_buffer;
#ifdef MODE_FILTER
layout(set = 0, binding = 4) uniform sampler2D hits_buffer;
layout(r16, set = 0, binding = 1) uniform restrict writeonly image2D output_hits;
#else
layout(r8, set = 0, binding = 1) uniform restrict writeonly image2D output_hits;
#endif

layout(set = 0, binding = 2, std430) restrict readonly buffer LineLights {
	LightData data[];
}
line_lights;

layout(push_constant, std430) uniform Params {
	mat4 projection;
	ivec2 screen_size;
	uint light_index;
	uint steps;
	float thickness;
	float max_pixels;
	float taa_frame_count;
	float shadow_atlas_size;
}
params;

#define STRATA 4u
// Rays reach this many shadow map texels: the shadow map resolves anything further.
#define REACH_TEXELS 4.0

// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
float interleaved_gradient_noise(vec2 pos) {
	const vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
	return fract(magic.z * fract(dot(pos, magic.xy)));
}

vec3 view_position(vec2 uv, float depth, mat4 inv_projection) {
	vec4 p = inv_projection * vec4(uv * 2.0 - 1.0, depth, 1.0);
	return p.xyz / p.w;
}

// View depth of the surface seen at `pixel`.
float scene_z(ivec2 pixel, mat4 inv_projection) {
	pixel = clamp(pixel, ivec2(0), params.screen_size - 1);
	float depth = texelFetch(depth_buffer, pixel, 0).r;
	return view_position((vec2(pixel) + 0.5) / vec2(params.screen_size), depth, inv_projection).z;
}

vec3 scene_normal(ivec2 pixel) {
	return normalize(texelFetch(normal_roughness_buffer, clamp(pixel, ivec2(0), params.screen_size - 1), 0).xyz * 2.0 - 1.0);
}

// View depth change over a pixel in x plus one in y, one-sided so edges do not count.
float depth_slope(ivec2 pixel, float z, mat4 inv_projection) {
	float slope = 0.0;
	for (int axis = 0; axis < 2; axis++) {
		ivec2 o = ivec2(axis == 0, axis == 1);
		slope += min(abs(scene_z(pixel + o, inv_projection) - z), abs(scene_z(pixel - o, inv_projection) - z));
	}
	return slope;
}

#ifdef MODE_FILTER

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.screen_size))) {
		return;
	}
	uint bits[9];
	uint any_bits = 0u;
	for (int i = 0; i < 9; i++) {
		ivec2 q = clamp(pixel + ivec2(i % 3 - 1, i / 3 - 1), ivec2(0), params.screen_size - 1);
		bits[i] = uint(texelFetch(hits_buffer, q, 0).r * 255.0 + 0.5);
		any_bits |= bits[i];
	}
	if (any_bits == 0u) {
		imageStore(output_hits, pixel, vec4(0.0));
		return;
	}

	mat4 inv_projection = inverse(params.projection);
	float z = scene_z(pixel, inv_projection);
	vec3 normal = scene_normal(pixel);
	float tolerance = 1.5 * depth_slope(pixel, z, inv_projection) + 0.002 * -z;

	// A 3x3 tent over neighbours on the same surface, so contact edges stay sharp.
	vec4 sum = vec4(0.0);
	float total = 0.0;
	for (int i = 0; i < 9; i++) {
		ivec2 o = ivec2(i % 3 - 1, i / 3 - 1);
		ivec2 q = clamp(pixel + o, ivec2(0), params.screen_size - 1);
		if (abs(scene_z(q, inv_projection) - z) > tolerance || dot(scene_normal(q), normal) < 0.9) {
			continue;
		}
		float w = float((2 - abs(o.x)) * (2 - abs(o.y)));
		sum += w * vec4(notEqual(uvec4(bits[i]) & uvec4(1u, 2u, 4u, 8u), uvec4(0u)));
		total += w;
	}
	uvec4 f = uvec4(round(sum / max(total, 1.0) * 15.0));
	uint fractions = f.x | (f.y << 4u) | (f.z << 8u) | (f.w << 12u);
	imageStore(output_hits, pixel, vec4((float(fractions) + 0.25) / 65535.0));
}

#else

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.screen_size))) {
		return;
	}

	uint hits = 0u;
	float depth = texelFetch(depth_buffer, pixel, 0).r;
	mat4 inv_projection = inverse(params.projection);
	vec2 screen = vec2(params.screen_size);
	vec3 vertex = view_position((vec2(pixel) + 0.5) / screen, depth, inv_projection);

	LightData light = line_lights.data[params.light_index];
	vec3 segment = light.area_width;
	float seg_length = length(segment);
	vec3 wt = segment / max(seg_length, 1e-6);
	vec3 to_center = light.position - vertex;
	float l_center = dot(to_center, wt);
	float d = length(to_center - wt * l_center);
	float closest = length(vec2(d, max(abs(l_center) - 0.5 * seg_length, 0.0)));

	if (depth > 0.0 && seg_length > 1e-6 && closest * light.inv_radius < 1.0) {
		// The receiver's own cells, as line_shadow_visibility() builds them.
		d = max(d, max(light.size, 0.001));
		LineShadowCells cells = line_shadow_cells(-l_center, d, 0.5 * seg_length, 0.0);
		vec3 normal = scene_normal(pixel);
		float frame = params.taa_frame_count;
		float step_jitter = interleaved_gradient_noise(vec2(pixel) + frame * 5.588238);
		float stratum_jitter = interleaved_gradient_noise(vec2(pixel.yx) + 29.0 + frame * 3.371);
		// Samples snap to pixels, so a surface can seem that far in front of a ray leaving it.
		float tolerance = depth_slope(pixel, vertex.z, inv_projection) + 0.001 * -vertex.z;
		// A texel of the polar shadow map spans pi / size radians.
		float reach = REACH_TEXELS * closest * 3.14159265 / max(light.atlas_rect.z * params.shadow_atlas_size, 1.0);
		float max_pixels = params.max_pixels;
		if (light.area_height.x != 0.0) {
			// The simplified map, seen from the segment's middle, loses blockers hidden behind
			// higher ones; these rays follow the true rays, so they reach further.
			reach = 0.1 * closest;
			max_pixels *= 2.0;
		}

		for (uint k = 0u; k < STRATA; k++) {
			float u = line_shadow_cell_u(cells, (float(k) + stratum_jitter) * float(LINE_SHADOW_CELLS / STRATA));
			vec3 ray = light.position + wt * u - vertex;
			float ray_length = length(ray);
			vec3 dir = ray / max(ray_length, 1e-6);
			if (ray_length < 1e-4 || dot(dir, normal) <= 0.0) {
				continue; // Below the horizon: the receiver hides it itself.
			}

			// Stop short of the light (and emitter meshes) and of what the shadow map resolves.
			vec4 a = params.projection * vec4(vertex, 1.0);
			float probe = 0.01 * min(ray_length, -vertex.z);
			vec4 b = params.projection * vec4(vertex + dir * probe, 1.0);
			if (b.w <= 0.0) {
				continue;
			}
			float pixels_per_unit = length((b.xy / b.w - a.xy / a.w) * 0.5 * screen) / probe;
			float march = min(min(max_pixels / max(pixels_per_unit, 1e-6), 0.8 * ray_length), reach);

			float pixels = march * pixels_per_unit;
			if (pixels < 1.0) {
				continue;
			}
			uint steps = min(params.steps, uint(ceil(pixels)));
			for (uint i = 0u; i < steps; i++) {
				// Squared spacing keeps steps near the receiver short, where contact is.
				float t = (float(i) + step_jitter) / float(steps);
				t = march * max(t * t, 1.0 / pixels);
				vec3 p = vertex + dir * t;
				vec4 clip = params.projection * vec4(p, 1.0);
				if (clip.w <= 0.0) {
					break;
				}
				vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
				if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) {
					break;
				}
				float z = scene_z(ivec2(uv * screen), inv_projection);
				// In front of the ray, but no further than an assumed thickness.
				float in_front = z - p.z;
				if (in_front > tolerance && in_front < params.thickness * -p.z) {
					hits |= 1u << k;
					break;
				}
			}
		}
	}

	// Offset so that either rounding a driver may use stores `hits`.
	imageStore(output_hits, pixel, vec4((float(hits) + 0.25) / 255.0));
}

#endif
