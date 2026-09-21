#[compute]

#version 450

#VERSION_DEFINES

#extension GL_EXT_control_flow_attributes : require

// Line light visibility per pixel of the depth prepass, which MODE_FILTER smooths over
// neighbours on the same surface for the forward pass to read.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define MAX_VIEWS 2
#include "../scene_data_inc.glsl"

layout(set = 0, binding = 0) uniform sampler2D depth_buffer;
layout(rg16, set = 0, binding = 1) uniform restrict writeonly image2D output_visibility;
layout(set = 0, binding = 3) uniform sampler2D normal_roughness_buffer;
// MODE_FILTER: the visibility to filter. Otherwise: the light's unfiltered contact hits.
layout(set = 0, binding = 4) uniform sampler2D source_buffer;
layout(set = 0, binding = 11, std140) uniform SceneDataBlock {
	SceneData data;
}
scene_data_block;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	uint light_index;
	uint view;
	uint use_contact;
	int tap_step; // MODE_FILTER: tap spacing in pixels.
	float taa_frame_count;
	float pad;
}
params;

vec3 view_position(ivec2 pixel, float depth) {
	vec2 ndc = (vec2(pixel) + 0.5) / vec2(params.screen_size) * 2.0 - 1.0;
	vec4 p = scene_data_block.data.inv_projection_matrix_view[params.view] * vec4(ndc, depth, 1.0);
	return p.xyz / p.w;
}

vec3 scene_position(ivec2 pixel) {
	pixel = clamp(pixel, ivec2(0), params.screen_size - 1);
	return view_position(pixel, texelFetch(depth_buffer, pixel, 0).r);
}

float scene_z(ivec2 pixel) {
	return scene_position(pixel).z;
}

vec3 scene_normal(ivec2 pixel) {
	return normalize(texelFetch(normal_roughness_buffer, clamp(pixel, ivec2(0), params.screen_size - 1), 0).xyz * 2.0 - 1.0);
}

#ifdef MODE_FILTER

// Visibility differences beyond this are shadow edges, not dither.
#define VISIBILITY_SIGMA (10.0 / 64.0)

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.screen_size))) {
		return;
	}
	vec2 center = texelFetch(source_buffer, pixel, 0).rg;
	vec2 taps[9];
	vec2 lo = center;
	vec2 hi = center;
	for (int i = 0; i < 9; i++) {
		taps[i] = texelFetch(source_buffer, clamp(pixel + ivec2(i % 3 - 1, i / 3 - 1) * params.tap_step, ivec2(0), params.screen_size - 1), 0).rg;
		lo = min(lo, taps[i]);
		hi = max(hi, taps[i]);
	}
	if (all(equal(lo, hi))) {
		imageStore(output_visibility, pixel, vec4(center, 0.0, 0.0));
		return;
	}

	float z = scene_z(pixel);
	vec3 normal = scene_normal(pixel);
	// Depth change over one pixel in x plus one in y, one-sided so edges do not count.
	float slope = 0.0;
	for (int axis = 0; axis < 2; axis++) {
		ivec2 o = ivec2(axis == 0, axis == 1);
		slope += min(abs(scene_z(pixel + o) - z), abs(scene_z(pixel - o) - z));
	}
	float tolerance = (1.5 * slope + 0.002 * -z) * float(params.tap_step);

	// A 3x3 box, over which interleaved gradient noise averages out.
	vec2 sum = vec2(0.0);
	vec2 sum_w = vec2(0.0);
	for (int i = 0; i < 9; i++) {
		ivec2 o = ivec2(i % 3 - 1, i / 3 - 1);
		ivec2 q = clamp(pixel + o * params.tap_step, ivec2(0), params.screen_size - 1);
		if (abs(scene_z(q) - z) > tolerance || dot(scene_normal(q), normal) < 0.9) {
			continue;
		}
		vec2 w = exp(-abs(taps[i] - center) / VISIBILITY_SIGMA);
		sum += w * taps[i];
		sum_w += w;
	}
	imageStore(output_visibility, pixel, vec4(sum / sum_w, 0.0, 0.0));
}

#else

#include "../half_inc.glsl"
#include "../light_data_inc.glsl"

layout(set = 0, binding = 2, std430) restrict readonly buffer LineLights {
	LightData data[];
}
line_lights;

layout(set = 0, binding = 5) uniform texture2D shadow_atlas;
layout(set = 0, binding = 6) uniform texture2D line_shadow_pyramid;
layout(set = 0, binding = 7) uniform sampler SAMPLER_NEAREST_CLAMP;
layout(set = 0, binding = 8) uniform sampler SAMPLER_LINEAR_CLAMP;
layout(set = 0, binding = 9) uniform texture2D ltc_lut1;
layout(set = 0, binding = 10) uniform texture2D ltc_lut2;

#include "../area_lights_inc.glsl"
#include "../line_light_shadow_inc.glsl"

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.screen_size))) {
		return;
	}

	uint idx = params.light_index;
	half vis_diffuse = half(1.0);
	half vis_specular = half(1.0);
	float depth = texelFetch(depth_buffer, pixel, 0).r;
	vec3 segment = line_lights.data[idx].area_width;
	float length_sq = dot(segment, segment);
	vec3 vertex = view_position(pixel, depth);
	vec3 light_center = line_lights.data[idx].position - vertex;
	vec3 p1 = light_center - 0.5 * segment;
	vec3 p2 = light_center + 0.5 * segment;
	float min_radius = max(line_lights.data[idx].size, 0.0);
	float dist_to_segment = max(length(p1 + segment * clamp(dot(-p1, segment) / max(length_sq, 1e-12), 0.0, 1.0)), min_radius);

	if (depth > 0.0 && length_sq >= 1e-12 && dist_to_segment * line_lights.data[idx].inv_radius < 1.0) {
		vec3 eye_vec = -normalize(vertex - scene_data_block.data.eye_offset[params.view].xyz);
		vec4 normal_roughness = texelFetch(normal_roughness_buffer, pixel, 0);
		vec3 normal = normalize(normal_roughness.xyz * 2.0 - 1.0);
		// The 8-bit normal is ~0.1 degrees off, enough to move a glossy lobe; on flat
		// surfaces the normal from depth is exact.
		vec3 axes[2];
		for (int axis = 0; axis < 2; axis++) {
			ivec2 o = ivec2(axis == 0, axis == 1);
			vec3 forward = scene_position(pixel + o) - vertex;
			vec3 backward = vertex - scene_position(pixel - o);
			axes[axis] = abs(forward.z) < abs(backward.z) ? forward : backward;
		}
		vec3 flat_normal = cross(axes[0], axes[1]);
		if (dot(flat_normal, flat_normal) > 1e-30) { // Degenerate at the screen border.
			flat_normal = normalize(flat_normal);
			flat_normal = dot(flat_normal, normal) < 0.0 ? -flat_normal : flat_normal;
			if (dot(flat_normal, normal) > 0.99996) { // Within 0.5 degrees.
				normal = flat_normal;
			}
		}
		float roughness = normal_roughness.w > 0.5 ? 1.0 - normal_roughness.w : normal_roughness.w;
		roughness /= 127.0 / 255.0;

		vec2 ltc_fresnel;
		mat3 cos_xform = ltc_line_cos_xform(normal, eye_vec, ltc_matrix(normal, eye_vec, roughness, SAMPLER_LINEAR_CLAMP, ltc_lut1, ltc_lut2, ltc_fresnel));
		float ltc_min_radius = ltc_line_width_factor(cos_xform, p1, p2) * min_radius;

		LineShadowContext ctx = line_shadow_begin(idx, vertex, normal, params.taa_frame_count, vec2(pixel) + 0.5, scene_data_block.data.shadow_atlas_pixel_size);
		uvec4 contact = uvec4(0u);
		if (params.use_contact != 0u) {
			// A hit hides its whole quarter.
			uint bits = uint(texelFetch(source_buffer, pixel, 0).r * 255.0 + 0.5);
			contact = line_shadow_contact_cells(((bits & 1u) * 15u) | ((bits & 2u) * 120u) | ((bits & 4u) * 960u) | ((bits & 8u) * 7680u), ctx.jitter);
		}

		float seg_length = sqrt(length_sq);
		vec3 wt = segment / seg_length;
		float l1 = dot(p1, wt);
		vec3 po = p1 - l1 * wt;
		float d0 = length(po);
		float d = max(d0, max(min_radius, LINE_LIGHT_MIN_DISTANCE));
		line_shadow_visibility(idx, ctx, normal, eye_vec, roughness * roughness, po * (d / max(d0, 1e-9)), wt,
				d, l1, l1 + seg_length, l1 + 0.5 * seg_length, cos_xform, ltc_min_radius, contact, true,
				vis_diffuse, vis_specular);
	}

	imageStore(output_visibility, pixel, vec4(vis_diffuse, vis_specular, 0.0, 0.0));
}

#endif
