#[compute]

#version 450

#VERSION_DEFINES

#extension GL_EXT_control_flow_attributes : require

// Line light visibility per pixel of the depth prepass, which MODE_FILTER smooths over
// neighbours on the same surface and MODE_PENUMBRA widens by the light's radius, for
// the forward pass to read. Layers hold diffuse and specular visibility, then the hidden
// cells' mean blocker distance from the line over the receiver's times a, and a: 1 where
// there was a blocker, so blending keeps pixels without one out of the mean.

#ifdef MODE_WALK
// One thread per pixel MODE_CLASSIFY listed, so a warp holds 64 walkers and nothing else.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
#else
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
#endif

#define MAX_VIEWS 2
#include "../scene_data_inc.glsl"

layout(set = 0, binding = 0) uniform sampler2D depth_buffer;
// MODE_CLASSIFY leaves the reused answer, or the history to blend onto, for MODE_WALK.
layout(rgba16, set = 0, binding = 1) uniform restrict image2D output_visibility;
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
	uint frame; // Counts calls, for the refresh schedule.
	// Temporal reuse: current view space to last frame's clip space, and the row of
	// current view space to last frame's view-space z.
	mat4 reprojection;
	vec4 prev_view_z;
	uint temporal_frames; // Each tile walks once in this many frames; 0 is every frame.
	uint history_flags; // HISTORY_*.
	uint pad[2];
}
params;

// Last frame's visibility is this light's, from the same shadow maps' viewpoints.
#define HISTORY_VALID 1u
// Its shadow maps were redrawn within the last `temporal_frames` frames: what tiles
// walk replaces the history, so every tile catches up once in that window.
#define HISTORY_REDRAWN 2u
// Walks a pixel's history averages at most; the newest weighs at least 1 / this.
#define HISTORY_MAX_COUNT 16.0
// Pixels with fewer walks than this walk every frame, not only on their tile's turn.
#define HISTORY_MIN_COUNT 3.0

// A listed walker: its pixel, and whether only the specular lobe has to be redone.
#define WALK_SPECULAR_ONLY 0x80000000u
// Groups along x of the walk's dispatch; the rest of the list goes in further rows.
#define WALK_GROUPS_PER_ROW 256u

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

// The surface's own plane, from the nearer neighbour each way so edges do not bend
// it, with how much its depth changes over a pixel. Zero where it is degenerate.
vec3 surface_plane(ivec2 pixel, vec3 vertex, out float slope) {
	vec3 axes[2];
	slope = 0.0;
	for (int axis = 0; axis < 2; axis++) {
		ivec2 o = ivec2(axis == 0, axis == 1);
		vec3 forward = scene_position(pixel + o) - vertex;
		vec3 backward = vertex - scene_position(pixel - o);
		axes[axis] = abs(forward.z) < abs(backward.z) ? forward : backward;
		slope += min(abs(forward.z), abs(backward.z));
	}
	vec3 n = cross(axes[0], axes[1]);
	return dot(n, n) > 1e-30 ? normalize(n) : vec3(0.0);
}

// Two normals share one texel, 8 bits per octahedral axis: about a degree, far under
// the 25 the test allows.
vec2 oct_encode(vec3 n) {
	n /= abs(n.x) + abs(n.y) + abs(n.z);
	vec2 e = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
	return e * 0.5 + 0.5;
}

vec3 oct_decode(vec2 e) {
	e = e * 2.0 - 1.0;
	vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
	float t = max(-n.z, 0.0);
	n.xy += vec2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
	return normalize(n);
}

vec3 scene_normal(ivec2 pixel) {
	return normalize(texelFetch(normal_roughness_buffer, clamp(pixel, ivec2(0), params.screen_size - 1), 0).xyz * 2.0 - 1.0);
}

// Included once for every mode: the shader compiler inlines each file only once.
#include "../half_inc.glsl"
#include "../light_data_inc.glsl"
#include "../area_lights_inc.glsl"

#ifdef MODE_FILTER

// Visibility differences beyond this are shadow edges, not dither.
#define VISIBILITY_SIGMA (10.0 / 64.0)

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.screen_size))) {
		return;
	}
	vec4 center_texel = texelFetch(source_buffer, pixel, 0);
	vec2 center = center_texel.rg;
	vec2 taps[9];
	vec2 lo = center;
	vec2 hi = center;
	for (int i = 0; i < 9; i++) {
		taps[i] = texelFetch(source_buffer, clamp(pixel + ivec2(i % 3 - 1, i / 3 - 1) * params.tap_step, ivec2(0), params.screen_size - 1), 0).rg;
		lo = min(lo, taps[i]);
		hi = max(hi, taps[i]);
	}
	if (all(equal(lo, hi))) {
		imageStore(output_visibility, pixel, center_texel);
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
	imageStore(output_visibility, pixel, vec4(sum / sum_w, center_texel.ba));
}

#elif defined(MODE_RECORD)

layout(rg32f, set = 0, binding = 14) uniform restrict writeonly image2D current_z;
layout(rgba8, set = 0, binding = 15) uniform restrict writeonly image2D current_normal;

// What the next frame needs to know about this one's surfaces, out of the walk's way:
// view-space z with its change over a pixel, then the shaded normal and the surface's
// own plane, both in world space and octahedral.
void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.screen_size))) {
		return;
	}
	float depth = texelFetch(depth_buffer, pixel, 0).r;
	vec3 vertex = view_position(pixel, depth);
	float slope = 0.0;
	vec3 plane = depth > 0.0 ? surface_plane(pixel, vertex, slope) : vec3(0.0);
	vec3 view_normal = scene_normal(pixel);
	vec3 view_plane = dot(plane, plane) > 0.5 ? plane : view_normal;
	vec3 world_normal = vec3(dot(scene_data_block.data.inv_view_matrix[0].xyz, view_normal), dot(scene_data_block.data.inv_view_matrix[1].xyz, view_normal), dot(scene_data_block.data.inv_view_matrix[2].xyz, view_normal));
	vec3 world_plane = vec3(dot(scene_data_block.data.inv_view_matrix[0].xyz, view_plane), dot(scene_data_block.data.inv_view_matrix[1].xyz, view_plane), dot(scene_data_block.data.inv_view_matrix[2].xyz, view_plane));
	imageStore(current_z, pixel, vec4(depth > 0.0 ? vertex.z : 0.0, slope, 0.0, 0.0));
	imageStore(current_normal, pixel, vec4(oct_encode(world_normal), oct_encode(world_plane)));
}

#elif defined(MODE_CLASSIFY)

layout(set = 0, binding = 2, std430) restrict readonly buffer LineLights {
	LightData data[];
}
line_lights;

// Temporal reuse: last frame's unfiltered visibility, and what MODE_RECORD wrote about
// this frame's surfaces and the last one's.
layout(set = 0, binding = 12) uniform sampler2D history_buffer;
layout(set = 0, binding = 13) uniform sampler2D previous_z;
layout(set = 0, binding = 14) uniform sampler2D current_z;
layout(set = 0, binding = 15) uniform sampler2D current_normal;
layout(set = 0, binding = 16) uniform sampler2D previous_normal;
// Walks a pixel's history is worth, and the point on the tube its lobe looked at.
layout(rg32f, set = 0, binding = 17) uniform restrict writeonly image2D sample_count;
layout(set = 0, binding = 18) uniform sampler2D previous_count;
layout(set = 0, binding = 19, std430) restrict buffer WalkList {
	uint count;
	uint pixels[];
}
walk_list;
// Apart from the list, since only one access to a buffer per dispatch is barriered and
// the walk needs this one as dispatch arguments.
layout(set = 0, binding = 20, std430) restrict buffer WalkArgs {
	uint groups[3];
}
walk_args;

// Last frame's pixel `prev` showed the surface expected here.
bool _same_surface(ivec2 prev, float expected_z, float tolerance, vec3 world_normal, vec3 world_plane) {
	if (any(lessThan(prev, ivec2(0))) || any(greaterThanEqual(prev, params.screen_size))) {
		return false;
	}
	float prev_z = texelFetch(previous_z, prev, 0).r;
	vec4 prev_normal = texelFetch(previous_normal, prev, 0);
	// A normal tells apart surfaces meeting at a depth both match, such as a bevel and the
	// face it rounds. Either it or the surface's own plane agreeing is enough: a normal map
	// moves the first under the jitter, geometry finer than a pixel the second.
	bool same_shaded = dot(oct_decode(prev_normal.xy), world_normal) > 0.9;
	bool same_plane = dot(oct_decode(prev_normal.zw), world_plane) > 0.9;
	return prev_z < 0.0 && abs(prev_z - expected_z) <= tolerance && (same_shaded || same_plane);
}

// Walkers are listed per workgroup, so a tile's pixels stay together in the list.
shared uint tile_first;
shared uint tile_walkers;

// Sorts each pixel into reuse, a specular-only walk or a full walk, answers the reused
// ones and lists the rest for MODE_WALK. The walk diverges so badly that one walking
// pixel costs its neighbours a full walk, so it is worth handing it only walkers.
void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	bool inside = all(lessThan(pixel, params.screen_size));
	uint idx = params.light_index;
	float depth = inside ? texelFetch(depth_buffer, pixel, 0).r : 0.0;
	vec3 vertex = view_position(pixel, depth);
	vec3 segment = line_lights.data[idx].area_width;
	float length_sq = dot(segment, segment);
	vec3 p1 = line_lights.data[idx].position - vertex - 0.5 * segment;
	float min_radius = max(line_lights.data[idx].size, 0.0);
	float dist_to_segment = max(length(p1 + segment * clamp(dot(-p1, segment) / max(length_sq, 1e-12), 0.0, 1.0)), min_radius);

	// The sky and anything out of the light's reach is lit, and never enters the walk.
	bool walks = inside && depth > 0.0 && length_sq >= 1e-12 && dist_to_segment * line_lights.data[idx].inv_radius < 1.0;
	bool specular_only = false;
	if (inside && !walks) {
		imageStore(output_visibility, pixel, vec4(1.0, 1.0, 0.0, 0.0));
		if (params.temporal_frames > 0u) {
			imageStore(sample_count, pixel, vec4(0.0));
		}
	}

	vec4 history = vec4(1.0, 1.0, 0.0, 0.0);
	const bool lit = walks;
	if (walks && params.temporal_frames > 0u) {
		// Where the specular lobe looks along the tube. It follows the camera, so on a
		// glossy surface a history walked for another mirror point is stale even though
		// the surface itself matches.
		vec4 normal_roughness = texelFetch(normal_roughness_buffer, pixel, 0);
		vec3 normal = normalize(normal_roughness.xyz * 2.0 - 1.0);
		float roughness = normal_roughness.w > 0.5 ? 1.0 - normal_roughness.w : normal_roughness.w;
		roughness /= 127.0 / 255.0;
		float seg_length = sqrt(length_sq);
		vec3 wt = segment / seg_length;
		float l1 = dot(p1, wt);
		vec3 po = p1 - l1 * wt;
		float d0 = length(po);
		vec3 po_w = po * (max(d0, max(min_radius, LINE_LIGHT_MIN_DISTANCE)) / max(d0, 1e-9));
		vec3 eye_vec = -normalize(vertex - scene_data_block.data.eye_offset[params.view].xyz);
		vec3 refl = reflect(-eye_vec, normal);
		float a = dot(wt, refl);
		float lm = clamp(abs(a) < 0.999 ? a * dot(po_w, refl) / ((1.0 - a) * (1.0 + a)) : 0.0, l1, l1 + seg_length);
		// Half the lobe's footprint there: within it the lobe covers the same tube.
		float slack = roughness * roughness * length(po_w + wt * lm);
		float lm_frac = (lm - l1) / seg_length;

		float count = 0.0;
		float lm_history = lm_frac;
		vec4 surface = texelFetch(current_normal, pixel, 0);
		vec3 world_normal = oct_decode(surface.xy);
		vec3 world_plane = oct_decode(surface.zw);
		float slope = texelFetch(current_z, pixel, 0).g;
		if ((params.history_flags & HISTORY_VALID) != 0u) {
			vec4 clip = params.reprojection * vec4(vertex, 1.0);
			if (clip.w > 0.0) {
				// History is kept per pixel as if unjittered, so this frame's jitter is
				// added back: a still camera reads each pixel from itself, and repeated
				// reads do not blur.
				vec2 prev_pos = ((clip.xy / clip.w + scene_data_block.data.taa_jitter) * 0.5 + 0.5) * vec2(params.screen_size);
				float expected_z = dot(params.prev_view_z, vec4(vertex, 1.0));
				// Reprojection lands between pixels, so a sloped surface may differ by
				// its depth change over one each way. A different surface does not stay
				// within it. Next to the sky the change is unbounded, hence the clamp.
				float tolerance = 1.5 * min(slope, 0.05 * -vertex.z) + 0.002 * -expected_z + 0.02;

				// Bilinear: nearest would pin the shadow to the screen under sub-pixel
				// camera motion. Walks are weighted too, so an unused tap cannot make it young.
				vec2 base_pos = prev_pos - 0.5;
				ivec2 base = ivec2(floor(base_pos));
				vec2 f = base_pos - vec2(base);

				vec4 sum = vec4(0.0);
				float sum_w = 0.0;
				vec2 sum_count = vec2(0.0);
				for (int i = 0; i < 4; i++) {
					ivec2 prev = base + ivec2(i & 1, i >> 1);
					if (!_same_surface(prev, expected_z, tolerance, world_normal, world_plane)) {
						continue;
					}
					float w = ((i & 1) != 0 ? f.x : 1.0 - f.x) * ((i >> 1) != 0 ? f.y : 1.0 - f.y);
					sum += w * texelFetch(history_buffer, prev, 0);
					sum_w += w;
					sum_count += w * texelFetch(previous_count, prev, 0).rg;
				}
				bool found = sum_w > 0.05;
				if (!found) {
					// With jitter, a pixel on an edge sees the other surface every other
					// frame; its own surface is then usually one pixel further out.
					const ivec2 around[8] = ivec2[](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1), ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1));
					ivec2 center = ivec2(floor(prev_pos));
					for (int i = 0; i < 8 && !found; i++) {
						ivec2 prev = center + around[i];
						if (_same_surface(prev, expected_z, tolerance, world_normal, world_plane)) {
							sum = texelFetch(history_buffer, prev, 0);
							sum_count = texelFetch(previous_count, prev, 0).rg;
							sum_w = 1.0;
							found = true;
						}
					}
				}
				if (found) {
					history = sum / sum_w;
					count = floor(sum_count.r / sum_w + 0.01);
					lm_history = sum_count.g / sum_w;
				}
			}
		}

		if (count > 0.0) {
			uvec2 tile = gl_WorkGroupID.xy;
			uint tile_hash = (tile.x * 73856093u) ^ (tile.y * 19349663u);
			bool refresh = (tile_hash + params.frame) % params.temporal_frames == 0u;
			// While the maps are new, young pixels wait for their tile too: they start over
			// once it is past.
			if (!refresh && (count >= HISTORY_MIN_COUNT || (params.history_flags & HISTORY_REDRAWN) != 0u)) {
				// Only the lobe follows the camera, so only it has to go round again.
				specular_only = abs(lm_frac - lm_history) * seg_length > slack;
				walks = specular_only;
			}
		}
		// Staleness is measured from the mirror point the stored answer was walked at, so
		// a reuse keeps that one and cannot drift there a lobe's width at a time.
		imageStore(sample_count, pixel, vec4(count, walks ? lm_frac : lm_history, 0.0, 0.0));
	}
	if (lit) {
		// The reused answer, or what the walk blends onto. Written for a walker too, so a
		// pixel never keeps the last frame's, which the penumbra would feed back into itself.
		imageStore(output_visibility, pixel, history);
	}

	if (gl_LocalInvocationIndex == 0u) {
		tile_walkers = 0u;
	}
	barrier();
	uint slot = walks ? atomicAdd(tile_walkers, 1u) : 0u;
	barrier();
	if (gl_LocalInvocationIndex == 0u && tile_walkers > 0u) {
		tile_first = atomicAdd(walk_list.count, tile_walkers);
		// Rows, because a screen's worth of walkers is more groups than one dimension of a
		// dispatch is guaranteed to hold. The last row's spare groups return at once.
		atomicMax(walk_args.groups[1], (tile_first + tile_walkers + WALK_GROUPS_PER_ROW * 64u - 1u) / (WALK_GROUPS_PER_ROW * 64u));
	}
	barrier();
	if (walks) {
		walk_list.pixels[tile_first + slot] = uint(pixel.x) | (uint(pixel.y) << 16) | (specular_only ? WALK_SPECULAR_ONLY : 0u);
	}
}

#elif defined(MODE_PENUMBRA)


layout(set = 0, binding = 2, std430) restrict readonly buffer LineLights {
	LightData data[];
}
line_lights;

// A blocker k times as far from the receiver as from the line spreads the shadow of a
// line of radius r over r * k either way. Blockers on the line itself would spread it
// without bound; beyond this the shadow is too faint to matter.
#define PENUMBRA_MAX_K 8.0
#define PENUMBRA_SEARCH_TAPS 8
#define PENUMBRA_FILTER_TAPS 16

// Taps on the receiver's surface, laid out in its own plane so grazing views do not
// stretch them. Returns false for taps off screen, behind the camera or on another surface.
bool _penumbra_tap(vec3 vertex, vec3 normal, vec3 offset, out ivec2 q) {
	vec4 clip = scene_data_block.data.projection_matrix_view[params.view] * vec4(vertex + offset, 1.0);
	if (clip.w <= 1e-6) {
		return false;
	}
	q = ivec2(floor((clip.xy / clip.w * 0.5 + 0.5) * vec2(params.screen_size)));
	if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, params.screen_size))) {
		return false;
	}
	vec3 p = scene_position(q);
	return abs(dot(p - vertex, normal)) <= 0.01 * -vertex.z + 0.05 && dot(scene_normal(q), normal) > 0.9;
}

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.screen_size))) {
		return;
	}
	vec4 center = texelFetch(source_buffer, pixel, 0);
	float depth = texelFetch(depth_buffer, pixel, 0).r;
	float radius = line_lights.data[params.light_index].size;
	if (depth <= 0.0 || radius <= 0.0) {
		imageStore(output_visibility, pixel, center);
		return;
	}
	vec3 vertex = view_position(pixel, depth);
	vec3 normal = scene_normal(pixel);
	vec3 tangent = normalize(cross(normal, abs(normal.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
	vec3 bitangent = cross(normal, tangent);
	// Rotated per pixel and frame; temporal antialiasing averages the pattern.
	const vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
	float angle = 6.2831853 * fract(magic.z * fract(dot(vec2(pixel) + params.taa_frame_count * 5.588238, magic.xy)));

	// Blocker search over the widest penumbra that could reach here.
	float k_sum = 0.0;
	float k_count = 0.0;
	if (center.a > 0.01) {
		float f = center.b / center.a;
		k_sum = (1.0 - f) / max(f, 1e-3);
		k_count = 1.0;
	}
	float search = radius * PENUMBRA_MAX_K;
	for (int i = 0; i < PENUMBRA_SEARCH_TAPS; i++) {
		// Vogel disk.
		float t = sqrt((float(i) + 0.5) / float(PENUMBRA_SEARCH_TAPS));
		float a = angle + float(i) * 2.3999632;
		ivec2 q;
		if (_penumbra_tap(vertex, normal, search * t * (cos(a) * tangent + sin(a) * bitangent), q)) {
			vec2 blocker = texelFetch(source_buffer, q, 0).ba;
			if (blocker.y > 0.01) {
				float f = blocker.x / blocker.y;
				k_sum += (1.0 - f) / max(f, 1e-3);
				k_count += 1.0;
			}
		}
	}
	if (k_count == 0.0) {
		imageStore(output_visibility, pixel, center);
		return;
	}
	float spread = radius * min(k_sum / k_count, PENUMBRA_MAX_K);
	// Under a pixel there is nothing to spread. Orthographic pixels do not grow with depth.
	mat4 projection = scene_data_block.data.projection_matrix_view[params.view];
	float pixel_size = (projection[3][3] != 0.0 ? 1.0 : -vertex.z) * 2.0 / (abs(projection[1][1]) * float(params.screen_size.y));
	if (spread < pixel_size) {
		imageStore(output_visibility, pixel, center);
		return;
	}

	vec2 sum = center.rg;
	float sum_w = 1.0;
	for (int i = 0; i < PENUMBRA_FILTER_TAPS; i++) {
		float t = sqrt((float(i) + 0.5) / float(PENUMBRA_FILTER_TAPS));
		float a = angle + float(i) * 2.3999632;
		ivec2 q;
		if (_penumbra_tap(vertex, normal, spread * t * (cos(a) * tangent + sin(a) * bitangent), q)) {
			sum += texelFetch(source_buffer, q, 0).rg;
			sum_w += 1.0;
		}
	}
	imageStore(output_visibility, pixel, vec4(sum / sum_w, center.ba));
}

#else


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
layout(rg32f, set = 0, binding = 17) uniform restrict image2D sample_count;
layout(set = 0, binding = 19, std430) restrict readonly buffer WalkList {
	uint count;
	uint pixels[];
}
walk_list;

#define LINE_SHADOW_BLOCKER
#include "../line_light_shadow_inc.glsl"

void main() {
	uint id = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 64u;
	if (id >= walk_list.count) {
		return;
	}
	uint entry = walk_list.pixels[id];
	ivec2 pixel = ivec2(int(entry & 0xFFFFu), int((entry >> 16) & 0x7FFFu));
	// Only the lobe moved: the diffuse answer and the blocker distance stand as they are.
	bool specular_only = (entry & WALK_SPECULAR_ONLY) != 0u;

	uint idx = params.light_index;
	line_shadow_blocker = vec2(0.0);
	float depth = texelFetch(depth_buffer, pixel, 0).r;
	vec3 segment = line_lights.data[idx].area_width;
	float length_sq = dot(segment, segment);
	vec3 vertex = view_position(pixel, depth);

	// What MODE_CLASSIFY left: the history to average into, and how many walks it stands for.
	vec4 history = vec4(1.0, 1.0, 0.0, 0.0);
	vec2 stored = vec2(0.0);
	if (params.temporal_frames > 0u) {
		history = imageLoad(output_visibility, pixel);
		stored = imageLoad(sample_count, pixel).rg;
	}
	float count = stored.r;
	bool has_history = count > 0.0;
	half vis_diffuse = half(1.0);
	half vis_specular = specular_only ? half(history.g) : half(1.0);

	vec3 light_center = line_lights.data[idx].position - vertex;
	vec3 p1 = light_center - 0.5 * segment;
	vec3 p2 = light_center + 0.5 * segment;
	float min_radius = max(line_lights.data[idx].size, 0.0);

	{
		vec3 eye_vec = -normalize(vertex - scene_data_block.data.eye_offset[params.view].xyz);
		vec4 normal_roughness = texelFetch(normal_roughness_buffer, pixel, 0);
		vec3 normal = normalize(normal_roughness.xyz * 2.0 - 1.0);
		// The 8-bit normal is ~0.1 degrees off, enough to move a glossy lobe; on flat
		// surfaces the plane's normal is exact.
		float walk_slope;
		vec3 walk_plane = surface_plane(pixel, vertex, walk_slope);
		walk_plane = dot(walk_plane, normal) < 0.0 ? -walk_plane : walk_plane;
		if (dot(walk_plane, normal) > 0.99996) { // Within 0.5 degrees.
			normal = walk_plane;
		}
		float roughness = normal_roughness.w > 0.5 ? 1.0 - normal_roughness.w : normal_roughness.w;
		roughness /= 127.0 / 255.0;

		vec2 ltc_fresnel;
		mat3 cos_xform = ltc_line_cos_xform(normal, eye_vec, ltc_matrix(normal, eye_vec, roughness, SAMPLER_LINEAR_CLAMP, ltc_lut1, ltc_lut2, ltc_fresnel));
		float ltc_min_radius = ltc_line_width_factor(cos_xform, p1, p2) * min_radius;

		// Refreshing tiles walk in step with the TAA phase, so it would give each the same
		// dither every time; the pass's own frame count does not repeat with them.
		float noise_frame = params.temporal_frames > 0u ? float(params.frame % 1024u) : params.taa_frame_count;
		LineShadowContext ctx = line_shadow_begin(idx, vertex, normal, noise_frame, vec2(pixel) + 0.5, scene_data_block.data.shadow_atlas_pixel_size);
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
				d, l1, l1 + seg_length, l1 + 0.5 * seg_length, cos_xform, ltc_min_radius, contact, !specular_only,
				vis_diffuse, vis_specular);
	}

	vec4 visibility = specular_only ? vec4(history.r, vis_specular, history.ba)
									: vec4(vis_diffuse, vis_specular, line_shadow_blocker.y > 0.0 ? vec2(line_shadow_blocker.x / line_shadow_blocker.y, 1.0) : vec2(0.0));
	if (params.temporal_frames > 0u) {
		// A specular-only walk leaves the diffuse average and its age alone.
		if (!specular_only) {
			if (has_history && (params.history_flags & HISTORY_REDRAWN) != 0u) {
				count = 0.0; // Older walks saw other maps.
			} else if (has_history) {
				visibility = mix(history, visibility, 1.0 / (min(count, HISTORY_MAX_COUNT - 1.0) + 1.0));
			}
			count = min(count + 1.0, HISTORY_MAX_COUNT);
		}
		imageStore(sample_count, pixel, vec4(count, stored.g, 0.0, 0.0));
	}
	imageStore(output_visibility, pixel, visibility);
}

#endif
