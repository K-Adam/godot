#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define MAX_CASCADES 8

layout(set = 0, binding = 1) uniform texture3D sdf_cascades[MAX_CASCADES];
layout(set = 0, binding = 2) uniform sampler linear_sampler;
layout(set = 0, binding = 3) uniform sampler linear_sampler_with_mipmaps;

layout(set = 0, binding = 4, std430) restrict readonly buffer DispatchData {
	uint x;
	uint y;
	uint z;
	uint total_count;
}
dispatch_data;

struct ProcessVoxel {
	uint position; // xyz 7 bit packed, extra 11 bits for neighbors.
	uint albedo; // rgb bits 0-15 albedo, bits 16-21 are normal bits (set if geometry exists toward that side), extra 11 bits for neighbors.
	uint light; // rgbe8985 encoded total saved light, extra 2 bits for neighbors.
	uint light_aniso; // 55555 light anisotropy, extra 2 bits for neighbors.
	//total neighbors: 26
};

#ifdef MODE_PROCESS_STATIC
layout(set = 0, binding = 5, std430) restrict buffer ProcessVoxels {
#else
layout(set = 0, binding = 5, std430) restrict buffer readonly ProcessVoxels {
#endif
	ProcessVoxel data[];
}
process_voxels;

layout(r32ui, set = 0, binding = 6) uniform restrict uimage3D dst_light;
layout(rgba8, set = 0, binding = 7) uniform restrict image3D dst_aniso0;
layout(rg8, set = 0, binding = 8) uniform restrict image3D dst_aniso1;

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 probe_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 9, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

#define LIGHT_TYPE_DIRECTIONAL 0
#define LIGHT_TYPE_OMNI 1
#define LIGHT_TYPE_SPOT 2
#define LIGHT_TYPE_AREA 3
#define LIGHT_TYPE_LINE 4

// Line lights are shadowed per sub-segment, one ray each.
#define LINE_LIGHT_GI_SAMPLES 4

#include "../area_lights_inc.glsl"

struct Light {
	vec3 color;
	float energy;

	vec3 direction;
	bool has_shadow;

	vec3 position;
	float attenuation;

	uint type;
	float cos_spot_angle;
	float inv_spot_attenuation;
	float radius;

	vec4 area_width;
	vec4 area_height;
	vec4 area_projector_rect;
};

layout(set = 0, binding = 10, std140) buffer restrict readonly Lights {
	Light data[];
}
lights;

layout(set = 0, binding = 11) uniform texture2DArray lightprobe_texture;
layout(set = 0, binding = 12) uniform texture3D occlusion_texture;

layout(set = 1, binding = 0) uniform texture2D area_light_atlas;

layout(push_constant, std430) uniform Params {
	vec3 grid_size;
	uint max_cascades;

	uint cascade;
	uint light_count;
	uint process_offset;
	uint process_increment;

	int probe_axis_size;
	float bounce_feedback;
	float y_mult;
	bool use_occlusion;
}
params;

vec2 octahedron_wrap(vec2 v) {
	vec2 signVal;
	signVal.x = v.x >= 0.0 ? 1.0 : -1.0;
	signVal.y = v.y >= 0.0 ? 1.0 : -1.0;
	return (1.0 - abs(v.yx)) * signVal;
}

vec2 octahedron_encode(vec3 n) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = n.z >= 0.0 ? n.xy : octahedron_wrap(n.xy);
	n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}

float get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

// SDFGI squashes Y by `y_mult`; lighting uses true distances, the ray march squashed ones.
vec3 sdfgi_unsquash(vec3 v) {
	return v * vec3(1.0, 1.0 / params.y_mult, 1.0);
}

// `light_vec` and `light_distance` are returned in SDF space, for the ray march.
void compute_area_light(uint index, vec3 position, out float attenuation, out vec3 light_vec, out float light_distance, out vec3 texture_color) {
	vec3 area_width = lights.data[index].area_width.xyz;
	vec3 area_height = lights.data[index].area_height.xyz;
	vec3 area_direction = lights.data[index].direction;
	float a_len = length(area_width);
	float b_len = length(area_height);
	vec3 area_width_norm = normalize(area_width);
	vec3 area_height_norm = normalize(area_height);
	float a_half_len = a_len / 2.0;
	float b_half_len = b_len / 2.0;
	vec3 light_center = sdfgi_unsquash(lights.data[index].position);
	position = sdfgi_unsquash(position);
	vec3 light_to_vert = position - light_center;
	vec3 pos_local_to_light = vec3(dot(light_to_vert, area_width_norm), dot(light_to_vert, area_height_norm), dot(light_to_vert, -area_direction)); // position in LIGHT SPACE
	vec3 closest_point_local_to_light = vec3(clamp(pos_local_to_light.x, -a_half_len, a_half_len), clamp(pos_local_to_light.y, -b_half_len, b_half_len), 0.0); // LIGHT SPACE
	float inv_center_range = lights.data[index].inv_spot_attenuation;
	vec3 closest_point_on_light = light_center + closest_point_local_to_light.x * area_width_norm + closest_point_local_to_light.y * area_height_norm; // VIEW SPACE
	vec3 light_rel_vec = closest_point_on_light - position;
	float true_distance = length(light_rel_vec);
	vec3 march_vec = light_rel_vec * vec3(1.0, params.y_mult, 1.0);
	light_distance = length(march_vec);
	light_vec = march_vec / light_distance;
	float EPSILON = 1e-4f;
	if (light_distance < EPSILON) {
		light_vec = area_direction;
	}
	float max_mipmap = lights.data[index].cos_spot_angle;

	if (true_distance * inv_center_range >= 1.0) { // position in range
		attenuation = 0.0;
		return;
	}

	vec3 h_area_width = area_width / 2.0;
	vec3 h_area_height = area_height / 2.0;
	vec3 light_points[4];
	light_points[0] = light_center - h_area_width - h_area_height - position;
	light_points[1] = light_center + h_area_width - h_area_height - position;
	light_points[2] = light_center + h_area_width + h_area_height - position;
	light_points[3] = light_center - h_area_width + h_area_height - position;

	attenuation = get_omni_attenuation(true_distance, 1.0 / lights.data[index].radius, lights.data[index].attenuation - 2.0);
	float ltc_diffuse = 0.0;
	vec3 normal = true_distance < EPSILON ? area_direction : light_rel_vec / true_distance;
	ltc_evaluate_diff(normal, light_points, lights.data[index].area_projector_rect, max_mipmap, area_light_atlas, linear_sampler_with_mipmaps, ltc_diffuse, texture_color);
	attenuation *= ltc_diffuse;
}

// Marches the SDF cascades from `position` towards the light; false on a hit.
bool sdf_light_visible(vec3 position, vec3 direction, float light_distance) {
	vec3 pos_to_uvw = 1.0 / params.grid_size;
	bool hit = false;

	vec3 ray_pos = position;
	vec3 ray_dir = direction;
	vec3 inv_dir = 1.0 / ray_dir;

	//this is how to properly bias outgoing rays
	float cell_size = 1.0 / cascades.data[params.cascade].to_cell;
	ray_pos += sign(direction) * cell_size * 0.48; // go almost to the box edge but remain inside
	ray_pos += ray_dir * 0.4 * cell_size; //apply a small bias from there

	for (uint j = params.cascade; j < params.max_cascades; j++) {
		//convert to local bounds
		vec3 pos = ray_pos - cascades.data[j].offset;
		pos *= cascades.data[j].to_cell;
		float local_distance = light_distance * cascades.data[j].to_cell;

		if (any(lessThan(pos, vec3(0.0))) || any(greaterThanEqual(pos, params.grid_size))) {
			continue; //already past bounds for this cascade, goto next
		}

		//find maximum advance distance (until reaching bounds)
		vec3 t0 = -pos * inv_dir;
		vec3 t1 = (params.grid_size - pos) * inv_dir;
		vec3 tmax = max(t0, t1);
		float max_advance = min(tmax.x, min(tmax.y, tmax.z));

		max_advance = min(local_distance, max_advance);

		float advance = 0.0;

		while (advance < max_advance) {
			//read how much to advance from SDF
			vec3 uvw = (pos + ray_dir * advance) * pos_to_uvw;

			float distance = texture(sampler3D(sdf_cascades[j], linear_sampler), uvw).r * 255.0 - 1.0;
			if (distance < 0.001) {
				//consider hit
				hit = true;
				break;
			}

			advance += distance;
		}

		if (hit) {
			break;
		}

		if (advance >= local_distance) {
			break; //past light distance, abandon search
		}
		//change ray origin to collision with bounds
		pos += ray_dir * max_advance;
		pos /= cascades.data[j].to_cell;
		pos += cascades.data[j].offset;
		light_distance -= max_advance / cascades.data[j].to_cell;
		ray_pos = pos;
	}

	return !hit;
}

void main() {
	uint voxel_index = uint(gl_GlobalInvocationID.x);

	//used for skipping voxels every N frames
	if (params.process_increment > 1) {
		voxel_index *= params.process_increment;
		voxel_index += params.process_offset;
	}

	if (voxel_index >= dispatch_data.total_count) {
		return;
	}

	uint voxel_position = process_voxels.data[voxel_index].position;

	//keep for storing to texture
	ivec3 positioni = ivec3((uvec3(voxel_position, voxel_position, voxel_position) >> uvec3(0, 7, 14)) & uvec3(0x7F));

	vec3 position = vec3(positioni) + vec3(0.5);
	position /= cascades.data[params.cascade].to_cell;
	position += cascades.data[params.cascade].offset;

	uint voxel_albedo = process_voxels.data[voxel_index].albedo;

	vec3 albedo = vec3(uvec3(voxel_albedo >> 10, voxel_albedo >> 5, voxel_albedo) & uvec3(0x1F)) / float(0x1F);
	vec3 light_accum[6] = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
	uint valid_aniso = (voxel_albedo >> 15) & 0x3F;

	const vec3 aniso_dir[6] = vec3[](
			vec3(1, 0, 0),
			vec3(0, 1, 0),
			vec3(0, 0, 1),
			vec3(-1, 0, 0),
			vec3(0, -1, 0),
			vec3(0, 0, -1));

	// Add indirect light first, in order to save computation resources
#ifdef MODE_PROCESS_DYNAMIC
	if (params.bounce_feedback > 0.001) {
		vec3 feedback = (params.bounce_feedback < 1.0) ? (albedo * params.bounce_feedback) : mix(albedo, vec3(1.0), params.bounce_feedback - 1.0);
		vec3 pos = (vec3(positioni) + vec3(0.5)) * float(params.probe_axis_size - 1) / params.grid_size;
		ivec3 probe_base_pos = ivec3(pos);

		float weight_accum[6] = float[](0, 0, 0, 0, 0, 0);

		ivec3 tex_pos = ivec3(probe_base_pos.xy, int(params.cascade));
		tex_pos.x += probe_base_pos.z * int(params.probe_axis_size);

		tex_pos.xy = tex_pos.xy * (OCT_SIZE + 2) + ivec2(1);

		vec3 base_tex_posf = vec3(tex_pos);
		vec2 tex_pixel_size = 1.0 / vec2(ivec2((OCT_SIZE + 2) * params.probe_axis_size * params.probe_axis_size, (OCT_SIZE + 2) * params.probe_axis_size));
		vec3 probe_uv_offset = vec3(ivec3(OCT_SIZE + 2, OCT_SIZE + 2, (OCT_SIZE + 2) * params.probe_axis_size)) * tex_pixel_size.xyx;

		for (uint j = 0; j < 8; j++) {
			ivec3 offset = (ivec3(j) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1);
			ivec3 probe_posi = probe_base_pos;
			probe_posi += offset;

			// Compute weight

			vec3 probe_pos = vec3(probe_posi);
			vec3 probe_to_pos = pos - probe_pos;
			vec3 probe_dir = normalize(-probe_to_pos);

			// Compute lightprobe texture position

			vec3 trilinear = vec3(1.0) - abs(probe_to_pos);

			for (uint k = 0; k < 6; k++) {
				if (bool(valid_aniso & (1 << k))) {
					vec3 n = aniso_dir[k];
					float weight = trilinear.x * trilinear.y * trilinear.z * max(0, dot(n, probe_dir));

					if (weight > 0.0 && params.use_occlusion) {
						ivec3 occ_indexv = abs((cascades.data[params.cascade].probe_world_offset + probe_posi) & ivec3(1, 1, 1)) * ivec3(1, 2, 4);
						vec4 occ_mask = mix(vec4(0.0), vec4(1.0), equal(ivec4(occ_indexv.x | occ_indexv.y), ivec4(0, 1, 2, 3)));

						vec3 occ_pos = (vec3(positioni) + aniso_dir[k] + vec3(0.5)) / params.grid_size;
						occ_pos.z += float(params.cascade);
						if (occ_indexv.z != 0) { //z bit is on, means index is >=4, so make it switch to the other half of textures
							occ_pos.x += 1.0;
						}
						occ_pos *= vec3(0.5, 1.0, 1.0 / float(params.max_cascades)); //renormalize
						float occlusion = dot(textureLod(sampler3D(occlusion_texture, linear_sampler), occ_pos, 0.0), occ_mask);

						weight *= occlusion;
					}

					if (weight > 0.0) {
						vec3 tex_posf = base_tex_posf + vec3(octahedron_encode(n) * float(OCT_SIZE), 0.0);
						tex_posf.xy *= tex_pixel_size;

						vec3 pos_uvw = tex_posf;
						pos_uvw.xy += vec2(offset.xy) * probe_uv_offset.xy;
						pos_uvw.x += float(offset.z) * probe_uv_offset.z;
						vec3 indirect_light = textureLod(sampler2DArray(lightprobe_texture, linear_sampler), pos_uvw, 0.0).rgb;

						light_accum[k] += indirect_light * weight;
						weight_accum[k] += weight;
					}
				}
			}
		}

		for (uint k = 0; k < 6; k++) {
			if (weight_accum[k] > 0.0) {
				light_accum[k] /= weight_accum[k];
				light_accum[k] *= feedback;
			}
		}
	}

#endif

	{
		uint rgbe = process_voxels.data[voxel_index].light;

		//read rgbe8985
		float r = float((rgbe & 0xff) << 1);
		float g = float((rgbe >> 8) & 0x1ff);
		float b = float(((rgbe >> 17) & 0xff) << 1);
		float e = float((rgbe >> 25) & 0x1F);
		float m = pow(2.0, e - 15.0 - 9.0);

		vec3 l = vec3(r, g, b) * m;

		uint aniso = process_voxels.data[voxel_index].light_aniso;
		for (uint i = 0; i < 6; i++) {
			float strength = ((aniso >> (i * 5)) & 0x1F) / float(0x1F);
			light_accum[i] += l * strength;
		}
	}

	// Raytrace light

	for (uint i = 0; i < params.light_count; i++) {
		float attenuation = 1.0;
		vec3 direction;
		float light_distance = 1e20;
		vec3 texture_color = vec3(1.0);

		switch (lights.data[i].type) {
			case LIGHT_TYPE_DIRECTIONAL: {
				direction = -lights.data[i].direction;
			} break;
			case LIGHT_TYPE_OMNI: {
				vec3 rel_vec = lights.data[i].position - position;
				direction = normalize(rel_vec);
				light_distance = length(rel_vec);
				attenuation = get_omni_attenuation(length(sdfgi_unsquash(rel_vec)), 1.0 / lights.data[i].radius, lights.data[i].attenuation);

			} break;
			case LIGHT_TYPE_SPOT: {
				vec3 rel_vec = lights.data[i].position - position;
				direction = normalize(rel_vec);
				light_distance = length(rel_vec);
				vec3 true_vec = sdfgi_unsquash(rel_vec);
				attenuation = get_omni_attenuation(length(true_vec), 1.0 / lights.data[i].radius, lights.data[i].attenuation);

				float cos_spot_angle = lights.data[i].cos_spot_angle;
				float cos_angle = dot(-normalize(true_vec), lights.data[i].direction);

				if (cos_angle < cos_spot_angle) {
					continue;
				}

				float scos = max(cos_angle, cos_spot_angle);
				float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - cos_spot_angle));
				attenuation *= 1.0 - pow(spot_rim, lights.data[i].inv_spot_attenuation);
			} break;
			case LIGHT_TYPE_AREA: {
				float EPSILON = 1e-7f;
				vec3 area_width = lights.data[i].area_width.xyz;
				vec3 area_height = lights.data[i].area_height.xyz;
				if (dot(area_width, area_width) < EPSILON || dot(area_height, area_height) < EPSILON) {
					continue; // area is 0
				}
				if (dot(lights.data[i].direction, sdfgi_unsquash(position - lights.data[i].position)) <= 0) {
					continue; // position is behind light
				}
				compute_area_light(i, position, attenuation, direction, light_distance, texture_color);
			} break;
			case LIGHT_TYPE_LINE: {
				// One ray per sub-segment, to its middle. `area_width` is squashed like `position`.
				vec3 segment = lights.data[i].area_width.xyz;
				float min_radius = lights.data[i].area_height.x;
				vec3 p1 = lights.data[i].position - 0.5 * segment - position;
				vec3 segment_true = sdfgi_unsquash(segment);
				vec3 p1_true = sdfgi_unsquash(p1);
				float length_sq = dot(segment_true, segment_true);
				if (length_sq < 1e-12) {
					continue;
				}
				float dist = max(length(p1_true + segment_true * clamp(dot(-p1_true, segment_true) / length_sq, 0.0, 1.0)), min_radius);
				if (dist >= lights.data[i].radius) {
					continue;
				}
				// The integral already falls off with the inverse square of distance.
				float scale = M_PI * lights.data[i].area_height.y * get_omni_attenuation(dist, 1.0 / lights.data[i].radius, lights.data[i].attenuation) * dist * dist;

				vec3 sub = segment / float(LINE_LIGHT_GI_SAMPLES);
				float sub_attenuation[LINE_LIGHT_GI_SAMPLES];
				float total = 0.0;
				for (int k = 0; k < LINE_LIGHT_GI_SAMPLES; k++) {
					vec3 a = p1 + sub * float(k);
					vec3 mid = sdfgi_unsquash(a + 0.5 * sub);
					// Voxels have no normal: light them as if they faced the sub-segment.
					sub_attenuation[k] = dot(mid, mid) > 1e-12 ? scale * line_integrate_diffuse(sdfgi_unsquash(a), sdfgi_unsquash(a + sub), normalize(mid), min_radius) : 0.0;
					total += sub_attenuation[k];
				}
				if (total < 0.001) {
					continue;
				}

				for (int k = 0; k < LINE_LIGHT_GI_SAMPLES; k++) {
					vec3 mid = p1 + sub * (float(k) + 0.5);
					float mid_distance = length(mid);
					if (sub_attenuation[k] <= 0.0 || !sdf_light_visible(position, mid / mid_distance, mid_distance)) {
						continue;
					}
					vec3 light = albedo * lights.data[i].color.rgb * lights.data[i].energy * sub_attenuation[k];
					for (int j = 0; j < 6; j++) {
						if (bool(valid_aniso & (1 << j))) {
							light_accum[j] += max(0.0, dot(aniso_dir[j], mid / mid_distance)) * light;
						}
					}
				}
				continue;
			}
		}

		if (attenuation < 0.001) {
			continue;
		}

		if (sdf_light_visible(position, direction, light_distance)) {
			vec3 light = albedo * lights.data[i].color.rgb * texture_color * lights.data[i].energy * attenuation;

			for (int j = 0; j < 6; j++) {
				if (bool(valid_aniso & (1 << j))) {
					light_accum[j] += max(0.0, dot(aniso_dir[j], direction)) * light;
				}
			}
		}
	}

	// Store the light in the light texture

	float lumas[6];
	vec3 light_total = vec3(0);

	for (int i = 0; i < 6; i++) {
		light_total += light_accum[i];
		lumas[i] = max(light_accum[i].r, max(light_accum[i].g, light_accum[i].b));
	}

	float luma_total = max(light_total.r, max(light_total.g, light_total.b));

	uint light_total_rgbe;

	{
		//compress to RGBE9995 to save space

		const float pow2to9 = 512.0f;
		const float B = 15.0f;
		const float N = 9.0f;
		const float LN2 = 0.6931471805599453094172321215;

		float cRed = clamp(light_total.r, 0.0, 65408.0);
		float cGreen = clamp(light_total.g, 0.0, 65408.0);
		float cBlue = clamp(light_total.b, 0.0, 65408.0);

		float cMax = max(cRed, max(cGreen, cBlue));

		float expp = max(-B - 1.0f, floor(log(cMax) / LN2)) + 1.0f + B;

		float sMax = floor((cMax / pow(2.0f, expp - B - N)) + 0.5f);

		float exps = expp + 1.0f;

		if (0.0 <= sMax && sMax < pow2to9) {
			exps = expp;
		}

		float sRed = floor((cRed / pow(2.0f, exps - B - N)) + 0.5f);
		float sGreen = floor((cGreen / pow(2.0f, exps - B - N)) + 0.5f);
		float sBlue = floor((cBlue / pow(2.0f, exps - B - N)) + 0.5f);
#ifdef MODE_PROCESS_STATIC
		//since its self-save, use RGBE8985
		light_total_rgbe = ((uint(sRed) & 0x1FF) >> 1) | ((uint(sGreen) & 0x1FF) << 8) | (((uint(sBlue) & 0x1FF) >> 1) << 17) | ((uint(exps) & 0x1F) << 25);

#else
		light_total_rgbe = (uint(sRed) & 0x1FF) | ((uint(sGreen) & 0x1FF) << 9) | ((uint(sBlue) & 0x1FF) << 18) | ((uint(exps) & 0x1F) << 27);
#endif
	}

#ifdef MODE_PROCESS_DYNAMIC

	vec4 aniso0;
	aniso0.r = lumas[0] / luma_total;
	aniso0.g = lumas[1] / luma_total;
	aniso0.b = lumas[2] / luma_total;
	aniso0.a = lumas[3] / luma_total;

	vec2 aniso1;
	aniso1.r = lumas[4] / luma_total;
	aniso1.g = lumas[5] / luma_total;

	//save to 3D textures
	imageStore(dst_aniso0, positioni, aniso0);
	imageStore(dst_aniso1, positioni, vec4(aniso1, 0.0, 0.0));
	imageStore(dst_light, positioni, uvec4(light_total_rgbe));

	//also fill neighbors, so light interpolation during the indirect pass works

	//recover the neighbor list from the leftover bits
	uint neighbors = (voxel_albedo >> 21) | ((voxel_position >> 21) << 11) | ((process_voxels.data[voxel_index].light >> 30) << 22) | ((process_voxels.data[voxel_index].light_aniso >> 30) << 24);

	const uint max_neighbours = 26;
	const ivec3 neighbour_positions[max_neighbours] = ivec3[](
			ivec3(-1, -1, -1),
			ivec3(-1, -1, 0),
			ivec3(-1, -1, 1),
			ivec3(-1, 0, -1),
			ivec3(-1, 0, 0),
			ivec3(-1, 0, 1),
			ivec3(-1, 1, -1),
			ivec3(-1, 1, 0),
			ivec3(-1, 1, 1),
			ivec3(0, -1, -1),
			ivec3(0, -1, 0),
			ivec3(0, -1, 1),
			ivec3(0, 0, -1),
			ivec3(0, 0, 1),
			ivec3(0, 1, -1),
			ivec3(0, 1, 0),
			ivec3(0, 1, 1),
			ivec3(1, -1, -1),
			ivec3(1, -1, 0),
			ivec3(1, -1, 1),
			ivec3(1, 0, -1),
			ivec3(1, 0, 0),
			ivec3(1, 0, 1),
			ivec3(1, 1, -1),
			ivec3(1, 1, 0),
			ivec3(1, 1, 1));

	for (uint i = 0; i < max_neighbours; i++) {
		if (bool(neighbors & (1 << i))) {
			ivec3 neighbour_pos = positioni + neighbour_positions[i];
			imageStore(dst_light, neighbour_pos, uvec4(light_total_rgbe));
			imageStore(dst_aniso0, neighbour_pos, aniso0);
			imageStore(dst_aniso1, neighbour_pos, vec4(aniso1, 0.0, 0.0));
		}
	}

#endif

#ifdef MODE_PROCESS_STATIC

	//save back the anisotropic

	uint light = process_voxels.data[voxel_index].light & (3 << 30);
	light |= light_total_rgbe;
	process_voxels.data[voxel_index].light = light; //replace

	uint light_aniso = process_voxels.data[voxel_index].light_aniso & (3 << 30);
	for (int i = 0; i < 6; i++) {
		light_aniso |= min(31, uint((lumas[i] / luma_total) * 31.0)) << (i * 5);
	}

	process_voxels.data[voxel_index].light_aniso = light_aniso;

#endif
}
