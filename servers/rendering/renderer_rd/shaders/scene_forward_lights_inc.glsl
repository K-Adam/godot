// Functions related to lighting

#extension GL_EXT_control_flow_attributes : require

#include "area_lights_inc.glsl"

// Uncomment to shade line lights by brute-force Gauss-Legendre quadrature of the
// actual BRDF instead of the analytic integrals. Far slower, and only meant as a
// ground truth to validate light_process_line() against. Note this path
// regularizes the distance to each sample rather than the perpendicular
// distance to the line, so the two necessarily diverge where a surface touches
// the segment.
//#define LINE_LIGHT_REFERENCE

// This annotation macro must be placed before any loops that rely on specialization constants as their upper bound.
// Drivers may choose to unroll these loops based on the possible range of the value that can be deduced from the
// spec constant, which can lead to their code generation taking a much longer time than desired.
#ifdef UBERSHADER
// Prefer to not unroll loops on the ubershader to reduce code size as much as possible.
#define SPEC_CONSTANT_LOOP_ANNOTATION [[dont_unroll]]
#else
// Don't make an explicit hint on specialized shaders.
#define SPEC_CONSTANT_LOOP_ANNOTATION
#endif

half D_GGX(half NoH, half roughness, hvec3 n, hvec3 h) {
	half a = NoH * roughness;
#ifdef EXPLICIT_FP16
	hvec3 NxH = cross(n, h);
	half k = roughness / (dot(NxH, NxH) + a * a);
#else
	float k = roughness / (1.0 - NoH * NoH + a * a);
#endif
	half d = k * k * half(1.0 / M_PI);
	return saturateHalf(d);
}

#ifdef LIGHT_TRANSMITTANCE_USED
#ifdef SSS_MODE_SKIN
hvec3 SSS_skin(half NdotL, half transmittance_depth, half transmittance_z, half transmittance_boost, hvec4 transmittance_color, hvec3 light_color) {
	half scale = half(8.25) / transmittance_depth;
	half d = scale * abs(transmittance_z);
	float dd = float(-d * d);
	hvec3 profile = hvec3(vec3(0.233, 0.455, 0.649) * exp(dd / 0.0064) +
			vec3(0.1, 0.336, 0.344) * exp(dd / 0.0484) +
			vec3(0.118, 0.198, 0.0) * exp(dd / 0.187) +
			vec3(0.113, 0.007, 0.007) * exp(dd / 0.567) +
			vec3(0.358, 0.004, 0.0) * exp(dd / 1.99) +
			vec3(0.078, 0.0, 0.0) * exp(dd / 7.41));

	return profile * transmittance_color.a * light_color * clamp(transmittance_boost - NdotL, half(0.0), half(1.0)) * half(1.0 / M_PI);
}
#else
hvec3 SSS(half NdotL, half transmittance_depth, half transmittance_z, half transmittance_boost, hvec4 transmittance_color, hvec3 light_color) {
	half scale = half(8.25) / transmittance_depth;
	half d = scale * abs(transmittance_z);
	half dd = -d * d;
	return exp(dd) * transmittance_color.rgb * transmittance_color.a * light_color * clamp(transmittance_boost - NdotL, half(0.0), half(1.0)) * half(1.0 / M_PI);
}
#endif
#endif

// From Earl Hammon, Jr. "PBR Diffuse Lighting for GGX+Smith Microsurfaces" https://www.gdcvault.com/play/1024478/PBR-Diffuse-Lighting-for-GGX
half V_GGX(half NdotL, half NdotV, half alpha) {
	half v = half(0.5) / mix(half(2.0) * NdotL * NdotV, NdotL + NdotV, alpha);
	return saturateHalf(v);
}

half D_GGX_anisotropic(half cos_theta_m, half alpha_x, half alpha_y, half cos_phi, half sin_phi) {
	half alpha2 = alpha_x * alpha_y;
	vec3 v = vec3(alpha_y * cos_phi, alpha_x * sin_phi, alpha2 * cos_theta_m);
	float v2 = dot(v, v);
	half w2 = half(float(alpha2) / v2);
	return alpha2 * w2 * w2 * half(1.0 / M_PI);
}

half V_GGX_anisotropic(half alpha_x, half alpha_y, half TdotV, half TdotL, half BdotV, half BdotL, half NdotV, half NdotL) {
	half Lambda_V = NdotL * length(hvec3(alpha_x * TdotV, alpha_y * BdotV, NdotV));
	half Lambda_L = NdotV * length(hvec3(alpha_x * TdotL, alpha_y * BdotL, NdotL));
	half v = half(0.5) / (Lambda_V + Lambda_L);
	return saturateHalf(v);
}

half SchlickFresnel(half u) {
	half m = half(1.0) - u;
	half m2 = m * m;
	return m2 * m2 * m; // pow(m,5)
}

hvec3 F0(half metallic, half specular, hvec3 albedo) {
	half dielectric = half(0.16) * specular * specular;
	// use albedo * metallic as colored specular reflectance at 0 angle for metallic materials;
	// see https://google.github.io/filament/Filament.md.html
	return mix(hvec3(dielectric), albedo, hvec3(metallic));
}

half V_Kelemen(half LdotH) {
	// Kelemen 2001, "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
	return saturateHalf(half(0.25) / (LdotH * LdotH + half(1e-4)));
}

hvec3 f0_Clear_Coat_To_Surface(hvec3 f0) {
	// Approximation of iorTof0(f0ToIor(f0), 1.5)
	// This assumes that the clear coat layer has an IOR of 1.5
	// see https://github.com/google/filament/blob/837b2715a05f4656d4f524bce50d1b23ff8f84c9/shaders/src/surface_material.fs#L54-L62
	return clamp(f0 * (f0 * (half(0.941892) - half(0.263008) * f0) + half(0.346479)) - half(0.0285998), half(0.0), half(1.0));
}

void light_compute(hvec3 N, hvec3 L, hvec3 V, half A, hvec3 light_color, bool is_directional, half attenuation, hvec3 f0, half roughness, half metallic, half specular_amount, hvec3 albedo, inout half alpha, vec2 screen_uv, hvec3 energy_compensation,
#ifdef LIGHT_BACKLIGHT_USED
		hvec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		hvec4 transmittance_color,
		half transmittance_depth,
		half transmittance_boost,
		half transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
		half rim, half rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		half clearcoat, half clearcoat_roughness, hvec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		hvec3 B, hvec3 T, half anisotropy,
#endif
		inout hvec3 diffuse_light, inout hvec3 specular_light) {
#if defined(LIGHT_CODE_USED)
	// Light is written by the user shader.
	mat4 inv_view_matrix = transpose(mat4(scene_data_block.data.inv_view_matrix[0],
			scene_data_block.data.inv_view_matrix[1],
			scene_data_block.data.inv_view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));
	mat4 read_view_matrix = transpose(mat4(scene_data_block.data.view_matrix[0],
			scene_data_block.data.view_matrix[1],
			scene_data_block.data.view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

#ifdef USING_MOBILE_RENDERER
	uint instance_index = draw_call.instance_index;
#else
	uint instance_index = instance_index_interp;
#endif

	mat4 read_model_matrix = transpose(mat4(instances.data[instance_index].transform[0],
			instances.data[instance_index].transform[1],
			instances.data[instance_index].transform[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

#undef projection_matrix
#define projection_matrix scene_data_block.data.projection_matrix
#undef inv_projection_matrix
#define inv_projection_matrix scene_data_block.data.inv_projection_matrix

	vec2 read_viewport_size = scene_data_block.data.viewport_size;

#ifdef LIGHT_BACKLIGHT_USED
	vec3 backlight_highp = vec3(backlight);
#endif
	float roughness_highp = float(roughness);
	float metallic_highp = float(metallic);
	vec3 albedo_highp = vec3(albedo);
	float alpha_highp = float(alpha);
	vec3 normal_highp = vec3(N);
	vec3 light_highp = vec3(L);
	vec3 view_highp = vec3(V);
	float specular_amount_highp = float(specular_amount);
	vec3 light_color_highp = vec3(light_color);
	float attenuation_highp = float(attenuation);
	vec3 diffuse_light_highp = vec3(diffuse_light);
	vec3 specular_light_highp = vec3(specular_light);
	bool is_area = false;
	vec3 area_diffuse = vec3(1.0);
	vec3 area_specular = vec3(1.0);
	vec3 area_diffuse_tex_color = vec3(1.0);
	vec3 area_specular_tex_color = vec3(1.0);

#CODE : LIGHT

	alpha = half(alpha_highp);
	diffuse_light = hvec3(diffuse_light_highp);
	specular_light = hvec3(specular_light_highp);
#else // !LIGHT_CODE_USED
	half NdotL = min(A + dot(N, L), half(1.0));
	half cNdotV = max(dot(N, V), half(1e-4));

#ifdef LIGHT_TRANSMITTANCE_USED
	{
#ifdef SSS_MODE_SKIN
		diffuse_light += SSS_skin(NdotL, transmittance_depth, transmittance_z, transmittance_boost, transmittance_color, light_color);
#else
		diffuse_light += SSS(NdotL, transmittance_depth, transmittance_z, transmittance_boost, transmittance_color, light_color);
#endif
	}
#endif //LIGHT_TRANSMITTANCE_USED

#if defined(LIGHT_RIM_USED)
	// Epsilon min to prevent pow(0, 0) singularity which results in undefined behavior.
	half rim_light = pow(max(half(1e-4), half(1.0) - cNdotV), max(half(0.0), (half(1.0) - roughness) * half(16.0)));
	diffuse_light += rim_light * rim * mix(hvec3(1.0), albedo, rim_tint) * light_color;
#endif

	half cc_attenuation = half(1.0);

	// We skip checking on attenuation on directional lights to avoid a branch that is not as beneficial for directional lights as the other ones.
	if (is_directional || attenuation > HALF_FLT_MIN) {
		half cNdotL = max(NdotL, half(0.0));
#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
		hvec3 H = normalize(V + L);
		half cLdotH = clamp(A + dot(L, H), half(0.0), half(1.0));
#endif

#if defined(LIGHT_CLEARCOAT_USED)
		// Clearcoat ignores normal_map, use vertex normal instead
		half ccNdotL = clamp(A + dot(vertex_normal, L), half(0.0), half(1.0));
		half ccNdotH = clamp(A + dot(vertex_normal, H), half(0.0), half(1.0));
		half cLdotH5 = SchlickFresnel(cLdotH);

		half Dr = D_GGX(ccNdotH, half(mix(half(0.001), half(0.1), clearcoat_roughness)), vertex_normal, H);
		half Gr = V_Kelemen(cLdotH);
		half Fr = mix(half(0.04), half(1.0), cLdotH5) * clearcoat;
		cc_attenuation = half(1.0) - Fr;
		half clearcoat_specular_brdf_NL = Dr * Gr * Fr * ccNdotL;
		specular_light += clearcoat_specular_brdf_NL * light_color * attenuation * specular_amount;
#endif // LIGHT_CLEARCOAT_USED

		if (metallic < half(1.0)) {
			half diffuse_brdf_NL; // BRDF times N.L for calculating diffuse radiance

#if defined(DIFFUSE_LAMBERT_WRAP)
			// Energy conserving lambert wrap shader.
			// https://web.archive.org/web/20210228210901/http://blog.stevemcauley.com/2011/12/03/energy-conserving-wrapped-diffuse/
			half op_roughness = half(1.0) + roughness;
			diffuse_brdf_NL = max(half(0.0), (NdotL + roughness) / (op_roughness * op_roughness)) * half(1.0 / M_PI);
#elif defined(DIFFUSE_TOON)

			diffuse_brdf_NL = smoothstep(-roughness, max(roughness, half(0.01)), NdotL) * half(1.0 / M_PI);

#elif defined(DIFFUSE_BURLEY)
			{
				half FD90_minus_1 = half(2.0) * cLdotH * cLdotH * roughness - half(0.5);
				half FdV = half(1.0) + FD90_minus_1 * SchlickFresnel(cNdotV);
				half FdL = half(1.0) + FD90_minus_1 * SchlickFresnel(cNdotL);
				diffuse_brdf_NL = half(1.0 / M_PI) * FdV * FdL * cNdotL;
			}
#else
			// lambert
			diffuse_brdf_NL = cNdotL * half(1.0 / M_PI);
#endif

			diffuse_light += light_color * diffuse_brdf_NL * attenuation * cc_attenuation;

#if defined(LIGHT_BACKLIGHT_USED)
			diffuse_light += light_color * (hvec3(1.0 / M_PI) - diffuse_brdf_NL) * backlight * attenuation;
#endif
		}

		if (roughness > half(0.0)) {
#if defined(SPECULAR_SCHLICK_GGX)
			half cNdotH = clamp(A + dot(N, H), half(0.0), half(1.0));
#endif
			// Apply specular light.
			// FIXME: roughness == 0 should not disable specular light entirely
#if defined(SPECULAR_TOON)
			hvec3 R = normalize(-reflect(L, N));
			half RdotV = dot(R, V);
			half mid = half(1.0) - roughness;
			mid *= mid;
			half intensity = smoothstep(mid - roughness * half(0.5), mid + roughness * half(0.5), RdotV) * mid;
			diffuse_light += light_color * intensity * attenuation * specular_amount; // write to diffuse_light, as in toon shading you generally want no reflection

#elif defined(SPECULAR_DISABLED)
			// Do nothing.

#elif defined(SPECULAR_SCHLICK_GGX)
			// shlick+ggx as default
			half alpha_ggx = roughness * roughness;
#if defined(LIGHT_ANISOTROPY_USED)
			half aspect = sqrt(half(1.0) - anisotropy * half(0.9));
			half ax = alpha_ggx / aspect;
			half ay = alpha_ggx * aspect;
			half XdotH = dot(T, H);
			half YdotH = dot(B, H);
			half D = D_GGX_anisotropic(cNdotH, ax, ay, XdotH, YdotH);
			half G = V_GGX_anisotropic(ax, ay, dot(T, V), dot(T, L), dot(B, V), dot(B, L), cNdotV, cNdotL);
#else // LIGHT_ANISOTROPY_USED
			half D = D_GGX(cNdotH, alpha_ggx, N, H);
			half G = V_GGX(cNdotL, cNdotV, alpha_ggx);
#endif // LIGHT_ANISOTROPY_USED
	   // F
#if !defined(LIGHT_CLEARCOAT_USED)
			half cLdotH5 = SchlickFresnel(cLdotH);
#endif
			// Calculate Fresnel using specular occlusion term from Filament:
			// https://google.github.io/filament/Filament.html#lighting/occlusion/specularocclusion
			half f90 = clamp(dot(f0, hvec3(50.0 * 0.33)), metallic, half(1.0));
			hvec3 F = f0 + (f90 - f0) * cLdotH5;
			hvec3 specular_brdf_NL = energy_compensation * cNdotL * D * F * G;
			specular_light += specular_brdf_NL * light_color * attenuation * cc_attenuation * specular_amount;
#endif
		}

#ifdef USE_SHADOW_TO_OPACITY
		alpha = min(alpha, clamp(half(1.0 - attenuation), half(0.0), half(1.0)));
#endif
	}
#endif // LIGHT_CODE_USED
}

#ifndef SHADOWS_DISABLED

// Interleaved Gradient Noise
// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
float quick_hash(vec2 pos) {
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	return fract(magic.z * fract(dot(pos, magic.xy)));
}

half sample_directional_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec4 coord, float taa_frame_count) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	//if only one sample is taken, take it from the center
	if (sc_directional_soft_shadow_samples() == 0) {
		return half(textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0)));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;

	SPEC_CONSTANT_LOOP_ANNOTATION
	for (uint i = 0; i < sc_directional_soft_shadow_samples(); i++) {
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos + shadow_pixel_size * (disk_rotation * scene_data_block.data.directional_soft_shadow_kernel[i].xy), depth, 1.0));
	}

	return half(avg * (1.0 / float(sc_directional_soft_shadow_samples())));
}

half sample_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec3 coord, float taa_frame_count) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	//if only one sample is taken, take it from the center
	if (sc_soft_shadow_samples() == 0) {
		return half(textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0)));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;

	SPEC_CONSTANT_LOOP_ANNOTATION
	for (uint i = 0; i < sc_soft_shadow_samples(); i++) {
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos + shadow_pixel_size * (disk_rotation * scene_data_block.data.soft_shadow_kernel[i].xy), depth, 1.0));
	}

	return half(avg * (1.0 / float(sc_soft_shadow_samples())));
}

half sample_omni_pcf_shadow(texture2D shadow, float blur_scale, vec2 coord, vec4 uv_rect, vec2 flip_offset, float depth, float taa_frame_count) {
	//if only one sample is taken, take it from the center
	if (sc_soft_shadow_samples() == 0) {
		vec2 pos = coord * 0.5 + 0.5;
		pos = uv_rect.xy + pos * uv_rect.zw;
		return half(textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0)));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;
	vec2 offset_scale = blur_scale * 2.0 * scene_data_block.data.shadow_atlas_pixel_size / uv_rect.zw;

	SPEC_CONSTANT_LOOP_ANNOTATION
	for (uint i = 0; i < sc_soft_shadow_samples(); i++) {
		vec2 offset = offset_scale * (disk_rotation * scene_data_block.data.soft_shadow_kernel[i].xy);
		vec2 sample_coord = coord + offset;

		float sample_coord_length_squared = dot(sample_coord, sample_coord);
		bool do_flip = sample_coord_length_squared > 1.0;

		if (do_flip) {
			float len = sqrt(sample_coord_length_squared);
			sample_coord = sample_coord * (2.0 / len - 1.0);
		}

		sample_coord = sample_coord * 0.5 + 0.5;
		sample_coord = uv_rect.xy + sample_coord * uv_rect.zw;

		if (do_flip) {
			sample_coord += flip_offset;
		}
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(sample_coord, depth, 1.0));
	}

	return half(avg * (1.0 / float(sc_soft_shadow_samples())));
}

half sample_directional_soft_shadow(texture2D shadow, vec3 pssm_coord, vec2 tex_scale, float taa_frame_count) {
	//find blocker
	float blocker_count = 0.0;
	float blocker_average = 0.0;

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	SPEC_CONSTANT_LOOP_ANNOTATION
	for (uint i = 0; i < sc_directional_penumbra_shadow_samples(); i++) {
		vec2 suv = pssm_coord.xy + (disk_rotation * scene_data_block.data.directional_penumbra_shadow_kernel[i].xy) * tex_scale;
		float d = textureLod(sampler2D(shadow, SAMPLER_LINEAR_CLAMP), suv, 0.0).r;
		if (d > pssm_coord.z) {
			blocker_average += d;
			blocker_count += 1.0;
		}
	}

	if (blocker_count > 0.0) {
		//blockers found, do soft shadow
		blocker_average /= blocker_count;
		float penumbra = (-pssm_coord.z + blocker_average) / (1.0 - blocker_average);
		tex_scale *= penumbra;

		float s = 0.0;

		SPEC_CONSTANT_LOOP_ANNOTATION
		for (uint i = 0; i < sc_directional_penumbra_shadow_samples(); i++) {
			vec2 suv = pssm_coord.xy + (disk_rotation * scene_data_block.data.directional_penumbra_shadow_kernel[i].xy) * tex_scale;
			s += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(suv, pssm_coord.z, 1.0));
		}

		return half(s / float(sc_directional_penumbra_shadow_samples()));

	} else {
		//no blockers found, so no shadow
		return half(1.0);
	}
}

#endif // SHADOWS_DISABLED

half get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return half(nd * pow(max(distance, 0.0001), -decay));
}

void light_process_omni(uint idx, vec3 vertex, hvec3 eye_vec, hvec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, hvec3 f0, half roughness, half metallic, float taa_frame_count, hvec3 albedo, inout half alpha, vec2 screen_uv, hvec3 energy_compensation,
#ifdef LIGHT_BACKLIGHT_USED
		hvec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		hvec4 transmittance_color,
		half transmittance_depth,
		half transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		half rim, half rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		half clearcoat, half clearcoat_roughness, hvec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		hvec3 binormal, hvec3 tangent, half anisotropy,
#endif
		inout hvec3 diffuse_light, inout hvec3 specular_light) {

	// Omni light attenuation.
	vec3 light_rel_vec = omni_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	half omni_attenuation = get_omni_attenuation(light_length, omni_lights.data[idx].inv_radius, omni_lights.data[idx].attenuation);

	// Compute size.
	half size = half(0.0);
	if (sc_use_light_soft_shadows() && omni_lights.data[idx].size > 0.0) {
		half t = half(omni_lights.data[idx].size / max(0.001, light_length));
		size = half(1.0) / sqrt(half(1.0) + t * t);
		size = max(half(1.0) - size, half(0.0));
	}

	half shadow = half(1.0);
#ifndef SHADOWS_DISABLED
	// Omni light shadow.
	if (omni_attenuation > HALF_FLT_MIN && omni_lights.data[idx].shadow_opacity > 0.001) {
		// there is a shadowmap
		vec2 texel_size = scene_data_block.data.shadow_atlas_pixel_size;
		vec4 base_uv_rect = omni_lights.data[idx].atlas_rect;
		base_uv_rect.xy += texel_size;
		base_uv_rect.zw -= texel_size * 2.0;

		// Omni lights use direction.xy to store to store the offset between the two paraboloid regions
		vec2 flip_offset = omni_lights.data[idx].direction.xy;

		vec3 local_vert = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;

		float shadow_len = length(local_vert); //need to remember shadow len from here
		vec3 shadow_dir = normalize(local_vert);

		vec3 local_normal = normalize(mat3(omni_lights.data[idx].shadow_matrix) * vec3(normal));
		vec3 normal_bias = local_normal * omni_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(local_normal, shadow_dir)));

		if (sc_use_light_soft_shadows() && omni_lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker

			float blocker_count = 0.0;
			float blocker_average = 0.0;

			mat2 disk_rotation;
			{
				float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
				float sr = sin(r);
				float cr = cos(r);
				disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
			}

			vec3 basis_normal = shadow_dir;
			vec3 v0 = abs(basis_normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
			vec3 tangent = normalize(cross(v0, basis_normal));
			vec3 bitangent = normalize(cross(tangent, basis_normal));
			float z_norm = 1.0 - shadow_len * omni_lights.data[idx].inv_radius;

			tangent *= omni_lights.data[idx].soft_shadow_size * omni_lights.data[idx].soft_shadow_scale;
			bitangent *= omni_lights.data[idx].soft_shadow_size * omni_lights.data[idx].soft_shadow_scale;

			SPEC_CONSTANT_LOOP_ANNOTATION
			for (uint i = 0; i < sc_penumbra_shadow_samples(); i++) {
				vec2 disk = disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy;

				vec3 pos = local_vert + tangent * disk.x + bitangent * disk.y;

				pos = normalize(pos);

				vec4 uv_rect = base_uv_rect;

				if (pos.z >= 0.0) {
					uv_rect.xy += flip_offset;
				}

				pos.z = 1.0 + abs(pos.z);
				pos.xy /= pos.z;

				pos.xy = pos.xy * 0.5 + 0.5;
				pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;

				float d = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), pos.xy, 0.0).r;
				if (d > z_norm) {
					blocker_average += d;
					blocker_count += 1.0;
				}
			}

			if (blocker_count > 0.0) {
				//blockers found, do soft shadow
				blocker_average /= blocker_count;
				float penumbra = (-z_norm + blocker_average) / (1.0 - blocker_average);
				tangent *= penumbra;
				bitangent *= penumbra;

				z_norm += omni_lights.data[idx].inv_radius * omni_lights.data[idx].shadow_bias;

				shadow = half(0.0);

				SPEC_CONSTANT_LOOP_ANNOTATION
				for (uint i = 0; i < sc_penumbra_shadow_samples(); i++) {
					vec2 disk = disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy;
					vec3 pos = local_vert + tangent * disk.x + bitangent * disk.y;

					pos = normalize(pos);
					pos = normalize(pos + normal_bias);

					vec4 uv_rect = base_uv_rect;

					if (pos.z >= 0.0) {
						uv_rect.xy += flip_offset;
					}

					pos.z = 1.0 + abs(pos.z);
					pos.xy /= pos.z;

					pos.xy = pos.xy * 0.5 + 0.5;
					pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;
					shadow += half(textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(pos.xy, z_norm, 1.0)));
				}

				shadow /= half(sc_penumbra_shadow_samples());
				shadow = mix(half(1.0), shadow, half(omni_lights.data[idx].shadow_opacity));

			} else {
				//no blockers found, so no shadow
				shadow = half(1.0);
			}
		} else {
			vec4 uv_rect = base_uv_rect;

			vec3 shadow_sample = normalize(shadow_dir + normal_bias);
			if (shadow_sample.z >= 0.0) {
				uv_rect.xy += flip_offset;
				flip_offset *= -1.0;
			}

			shadow_sample.z = 1.0 + abs(shadow_sample.z);
			vec2 pos = shadow_sample.xy / shadow_sample.z;
			float depth = shadow_len - omni_lights.data[idx].shadow_bias;
			depth *= omni_lights.data[idx].inv_radius;
			depth = 1.0 - depth;
			shadow = mix(half(1.0), sample_omni_pcf_shadow(shadow_atlas, omni_lights.data[idx].soft_shadow_scale / shadow_sample.z, pos, uv_rect, flip_offset, depth, taa_frame_count), half(omni_lights.data[idx].shadow_opacity));
		}
	}
#endif

	vec3 color = omni_lights.data[idx].color;

#ifdef LIGHT_TRANSMITTANCE_USED
	half transmittance_z = transmittance_depth; //no transmittance by default
	transmittance_color.a *= omni_attenuation;
#ifndef SHADOWS_DISABLED
	if (omni_lights.data[idx].shadow_opacity > 0.001) {
		// Redo shadowmapping, but shrink the model a bit to avoid artifacts.
		vec2 texel_size = scene_data_block.data.shadow_atlas_pixel_size;
		vec4 uv_rect = omni_lights.data[idx].atlas_rect;
		uv_rect.xy += texel_size;
		uv_rect.zw -= texel_size * 2.0;

		// Omni lights use direction.xy to store to store the offset between the two paraboloid regions
		vec2 flip_offset = omni_lights.data[idx].direction.xy;

		vec3 local_vert = (omni_lights.data[idx].shadow_matrix * vec4(vertex - normal * omni_lights.data[idx].transmittance_bias, 1.0)).xyz;

		float shadow_len = length(local_vert); //need to remember shadow len from here
		vec3 shadow_sample = normalize(local_vert);

		if (shadow_sample.z >= 0.0) {
			uv_rect.xy += flip_offset;
			flip_offset *= -1.0;
		}

		shadow_sample.z = 1.0 + abs(shadow_sample.z);
		vec2 pos = shadow_sample.xy / shadow_sample.z;
		float depth = shadow_len * omni_lights.data[idx].inv_radius;
		depth = 1.0 - depth;

		pos = pos * 0.5 + 0.5;
		pos = uv_rect.xy + pos * uv_rect.zw;
		float shadow_z = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), pos, 0.0).r;
		transmittance_z = half((depth - shadow_z) / omni_lights.data[idx].inv_radius);
	}
#endif // !SHADOWS_DISABLED
#endif // LIGHT_TRANSMITTANCE_USED

	if (sc_use_light_projector() && omni_lights.data[idx].projector_rect != vec4(0.0)) {
		vec3 local_v = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;
		local_v = normalize(local_v);

		vec4 atlas_rect = omni_lights.data[idx].projector_rect;

		if (local_v.z >= 0.0) {
			atlas_rect.y += atlas_rect.w;
		}

		local_v.z = 1.0 + abs(local_v.z);

		local_v.xy /= local_v.z;
		local_v.xy = local_v.xy * 0.5 + 0.5;
		vec2 proj_uv = local_v.xy * atlas_rect.zw;

		// Clamp the uv coordinates to prevent bleeding from neighboring decals.
		vec2 texel_size = 1.0 / vec2(textureSize(sampler2D(decal_atlas_srgb, light_projector_sampler), 0));
		proj_uv = clamp(proj_uv, texel_size * 0.5, atlas_rect.zw - texel_size * 0.5);

		if (sc_projector_use_mipmaps()) {
			vec2 proj_uv_ddx;
			vec2 proj_uv_ddy;
			{
				vec3 local_v_ddx = (omni_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0)).xyz;
				local_v_ddx = normalize(local_v_ddx);

				if (local_v_ddx.z >= 0.0) {
					local_v_ddx.z += 1.0;
				} else {
					local_v_ddx.z = 1.0 - local_v_ddx.z;
				}

				local_v_ddx.xy /= local_v_ddx.z;
				local_v_ddx.xy = local_v_ddx.xy * 0.5 + 0.5;

				proj_uv_ddx = local_v_ddx.xy * atlas_rect.zw - proj_uv;

				vec3 local_v_ddy = (omni_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0)).xyz;
				local_v_ddy = normalize(local_v_ddy);

				if (local_v_ddy.z >= 0.0) {
					local_v_ddy.z += 1.0;
				} else {
					local_v_ddy.z = 1.0 - local_v_ddy.z;
				}

				local_v_ddy.xy /= local_v_ddy.z;
				local_v_ddy.xy = local_v_ddy.xy * 0.5 + 0.5;

				proj_uv_ddy = local_v_ddy.xy * atlas_rect.zw - proj_uv;
			}

			vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + atlas_rect.xy, proj_uv_ddx, proj_uv_ddy);
			color *= proj.rgb * proj.a;
		} else {
			vec4 proj = textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + atlas_rect.xy, 0.0);
			color *= proj.rgb * proj.a;
		}
	}

	vec3 light_rel_vec_norm = light_rel_vec / light_length;
	light_compute(normal, hvec3(light_rel_vec_norm), eye_vec, size, hvec3(color), false, omni_attenuation * shadow, f0, roughness, metallic, half(omni_lights.data[idx].specular_amount), albedo, alpha, screen_uv, energy_compensation,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_boost,
			transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
			rim * omni_attenuation, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			binormal, tangent, anisotropy,
#endif
			diffuse_light,
			specular_light);
}

vec2 normal_to_panorama(vec3 n) {
	n = normalize(n);
	vec2 panorama_coords = vec2(atan(n.x, n.z), acos(-n.y));

	if (panorama_coords.x < 0.0) {
		panorama_coords.x += M_PI * 2.0;
	}

	panorama_coords /= vec2(M_PI * 2.0, M_PI);
	return panorama_coords;
}

void light_process_spot(uint idx, vec3 vertex, hvec3 eye_vec, hvec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, hvec3 f0, half roughness, half metallic, float taa_frame_count, hvec3 albedo, inout half alpha, vec2 screen_uv, hvec3 energy_compensation,
#ifdef LIGHT_BACKLIGHT_USED
		hvec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		hvec4 transmittance_color,
		half transmittance_depth,
		half transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		half rim, half rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		half clearcoat, half clearcoat_roughness, hvec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		hvec3 binormal, hvec3 tangent, half anisotropy,
#endif
		inout hvec3 diffuse_light,
		inout hvec3 specular_light) {

	// Spot light attenuation.
	vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	hvec3 light_rel_vec_norm = hvec3(light_rel_vec / light_length);
	half spot_attenuation = get_omni_attenuation(light_length, spot_lights.data[idx].inv_radius, spot_lights.data[idx].attenuation);
	vec3 spot_dir = spot_lights.data[idx].direction;
	float cone_angle = spot_lights.data[idx].cone_angle;
	float scos = max(dot(-vec3(light_rel_vec_norm), spot_dir), cone_angle);

	// This conversion to a highp float is crucial to prevent light leaking due to precision errors.
	float spot_rim = max(1e-4, (1.0 - scos) / (1.0 - cone_angle));
	spot_attenuation *= half(1.0 - pow(spot_rim, spot_lights.data[idx].cone_attenuation));

	// Compute size.
	half size = half(0.0);
	if (sc_use_light_soft_shadows() && spot_lights.data[idx].size > 0.0) {
		half t = half(spot_lights.data[idx].size / max(0.001, light_length));
		size = half(1.0) / sqrt(half(1.0) + t * t);
		size = max(half(1.0) - size, half(0.0));
	}

	half shadow = half(1.0);
#ifndef SHADOWS_DISABLED
	// Spot light shadow.
	if (spot_attenuation > HALF_FLT_MIN && spot_lights.data[idx].shadow_opacity > 0.001) {
		vec3 normal_bias = vec3(normal) * light_length * spot_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(normal, light_rel_vec_norm)));

		//there is a shadowmap
		vec4 v = vec4(vertex + normal_bias, 1.0);

		vec4 splane = (spot_lights.data[idx].shadow_matrix * v);
		splane.z += spot_lights.data[idx].shadow_bias;
		splane /= splane.w;

		if (sc_use_light_soft_shadows() && spot_lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker
			float z_norm = dot(spot_dir, -light_rel_vec) * spot_lights.data[idx].inv_radius;

			vec2 shadow_uv = splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy;

			float blocker_count = 0.0;
			float blocker_average = 0.0;

			mat2 disk_rotation;
			{
				float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
				float sr = sin(r);
				float cr = cos(r);
				disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
			}

			float uv_size = spot_lights.data[idx].soft_shadow_size * z_norm * spot_lights.data[idx].soft_shadow_scale;
			vec2 clamp_max = spot_lights.data[idx].atlas_rect.xy + spot_lights.data[idx].atlas_rect.zw;

			SPEC_CONSTANT_LOOP_ANNOTATION
			for (uint i = 0; i < sc_penumbra_shadow_samples(); i++) {
				vec2 suv = shadow_uv + (disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy) * uv_size;
				suv = clamp(suv, spot_lights.data[idx].atlas_rect.xy, clamp_max);
				float d = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), suv, 0.0).r;
				if (d > splane.z) {
					blocker_average += d;
					blocker_count += 1.0;
				}
			}

			if (blocker_count > 0.0) {
				//blockers found, do soft shadow
				blocker_average /= blocker_count;
				float penumbra = (-z_norm + blocker_average) / (1.0 - blocker_average);
				uv_size *= penumbra;

				shadow = half(0.0);

				SPEC_CONSTANT_LOOP_ANNOTATION
				for (uint i = 0; i < sc_penumbra_shadow_samples(); i++) {
					vec2 suv = shadow_uv + (disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy) * uv_size;
					suv = clamp(suv, spot_lights.data[idx].atlas_rect.xy, clamp_max);
					shadow += half(textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(suv, splane.z, 1.0)));
				}

				shadow /= half(sc_penumbra_shadow_samples());
				shadow = mix(half(1.0), shadow, half(spot_lights.data[idx].shadow_opacity));

			} else {
				//no blockers found, so no shadow
				shadow = half(1.0);
			}
		} else {
			//hard shadow
			vec3 shadow_uv = vec3(splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy, splane.z);
			shadow = mix(half(1.0), sample_pcf_shadow(shadow_atlas, spot_lights.data[idx].soft_shadow_scale * scene_data_block.data.shadow_atlas_pixel_size, shadow_uv, taa_frame_count), half(spot_lights.data[idx].shadow_opacity));
		}
	}
#endif // SHADOWS_DISABLED

	vec3 color = spot_lights.data[idx].color;

#ifdef LIGHT_TRANSMITTANCE_USED
	half transmittance_z = transmittance_depth;
	transmittance_color.a *= spot_attenuation;
#ifndef SHADOWS_DISABLED
	if (spot_lights.data[idx].shadow_opacity > 0.001) {
		vec4 splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex - vec3(normal) * spot_lights.data[idx].transmittance_bias, 1.0));
		splane /= splane.w;

		vec3 shadow_uv = vec3(splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy, splane.z);
		float shadow_z = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), shadow_uv.xy, 0.0).r;

		shadow_z = shadow_z * 2.0 - 1.0;
		float z_far = 1.0 / spot_lights.data[idx].inv_radius;
		float z_near = 0.01;
		shadow_z = 2.0 * z_near * z_far / (z_far + z_near - shadow_z * (z_far - z_near));

		//distance to light plane
		float z = dot(spot_dir, -light_rel_vec);
		transmittance_z = half(z - shadow_z);
	}
#endif // !SHADOWS_DISABLED
#endif // LIGHT_TRANSMITTANCE_USED

	if (sc_use_light_projector() && spot_lights.data[idx].projector_rect != vec4(0.0)) {
		vec4 splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex, 1.0));
		splane /= splane.w;

		vec2 proj_uv = splane.xy * spot_lights.data[idx].projector_rect.zw;

		if (sc_projector_use_mipmaps()) {
			//ensure we have proper mipmaps
			vec4 splane_ddx = (spot_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0));
			splane_ddx /= splane_ddx.w;
			vec2 proj_uv_ddx = splane_ddx.xy * spot_lights.data[idx].projector_rect.zw - proj_uv;

			vec4 splane_ddy = (spot_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0));
			splane_ddy /= splane_ddy.w;
			vec2 proj_uv_ddy = splane_ddy.xy * spot_lights.data[idx].projector_rect.zw - proj_uv;

			vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + spot_lights.data[idx].projector_rect.xy, proj_uv_ddx, proj_uv_ddy);
			color *= proj.rgb * proj.a;
		} else {
			vec4 proj = textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + spot_lights.data[idx].projector_rect.xy, 0.0);
			color *= proj.rgb * proj.a;
		}
	}

	light_compute(normal, hvec3(light_rel_vec_norm), eye_vec, size, hvec3(color), false, spot_attenuation * shadow, f0, roughness, metallic, half(spot_lights.data[idx].specular_amount), albedo, alpha, screen_uv, energy_compensation,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_boost,
			transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
			rim * spot_attenuation, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			binormal, tangent, anisotropy,
#endif

			diffuse_light, specular_light);
}

// implementation of area lights with Linearly Transformed Cosines (LTC): https://eheitzresearch.wordpress.com/415-2/
void light_process_area(uint idx, vec3 vertex, hvec3 eye_vec, hvec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, hvec3 f0, half roughness, half metallic, float taa_frame_count, hvec3 albedo, inout half alpha, vec2 screen_uv, hvec3 energy_compensation,
#ifdef LIGHT_BACKLIGHT_USED
		hvec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		hvec4 transmittance_color,
		half transmittance_depth,
		half transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		half rim, half rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		half clearcoat, half clearcoat_roughness, hvec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		hvec3 binormal, hvec3 tangent, half anisotropy,
#endif
		inout hvec3 diffuse_light, inout hvec3 specular_light) {
	half EPSILON = half(1e-7);
	hvec3 area_width = hvec3(area_lights.data[idx].area_width);
	hvec3 area_height = hvec3(area_lights.data[idx].area_height);

	if (dot(area_width, area_width) < EPSILON || dot(area_height, area_height) < EPSILON) { // area is 0
		return;
	}

	if (dot(area_lights.data[idx].direction, vertex - area_lights.data[idx].position) <= 0) {
		return; // vertex is behind light
	}

	half a_len = length(area_width);
	half b_len = length(area_height);
	half a_half_len = a_len / half(2.0);
	half b_half_len = b_len / half(2.0);
	hvec3 light_center = hvec3(area_lights.data[idx].position);
	hvec3 light_to_vert = hvec3(vertex) - light_center;
	hvec3 area_a_dir = normalize(area_width);
	hvec3 area_b_dir = normalize(area_height);
	hvec3 area_direction = hvec3(area_lights.data[idx].direction);
	hvec3 pos_local_to_light = hvec3(dot(light_to_vert, area_a_dir), dot(light_to_vert, area_b_dir), dot(light_to_vert, -area_direction));

	hvec3 closest_point_local_to_light = hvec3(clamp(pos_local_to_light.x, -a_half_len, a_half_len), clamp(pos_local_to_light.y, -b_half_len, b_half_len), 0);
	half dist = length(closest_point_local_to_light - pos_local_to_light);

	half light_length = max(half(0.0), dist);
	half light_attenuation_raw = get_omni_attenuation(float(light_length), area_lights.data[idx].inv_radius, area_lights.data[idx].attenuation);
	half light_attenuation_ltc = light_attenuation_raw * half(light_length * light_length); // solid angle already decreases by inverse square, so attenuation power is 2.0 by default -> subtract 2.0
	half shadow = half(1.0);

#ifndef SHADOWS_DISABLED
	// Area light shadow.
	if (light_attenuation_raw > HALF_FLT_MIN && area_lights.data[idx].shadow_opacity > 0.001) {
		// there is a shadowmap
		vec2 texel_size = scene_data_block.data.shadow_atlas_pixel_size;
		vec4 base_uv_rect = area_lights.data[idx].atlas_rect;
		base_uv_rect.xy += texel_size;
		base_uv_rect.zw -= texel_size * 2.0;

		vec3 local_vert = (area_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;

		float shadow_len = length(local_vert); //need to remember shadow len from here
		vec3 shadow_dir = normalize(local_vert);

		vec3 local_normal = normalize(mat3(area_lights.data[idx].shadow_matrix) * vec3(normal));
		vec3 normal_bias = local_normal * area_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(local_normal, shadow_dir)));

		float inv_center_range = area_lights.data[idx].cone_attenuation;

		if (sc_use_light_soft_shadows() && area_lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker

			float blocker_count = 0.0;
			float blocker_average = 0.0;

			mat2 disk_rotation;
			{
				float r = quick_hash(gl_FragCoord.xy + vec2(taa_frame_count * 5.588238)) * 2.0 * M_PI;
				float sr = sin(r);
				float cr = cos(r);
				disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
			}

			vec3 basis_normal = shadow_dir;
			vec3 v0 = abs(basis_normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
			vec3 tangent = normalize(cross(v0, basis_normal));
			vec3 bitangent = normalize(cross(tangent, basis_normal));
			float z_norm = 1.0 - shadow_len * inv_center_range;

			tangent *= area_lights.data[idx].soft_shadow_size * area_lights.data[idx].soft_shadow_scale;
			bitangent *= area_lights.data[idx].soft_shadow_size * area_lights.data[idx].soft_shadow_scale;

			SPEC_CONSTANT_LOOP_ANNOTATION
			for (uint i = 0; i < sc_penumbra_shadow_samples(); i++) {
				vec2 disk = disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy;

				vec3 pos = local_vert + tangent * disk.x + bitangent * disk.y;

				pos = normalize(pos);

				vec4 uv_rect = base_uv_rect;

				pos.z = 1.0 + abs(pos.z);
				pos.xy /= pos.z;

				pos.xy = pos.xy * 0.5 + 0.5;
				pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;

				float d = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), pos.xy, 0.0).r;
				if (d > z_norm) {
					blocker_average += d;
					blocker_count += 1.0;
				}
			}

			if (blocker_count > 0.0) {
				//blockers found, do soft shadow
				blocker_average /= blocker_count;
				float penumbra = (-z_norm + blocker_average) / (1.0 - blocker_average);
				tangent *= penumbra;
				bitangent *= penumbra;

				z_norm += inv_center_range * area_lights.data[idx].shadow_bias;

				shadow = half(0.0);

				SPEC_CONSTANT_LOOP_ANNOTATION
				for (uint i = 0; i < sc_penumbra_shadow_samples(); i++) {
					vec2 disk = disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy;
					vec3 pos = local_vert + tangent * disk.x + bitangent * disk.y;

					pos = normalize(pos);
					pos = normalize(pos + normal_bias);

					vec4 uv_rect = base_uv_rect;

					pos.z = 1.0 + abs(pos.z);
					pos.xy /= pos.z;

					pos.xy = pos.xy * 0.5 + 0.5;
					pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;
					shadow += half(textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(pos.xy, z_norm, 1.0)));
				}

				shadow /= half(sc_penumbra_shadow_samples());
				shadow = mix(half(1.0), shadow, half(area_lights.data[idx].shadow_opacity));

			} else {
				//no blockers found, so no shadow
				shadow = half(1.0);
			}
		} else {
			vec4 uv_rect = base_uv_rect;

			vec3 shadow_sample = normalize(shadow_dir + normal_bias);

			shadow_sample.z = 1.0 + abs(shadow_sample.z);
			vec2 pos = shadow_sample.xy / shadow_sample.z;
			float depth = shadow_len - area_lights.data[idx].shadow_bias;
			depth *= inv_center_range;
			depth = 1.0 - depth;
			shadow = mix(half(1.0), sample_omni_pcf_shadow(shadow_atlas, area_lights.data[idx].soft_shadow_scale / shadow_sample.z, pos, uv_rect, vec2(0), depth, taa_frame_count), half(area_lights.data[idx].shadow_opacity));
		}
	}
#endif
	light_attenuation_ltc = light_attenuation_ltc * shadow;
	half light_attenuation = light_attenuation_raw * shadow;
	hvec3 color = hvec3(area_lights.data[idx].color);
	float max_mipmap = area_lights.data[idx].cone_angle;

	vec3 points[4];
	hvec3 h_area_width = area_width / half(2.0);
	hvec3 h_area_height = area_height / half(2.0);
	points[0] = area_lights.data[idx].position - h_area_width - h_area_height - vertex;
	points[1] = area_lights.data[idx].position + h_area_width - h_area_height - vertex;
	points[2] = area_lights.data[idx].position + h_area_width + h_area_height - vertex;
	points[3] = area_lights.data[idx].position - h_area_width + h_area_height - vertex;

	float ltc_diffuse = 0.0;
	vec3 ltc_diffuse_tex_color = vec3(1.0);
	float ltc_specular = 0.0;
	vec3 ltc_specular_tex_color = vec3(1.0);
	vec2 ltc_fresnel = vec2(0.0);
	hvec3 fresnel_color = hvec3(0.0);
	ltc_evaluate(vec3(normal), vec3(eye_vec), mat3(1), points, area_lights.data[idx].projector_rect, max_mipmap, area_light_atlas, SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP, ltc_diffuse, ltc_diffuse_tex_color);

#if !defined(SPECULAR_DISABLED) || (defined(LIGHT_CODE_USED) && defined(AREA_LIGHT_CODE_USED))
	ltc_evaluate_specular(vec3(normal), vec3(eye_vec), roughness, points, area_lights.data[idx].projector_rect, max_mipmap, area_light_atlas, SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP, SAMPLER_LINEAR_CLAMP, ltc_lut1, ltc_lut2, ltc_specular, ltc_fresnel, ltc_specular_tex_color);
	half f90 = clamp(dot(f0, hvec3(50.0 * 0.33)), metallic, half(1.0));
	fresnel_color = f0 * max(half(ltc_fresnel.x), half(0.0)) + (f90 - f0) * max(half(ltc_fresnel.y), half(0.0));
#endif

#if defined(LIGHT_CODE_USED) && defined(AREA_LIGHT_CODE_USED)
	// Light is written by the user shader.
	mat4 inv_view_matrix = transpose(mat4(scene_data_block.data.inv_view_matrix[0],
			scene_data_block.data.inv_view_matrix[1],
			scene_data_block.data.inv_view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));
	mat4 read_view_matrix = transpose(mat4(scene_data_block.data.view_matrix[0],
			scene_data_block.data.view_matrix[1],
			scene_data_block.data.view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

#ifdef USING_MOBILE_RENDERER
	uint instance_index = draw_call.instance_index;
#else
	uint instance_index = instance_index_interp;
#endif

	mat4 read_model_matrix = transpose(mat4(instances.data[instance_index].transform[0],
			instances.data[instance_index].transform[1],
			instances.data[instance_index].transform[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

#undef projection_matrix
#define projection_matrix scene_data_block.data.projection_matrix
#undef inv_projection_matrix
#define inv_projection_matrix scene_data_block.data.inv_projection_matrix

	vec2 read_viewport_size = scene_data_block.data.viewport_size;

#ifdef LIGHT_BACKLIGHT_USED
	vec3 backlight_highp = vec3(backlight);
#endif
	float roughness_highp = float(roughness);
	float metallic_highp = float(metallic);
	vec3 albedo_highp = vec3(albedo);
	float alpha_highp = float(alpha);
	vec3 normal_highp = vec3(normal);
	vec3 light_highp = (light_center - vertex) / light_length;
	vec3 view_highp = vec3(eye_vec);
	float specular_amount_highp = float(area_lights.data[idx].specular_amount);
	vec3 light_color_highp = vec3(color);
	float attenuation_highp = float(light_attenuation_ltc);
	vec3 diffuse_light_highp = vec3(diffuse_light);
	vec3 specular_light_highp = vec3(specular_light);
	bool is_directional = false;
	bool is_area = true;
	vec3 area_diffuse = ltc_diffuse * ltc_diffuse_tex_color;
	vec3 area_specular = ltc_specular * vec3(fresnel_color) * ltc_specular_tex_color;

#CODE : LIGHT

	alpha = half(alpha_highp);
	diffuse_light = hvec3(diffuse_light_highp);
	specular_light = hvec3(specular_light_highp);

#else
	half specular_amount = half(area_lights.data[idx].specular_amount);
	half area = a_len * b_len;
	half cc_attenuation = half(1.0);

#if defined(LIGHT_TRANSMITTANCE_USED) || defined(LIGHT_BACKLIGHT_USED) || defined(LIGHT_RIM_USED) || defined(DIFFUSE_TOON)
	hvec3 isotropic_light_color = hvec3(1.0); // independent of normal
	if (area_lights.data[idx].projector_rect != vec4(0.0)) {
		half lod = dist / sqrt(area);
		lod = log(half(2048.0) * lod) / log(half(3.0));

		hvec2 uv = (closest_point_local_to_light.xy + hvec2(a_half_len, b_half_len)) / hvec2(a_len, b_len);
		isotropic_light_color = hvec3(fetch_ltc_lod(vec2(hvec2(1.0) - uv), area_lights.data[idx].projector_rect, float(lod), max_mipmap, area_light_atlas, SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP));
	}
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
	{
		half transmittance_z = transmittance_depth;
		transmittance_color.a *= light_attenuation_raw; // not including shadows or ltc falloff
#ifndef SHADOWS_DISABLED
		if (area_lights.data[idx].shadow_opacity > 0.001) {
			// Redo shadowmapping, but shrink the model a bit to avoid artifacts.
			vec2 texel_size = scene_data_block.data.shadow_atlas_pixel_size;
			vec4 uv_rect = area_lights.data[idx].atlas_rect;
			uv_rect.xy += texel_size;
			uv_rect.zw -= texel_size * 2.0;
			vec3 local_vert = (area_lights.data[idx].shadow_matrix * vec4(vertex - vec3(normal) * area_lights.data[idx].transmittance_bias, 1.0)).xyz;

			float shadow_len = length(local_vert); //need to remember shadow len from here
			vec3 shadow_sample = normalize(local_vert);

			shadow_sample.z = 1.0 + abs(shadow_sample.z);
			vec2 pos = shadow_sample.xy / shadow_sample.z;
			float inv_center_range = area_lights.data[idx].cone_attenuation;
			float depth = shadow_len * inv_center_range;
			depth = 1.0 - depth;

			pos = pos * 0.5 + 0.5;
			pos = uv_rect.xy + pos * uv_rect.zw;
			float shadow_z = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), pos, 0.0).r;
			transmittance_z = half((depth - shadow_z) / inv_center_range);
		}
#endif
		// transmission can't use ltc_diffuse_tex_color, because for backface pixels texture would have to be sampled for opposite normal direction.
#ifdef SSS_MODE_SKIN
		diffuse_light += SSS_skin(half(ltc_diffuse), transmittance_depth, transmittance_z, transmittance_boost, transmittance_color, color * isotropic_light_color * area);
#else
		diffuse_light += SSS(half(ltc_diffuse), transmittance_depth, transmittance_z, transmittance_boost, transmittance_color, color * isotropic_light_color * area);
#endif
	}
#endif //LIGHT_TRANSMITTANCE_USED

#if defined(LIGHT_BACKLIGHT_USED) || defined(LIGHT_RIM_USED) || defined(DIFFUSE_TOON) || defined(SPECULAR_TOON)
	float solid_angle = quad_solid_angle(points);
#endif

#if defined(LIGHT_CLEARCOAT_USED)
	vec3 cc_specular_tex_color = vec3(1.0);
	float cc_specular_ltc = 0.0;
	vec2 cc_fresnel;
	ltc_evaluate_specular(vec3(vertex_normal), vec3(eye_vec), sqrt(mix(0.001, 0.1, float(clearcoat_roughness))), points, area_lights.data[idx].projector_rect, max_mipmap, area_light_atlas, SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP, SAMPLER_LINEAR_CLAMP, ltc_lut1, ltc_lut2, cc_specular_ltc, cc_fresnel, cc_specular_tex_color);
	half Fr = (half(0.04) * max(half(cc_fresnel.x), half(0.0)) + half(1.0 - 0.04) * max(half(cc_fresnel.y), half(0.0))) * clearcoat;
	cc_attenuation = half(1.0) - Fr;
	specular_light += half(cc_specular_ltc) * hvec3(cc_specular_tex_color) * Fr * color * light_attenuation_ltc * specular_amount;
#endif // LIGHT_CLEARCOAT_USED

	if (metallic < half(1.0)) {
#if defined(DIFFUSE_TOON)
		float backface_ltc_diffuse = 0.0;
		vec3 backface_ltc_tex_color_discard = vec3(1.0);
		ltc_evaluate(vec3(-normal), vec3(eye_vec), mat3(1), points, vec4(0.0), max_mipmap, area_light_atlas, SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP, backface_ltc_diffuse, backface_ltc_tex_color_discard);
		half NdotL = half((ltc_diffuse - backface_ltc_diffuse) / (max(solid_angle, 0.001) / M_PI));
		half diffuse_brdf_NL = smoothstep(-roughness, max(roughness, half(0.01)), NdotL) * half(1.0 / M_PI);
		diffuse_light += diffuse_brdf_NL * isotropic_light_color * color * area * light_attenuation * cc_attenuation;
#else
		diffuse_light += half(ltc_diffuse) * hvec3(ltc_diffuse_tex_color) * color * light_attenuation_ltc * cc_attenuation;
#endif // DIFFUSE_TOON

#if defined(LIGHT_BACKLIGHT_USED)
		// backlight can't use ltc_diffuse_tex_color, because for backface pixels texture would have to sampled for opposite normal direction.
		diffuse_light += color * max(half(solid_angle / M_PI) - half(ltc_diffuse), half(0.0)) * backlight * isotropic_light_color * light_attenuation_ltc;
#endif
	}

#if defined(LIGHT_RIM_USED) // same as for point lights
	half cNdotV = max(dot(normal, eye_vec), half(1e-4));
	half rim_light = pow(max(half(1e-4), half(1.0) - cNdotV), max(half(0.0), (half(1.0) - roughness) * half(16.0)));
	diffuse_light += rim_light * rim * mix(hvec3(1.0), albedo, rim_tint) * isotropic_light_color * color * half(solid_angle) * light_attenuation_ltc;
#endif

#if defined(SPECULAR_TOON)
	// If ltc_specular turns out to be not similar enough to RdotV, since its based on GGX, toon shading would need its own lookup-table.
	half mid = half(1.0) - roughness;
	mid *= mid;

	half RdotV = half(ltc_specular / (max(solid_angle, 0.001) / (M_PI)));
	half intensity = smoothstep(mid - roughness * half(0.5), mid + roughness * half(0.5), RdotV) * mid; // should we use specular tex color here?? or diffuse? or white?
	diffuse_light += intensity * hvec3(ltc_specular_tex_color) * color * area * light_attenuation * specular_amount; // write to diffuse_light, as in toon shading you generally want no reflection
#elif defined(SPECULAR_DISABLED)
	// do nothing
#else
	hvec3 spec = half(ltc_specular) * hvec3(ltc_specular_tex_color) * color * fresnel_color;
	specular_light += spec * specular_amount * light_attenuation_ltc * cc_attenuation;
#endif // SPECULAR_TOON

#endif // LIGHT_CODE_USED
}

#ifndef SHADOWS_DISABLED

#include "line_light_shadow_inc.glsl"

// GGX alpha below which the screen-space pass's specular visibility is not used.
#define LINE_SHADOW_PASS_MIN_ALPHA 0.02

#endif // !SHADOWS_DISABLED

// Finite zero-thickness line light. Diffuse uses the exact Lambertian line
// integral; specular uses the same integral under the LTC transform for GGX.
// Line lights have no projector and no texture.
void light_process_line(uint idx, vec3 vertex, hvec3 eye_vec, hvec3 normal, hvec3 f0, half roughness, half metallic, float taa_frame_count, hvec3 albedo, inout half alpha, vec2 screen_uv, hvec3 energy_compensation,
#ifdef LIGHT_BACKLIGHT_USED
		hvec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		hvec4 transmittance_color,
		half transmittance_depth,
		half transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		half rim, half rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		half clearcoat, half clearcoat_roughness, hvec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		hvec3 binormal, hvec3 tangent, half anisotropy,
#endif
		inout hvec3 diffuse_light, inout hvec3 specular_light) {
	vec3 segment = line_lights.data[idx].area_width;
	float length_sq = dot(segment, segment);
	if (length_sq < 1e-12) {
		return; // Zero-length segment emits nothing.
	}

	// Endpoints relative to the shading point.
	vec3 half_segment = segment * 0.5;
	vec3 light_center = line_lights.data[idx].position - vertex;
	vec3 p1 = light_center - half_segment;
	vec3 p2 = light_center + half_segment;

	// Distance to the closest point on the segment drives the range falloff.
	float t = clamp(dot(-p1, segment) / length_sq, 0.0, 1.0);
	vec3 closest_point = p1 + segment * t;
	float min_radius = max(line_lights.data[idx].size, 0.0);
	float dist_to_segment = max(length(closest_point), min_radius);

	half light_attenuation = get_omni_attenuation(dist_to_segment, line_lights.data[idx].inv_radius, line_lights.data[idx].attenuation);
	// The line integral already falls off with the inverse square of distance, so
	// cancel it here and keep only the artistic range window, as area lights do.
	// This is the value that scales the result, so it is also what decides whether
	// there is anything to do; `light_attenuation` alone would cut the light off at
	// a fixed distance regardless of its range.
	float attenuation_ltc = float(light_attenuation) * dist_to_segment * dist_to_segment;
	if (attenuation_ltc < 1e-5) {
		return;
	}
	half light_attenuation_ltc = half(min(attenuation_ltc, 65504.0)); // Keep in fp16 range.

	hvec3 color = hvec3(line_lights.data[idx].color);

#ifdef LINE_LIGHT_REFERENCE
	// 16-point Gauss-Legendre quadrature of the real BRDF along the segment.
	// Nodes and weights are symmetric, so only the positive half is stored.
	const float quad_nodes[8] = float[](0.0950125098, 0.2816035508, 0.4580167777, 0.6178762444, 0.7554044084, 0.8656312024, 0.9445750231, 0.9894009350);
	const float quad_weights[8] = float[](0.1894506105, 0.1826034150, 0.1691565194, 0.1495959888, 0.1246289713, 0.0951585117, 0.0622535239, 0.0271524594);

	float seg_length = sqrt(length_sq);
	vec3 tangent_dir = segment / seg_length;
	float half_len = 0.5 * seg_length;
	float min_radius_sq = min_radius * min_radius;

#ifndef SHADOWS_DISABLED
	// Visibility per quadrature node, from the same cell mask.
	float ref_opacity = line_lights.data[idx].shadow_opacity;
	bool ref_shadowed = ref_opacity > 0.001;
	LineShadowCells ref_cells;
	uvec2 ref_mask = uvec2(0u);
	if (ref_shadowed) {
		LineShadowContext ref_ctx = line_shadow_begin(idx, vertex, vec3(normal), taa_frame_count, gl_FragCoord.xy, scene_data_block.data.shadow_atlas_pixel_size);
		ref_cells = line_shadow_cells(ref_ctx.receiver.z, max(length(ref_ctx.receiver.xy), LINE_LIGHT_MIN_DISTANCE), half_len, ref_ctx.jitter);
		uvec2 unused;
		line_shadow_mask(ref_ctx, ref_cells, ref_cells, true, false, ref_mask, unused);
	}
#endif

	for (int node_i = 0; node_i < 8; node_i++) {
		for (int node_side = 0; node_side < 2; node_side++) {
			float node = node_side == 0 ? -quad_nodes[node_i] : quad_nodes[node_i];
			vec3 sample_vec = light_center + tangent_dir * (node * half_len);
			float dist_sq = max(dot(sample_vec, sample_vec), min_radius_sq);
			vec3 sample_dir = sample_vec * inversesqrt(dist_sq);
			// sin(phi) is the cylinder's projected width towards this sample.
			float sin_phi = length(cross(sample_dir, tangent_dir));
			half sample_weight = half(quad_weights[node_i] * half_len * sin_phi / dist_sq);
#ifndef SHADOWS_DISABLED
			if (ref_shadowed) {
				uint cell = uint(clamp(floor(line_shadow_cell_of(ref_cells, node * half_len) + ref_cells.jitter), 0.0, float(LINE_SHADOW_CELLS - 1u)));
				if (line_mask_test(ref_mask, cell)) {
					sample_weight *= half(1.0 - ref_opacity);
				}
			}
#endif

			light_compute(normal, hvec3(sample_dir), eye_vec, half(0.0), color, false, sample_weight * light_attenuation_ltc, f0, roughness, metallic, half(line_lights.data[idx].specular_amount), albedo, alpha, screen_uv, energy_compensation,
#ifdef LIGHT_BACKLIGHT_USED
					backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
					transmittance_color,
					transmittance_depth,
					transmittance_boost,
					half(0.0),
#endif
#ifdef LIGHT_RIM_USED
					rim, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
					clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
					binormal, tangent, anisotropy,
#endif
					diffuse_light,
					specular_light);
		}
	}
	return;
#endif // LINE_LIGHT_REFERENCE

	float ltc_diffuse = line_integrate_diffuse(p1, p2, vec3(normal), min_radius);

	float ltc_specular = 0.0;
	vec2 ltc_fresnel = vec2(0.0);
	hvec3 fresnel_color = hvec3(0.0);
	// Zero means "there is no specular lobe to estimate visibility for".
	mat3 cos_xform = mat3(0.0);
	float ltc_min_radius = 0.0;
#if !defined(SPECULAR_DISABLED) || (defined(LIGHT_CODE_USED) && defined(AREA_LIGHT_CODE_USED))
#ifdef LIGHT_ANISOTROPY_USED
	cos_xform = ltc_line_cos_xform_anisotropic(vec3(normal), vec3(tangent), vec3(binormal), vec3(eye_vec), float(roughness), float(anisotropy), SAMPLER_LINEAR_CLAMP, ltc_lut2, ltc_lut_aniso, ltc_fresnel);
#else
	cos_xform = ltc_line_cos_xform(vec3(normal), vec3(eye_vec), ltc_matrix(vec3(normal), vec3(eye_vec), float(roughness), SAMPLER_LINEAR_CLAMP, ltc_lut1, ltc_lut2, ltc_fresnel));
#endif
	// Needed by the integral and by the shadow sampler's regularization.
	float ltc_w = ltc_line_width_factor(cos_xform, p1, p2);
	ltc_specular = ltc_evaluate_line_transformed(cos_xform, p1, p2, min_radius, ltc_w);
	ltc_min_radius = ltc_w * min_radius;

	half f90 = clamp(dot(f0, hvec3(50.0 * 0.33)), metallic, half(1.0));
	fresnel_color = f0 * max(half(ltc_fresnel.x), half(0.0)) + (f90 - f0) * max(half(ltc_fresnel.y), half(0.0));
#endif

	// Per-lobe visibility; exactly 1.0 when unshadowed.
	half vis_diffuse = half(1.0);
	half vis_specular = half(1.0);
#ifndef SHADOWS_DISABLED
	bool shadow_walk = line_lights.data[idx].shadow_opacity > 0.001;
	bool walk_diffuse = true;
// The pass shaded the prepass depth: not the surface of materials that move it.
#if defined(USE_LINE_SHADOW_PASS) && !defined(LIGHT_VERTEX_USED) && !defined(DEPTH_USED) && !defined(Z_CLIP_SCALE_USED)
	// Screen-space layer + 1, negative without contact shadows.
	int layer = abs(int(line_lights.data[idx].cone_angle));
	bool pass_match = line_shadow_pass_match;
	if (pass_match && bool(implementation_data.ss_effects_flags & SCREEN_SPACE_EFFECTS_FLAGS_LINE_SHADOW_PASS_CHECK_SURFACE)) {
		// Where surfaces meet their depths agree, but not their normals.
#ifdef USE_MULTIVIEW
		vec3 pass_normal = texelFetch(sampler2DArray(normal_roughness_buffer, SAMPLER_NEAREST_CLAMP), ivec3(gl_FragCoord.xy, ViewIndex), 0).xyz;
#else
		vec3 pass_normal = texelFetch(sampler2D(normal_roughness_buffer, SAMPLER_NEAREST_CLAMP), ivec2(gl_FragCoord.xy), 0).xyz;
#endif
		pass_match = dot(normalize(pass_normal * 2.0 - 1.0), vec3(normal)) > 0.9;
	}
	if (shadow_walk && layer > 0 && pass_match && bool(implementation_data.ss_effects_flags & SCREEN_SPACE_EFFECTS_FLAGS_USE_LINE_SHADOW_PASS)) {
#ifdef USE_MULTIVIEW
		layer = (layer - 1) * 2 + int(ViewIndex);
#else
		layer -= 1;
#endif
		vec2 vis = texelFetch(sampler2DArray(line_shadow_pass_buffer, SAMPLER_NEAREST_CLAMP), ivec3(gl_FragCoord.xy, layer), 0).rg;
		vis_diffuse = half(vis.x);
		vis_specular = half(vis.y);
		// The G-buffer normal is too coarse for very narrow lobes.
		walk_diffuse = false;
#ifdef LIGHT_ANISOTROPY_USED
		shadow_walk = true; // The pass places an isotropic lobe.
#else
		shadow_walk = float(roughness) * float(roughness) < LINE_SHADOW_PASS_MIN_ALPHA;
#endif
	}
#endif
	if (shadow_walk) {
		LineShadowContext ctx = line_shadow_begin(idx, vertex, vec3(normal), taa_frame_count, gl_FragCoord.xy, scene_data_block.data.shadow_atlas_pixel_size);
		uvec4 contact = uvec4(0u);
#ifdef USE_LINE_CONTACT_SHADOWS
		int contact_layer = int(line_lights.data[idx].cone_angle);
		if (contact_layer > 0 && bool(implementation_data.ss_effects_flags & SCREEN_SPACE_EFFECTS_FLAGS_USE_LINE_SSCS)) {
#ifdef USE_MULTIVIEW
			contact_layer = (contact_layer - 1) * 2 + int(ViewIndex);
#else
			contact_layer -= 1;
#endif
			uint hits = uint(texelFetch(sampler2DArray(line_contact_shadows, SAMPLER_NEAREST_CLAMP), ivec3(gl_FragCoord.xy, contact_layer), 0).r * 65535.0 + 0.5);
			contact = line_shadow_contact_cells(hits, ctx.jitter);
		}
#endif
		float seg_length = sqrt(length_sq);
		// Same regularized geometry as line_integrate_diffuse().
		vec3 wt = segment / seg_length;
		float l1 = dot(p1, wt);
		vec3 po = p1 - l1 * wt;
		float d0 = length(po);
		float d = max(d0, max(min_radius, LINE_LIGHT_MIN_DISTANCE));

		half walk_diffuse_vis;
		line_shadow_visibility(idx, ctx, vec3(normal), vec3(eye_vec), float(roughness) * float(roughness), po * (d / max(d0, 1e-9)), wt,
				d, l1, l1 + seg_length, l1 + 0.5 * seg_length, cos_xform, ltc_min_radius, contact, walk_diffuse,
				walk_diffuse_vis, vis_specular);
		if (walk_diffuse) {
			vis_diffuse = walk_diffuse_vis;
		}
	}
#endif // !SHADOWS_DISABLED

#if defined(LIGHT_CODE_USED) && defined(AREA_LIGHT_CODE_USED)
	// Light is written by the user shader. Line lights reuse the area light
	// contract: the integral is done here, the shader only modulates it.
	mat4 inv_view_matrix = transpose(mat4(scene_data_block.data.inv_view_matrix[0],
			scene_data_block.data.inv_view_matrix[1],
			scene_data_block.data.inv_view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));
	mat4 read_view_matrix = transpose(mat4(scene_data_block.data.view_matrix[0],
			scene_data_block.data.view_matrix[1],
			scene_data_block.data.view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

#ifdef USING_MOBILE_RENDERER
	uint instance_index = draw_call.instance_index;
#else
	uint instance_index = instance_index_interp;
#endif

	mat4 read_model_matrix = transpose(mat4(instances.data[instance_index].transform[0],
			instances.data[instance_index].transform[1],
			instances.data[instance_index].transform[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

#undef projection_matrix
#define projection_matrix scene_data_block.data.projection_matrix
#undef inv_projection_matrix
#define inv_projection_matrix scene_data_block.data.inv_projection_matrix

	vec2 read_viewport_size = scene_data_block.data.viewport_size;

#ifdef LIGHT_BACKLIGHT_USED
	vec3 backlight_highp = vec3(backlight);
#endif
	float roughness_highp = float(roughness);
	float metallic_highp = float(metallic);
	vec3 albedo_highp = vec3(albedo);
	float alpha_highp = float(alpha);
	vec3 normal_highp = vec3(normal);
	vec3 light_highp = closest_point / max(length(closest_point), 1e-6);
	vec3 view_highp = vec3(eye_vec);
	float specular_amount_highp = float(line_lights.data[idx].specular_amount);
	vec3 light_color_highp = vec3(color);
	float attenuation_highp = float(light_attenuation_ltc);
	vec3 diffuse_light_highp = vec3(diffuse_light);
	vec3 specular_light_highp = vec3(specular_light);
	bool is_directional = false;
	bool is_area = true;
	vec3 area_diffuse = vec3(ltc_diffuse * float(vis_diffuse));
	vec3 area_specular = ltc_specular * float(vis_specular) * vec3(fresnel_color);

#CODE : LIGHT

	alpha = half(alpha_highp);
	diffuse_light = hvec3(diffuse_light_highp);
	specular_light = hvec3(specular_light_highp);

#else
	half specular_amount = half(line_lights.data[idx].specular_amount);
	half segment_length = half(sqrt(length_sq));
	half cc_attenuation = half(1.0);

#if defined(LIGHT_BACKLIGHT_USED) || defined(LIGHT_RIM_USED) || defined(DIFFUSE_TOON) || defined(SPECULAR_TOON)
	// Unclamped measure, the line counterpart of the quad's solid angle.
	half measure = half(line_measure(p1, p2, min_radius));
#endif

#ifdef LIGHT_TRANSMITTANCE_USED
	{
		// Without shadows there is no blocker depth, so this degenerates the same
		// way a shadowless area light does.
		transmittance_color.a *= light_attenuation;
#ifdef SSS_MODE_SKIN
		diffuse_light += SSS_skin(half(ltc_diffuse), transmittance_depth, transmittance_depth, transmittance_boost, transmittance_color, color * segment_length);
#else
		diffuse_light += SSS(half(ltc_diffuse), transmittance_depth, transmittance_depth, transmittance_boost, transmittance_color, color * segment_length);
#endif
	}
#endif // LIGHT_TRANSMITTANCE_USED

#if defined(LIGHT_CLEARCOAT_USED)
	{
		float cc_specular_ltc = 0.0;
		vec2 cc_fresnel;
		ltc_evaluate_line_specular(vec3(vertex_normal), vec3(eye_vec), sqrt(mix(0.001, 0.1, float(clearcoat_roughness))), p1, p2, min_radius, SAMPLER_LINEAR_CLAMP, ltc_lut1, ltc_lut2, cc_specular_ltc, cc_fresnel);
		half Fr = (half(0.04) * max(half(cc_fresnel.x), half(0.0)) + half(1.0 - 0.04) * max(half(cc_fresnel.y), half(0.0))) * clearcoat;
		// The coat/base energy split is a material property, so only the light
		// reaching the coat is shadowed, not `Fr`.
		cc_attenuation = half(1.0) - Fr;
		specular_light += half(cc_specular_ltc) * vis_specular * Fr * color * light_attenuation_ltc * specular_amount;
	}
#endif // LIGHT_CLEARCOAT_USED

	if (metallic < half(1.0)) {
#if defined(DIFFUSE_TOON)
		half NdotL = half(ltc_diffuse - line_integrate_diffuse(p1, p2, vec3(-normal), min_radius)) / max(measure / half(M_PI), half(0.001));
		half diffuse_brdf_NL = smoothstep(-roughness, max(roughness, half(0.01)), NdotL) * half(1.0 / M_PI);
		diffuse_light += diffuse_brdf_NL * vis_diffuse * color * segment_length * light_attenuation * cc_attenuation;
#else
		diffuse_light += half(ltc_diffuse) * vis_diffuse * color * light_attenuation_ltc * cc_attenuation;
#endif // DIFFUSE_TOON

#if defined(LIGHT_BACKLIGHT_USED)
		diffuse_light += color * max(measure / half(M_PI) - half(ltc_diffuse), half(0.0)) * vis_diffuse * backlight * light_attenuation_ltc;
#endif
	}

#if defined(LIGHT_RIM_USED) // same as for point lights
	half cNdotV = max(dot(normal, eye_vec), half(1e-4));
	half rim_light = pow(max(half(1e-4), half(1.0) - cNdotV), max(half(0.0), (half(1.0) - roughness) * half(16.0)));
	diffuse_light += rim_light * rim * mix(hvec3(1.0), albedo, rim_tint) * color * measure * vis_diffuse * light_attenuation_ltc;
#endif

#if defined(SPECULAR_TOON)
	half mid = half(1.0) - roughness;
	mid *= mid;

	half RdotV = half(ltc_specular) / max(measure / half(M_PI), half(0.001));
	half intensity = smoothstep(mid - roughness * half(0.5), mid + roughness * half(0.5), RdotV) * mid;
	diffuse_light += intensity * vis_specular * color * segment_length * light_attenuation * specular_amount; // write to diffuse_light, as in toon shading you generally want no reflection
#elif defined(SPECULAR_DISABLED)
	// do nothing
#else
	specular_light += half(ltc_specular) * vis_specular * color * fresnel_color * specular_amount * light_attenuation_ltc * cc_attenuation * energy_compensation;
#endif // SPECULAR_TOON

#endif // LIGHT_CODE_USED
}

void reflection_process(uint ref_index, vec3 vertex, hvec3 ref_vec, hvec3 normal, half roughness, hvec3 ambient_light,
#ifdef LIGHT_CLEARCOAT_USED
		hvec3 cc_ref_vec, half cc_roughness, inout hvec4 cc_reflection_accum,
#endif
		inout hvec4 ambient_accum, inout hvec4 reflection_accum) {
	vec3 box_extents = reflections.data[ref_index].box_extents;
	vec3 local_pos = (reflections.data[ref_index].local_matrix * vec4(vertex, 1.0)).xyz;

	if (any(greaterThan(abs(local_pos), box_extents))) { //out of the reflection box
		return;
	}

	half blend = half(1.0);
	if (reflections.data[ref_index].blend_distance != 0.0) {
		vec3 axis_blend_distance = min(vec3(reflections.data[ref_index].blend_distance), box_extents);
		vec3 blend_axes_highp = abs(local_pos) - box_extents + axis_blend_distance;
		hvec3 blend_axes = hvec3(blend_axes_highp / axis_blend_distance);
		blend_axes = clamp(half(1.0) - blend_axes, hvec3(0.0), hvec3(1.0));
		blend = pow(blend_axes.x * blend_axes.y * blend_axes.z, half(2.0));
	}

	vec2 border_size = scene_data_block.data.reflection_atlas_border_size;
	if (reflections.data[ref_index].intensity > 0.0 && reflection_accum.a < half(1.0)) { // compute reflection

		vec3 local_ref_vec = (reflections.data[ref_index].local_matrix * vec4(ref_vec, 0.0)).xyz;

		if (reflections.data[ref_index].box_project) { //box project

			vec3 nrdir = normalize(local_ref_vec);
			vec3 rbmax = (box_extents - local_pos) / nrdir;
			vec3 rbmin = (-box_extents - local_pos) / nrdir;

			vec3 rbminmax = mix(rbmin, rbmax, greaterThan(nrdir, vec3(0.0, 0.0, 0.0)));

			float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			vec3 posonbox = local_pos + nrdir * fa;
			local_ref_vec = posonbox - reflections.data[ref_index].box_offset;
		}

		hvec4 reflection;
		half reflection_blend = max(half(0.0), blend - reflection_accum.a);

		float roughness_lod = sqrt(roughness) * MAX_ROUGHNESS_LOD;
		vec2 reflection_uv = vec3_to_oct_with_border(local_ref_vec, border_size);
		reflection.rgb = hvec3(textureLod(sampler2DArray(reflection_atlas, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec3(reflection_uv, reflections.data[ref_index].index), roughness_lod).rgb) * REFLECTION_MULTIPLIER;
		reflection.rgb *= half(reflections.data[ref_index].exposure_normalization);
		reflection.a = reflection_blend;

		reflection.rgb *= half(reflections.data[ref_index].intensity);
		reflection.rgb *= reflection.a;

		reflection_accum += reflection;
	}

#ifdef LIGHT_CLEARCOAT_USED
	if (reflections.data[ref_index].intensity > 0.0 && cc_reflection_accum.a < half(1.0)) { // compute clearcoat reflection
		vec3 local_cc_ref_vec = (reflections.data[ref_index].local_matrix * vec4(cc_ref_vec, 0.0)).xyz;

		if (reflections.data[ref_index].box_project) { // Box project.

			vec3 nrdir = normalize(local_cc_ref_vec);
			vec3 rbmax = (box_extents - local_pos) / nrdir;
			vec3 rbmin = (-box_extents - local_pos) / nrdir;

			vec3 rbminmax = mix(rbmin, rbmax, greaterThan(nrdir, vec3(0.0, 0.0, 0.0)));

			float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			vec3 posonbox = local_pos + nrdir * fa;
			local_cc_ref_vec = posonbox - reflections.data[ref_index].box_offset;
		}

		hvec4 cc_reflection;
		half cc_reflection_blend = max(half(0.0), blend - cc_reflection_accum.a);

		float cc_roughness_lod = sqrt(cc_roughness) * MAX_ROUGHNESS_LOD;
		vec2 cc_reflection_uv = vec3_to_oct_with_border(local_cc_ref_vec, border_size);
		cc_reflection.rgb = hvec3(textureLod(sampler2DArray(reflection_atlas, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec3(cc_reflection_uv, reflections.data[ref_index].index), cc_roughness_lod).rgb) * REFLECTION_MULTIPLIER;
		cc_reflection.rgb *= half(reflections.data[ref_index].exposure_normalization);
		cc_reflection.a = cc_reflection_blend;

		cc_reflection.rgb *= half(reflections.data[ref_index].intensity);
		cc_reflection.rgb *= cc_reflection.a;

		cc_reflection_accum += cc_reflection;
	}
#endif // LIGHT_CLEARCOAT_USED

	if (ambient_accum.a >= half(1.0)) {
		return;
	}

	switch (reflections.data[ref_index].ambient_mode) {
		case REFLECTION_AMBIENT_DISABLED: {
			//do nothing
		} break;
		case REFLECTION_AMBIENT_ENVIRONMENT: {
			vec3 local_amb_vec = (reflections.data[ref_index].local_matrix * vec4(normal, 0.0)).xyz;
			hvec4 ambient_out;
			half ambient_blend = max(half(0.0), blend - ambient_accum.a);

			float roughness_lod = MAX_ROUGHNESS_LOD;
			vec2 ambient_uv = vec3_to_oct_with_border(local_amb_vec, border_size);
			ambient_out.rgb = hvec3(textureLod(sampler2DArray(reflection_atlas, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec3(ambient_uv, reflections.data[ref_index].index), roughness_lod).rgb);
			ambient_out.rgb *= half(reflections.data[ref_index].exposure_normalization);
			ambient_out.a = ambient_blend;
			ambient_out.rgb *= ambient_out.a;
			ambient_accum += ambient_out;
		} break;
		case REFLECTION_AMBIENT_COLOR: {
			hvec4 ambient_out;
			half ambient_blend = max(half(0.0), blend - ambient_accum.a);

			ambient_out.rgb = hvec3(reflections.data[ref_index].ambient);
			ambient_out.a = ambient_blend;
			ambient_out.rgb *= ambient_out.a;
			ambient_accum += ambient_out;
		} break;
	}
}

half blur_shadow(half shadow) {
	return shadow;
#if 0
	//disabling for now, will investigate later
	float interp_shadow = shadow;
	if (gl_HelperInvocation) {
		interp_shadow = -4.0; // technically anything below -4 will do but just to make sure
	}

	uvec2 fc2 = uvec2(gl_FragCoord.xy);
	interp_shadow -= dFdx(interp_shadow) * (float(fc2.x & 1) - 0.5);
	interp_shadow -= dFdy(interp_shadow) * (float(fc2.y & 1) - 0.5);

	if (interp_shadow >= 0.0) {
		shadow = interp_shadow;
	}
	return shadow;
#endif
}
