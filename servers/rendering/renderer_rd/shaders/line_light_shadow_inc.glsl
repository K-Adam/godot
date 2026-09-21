// Line light shadows: each viewpoint's polar map stores the plane through the receiver
// and the line as one row. Surfaces found there inside the receiver-segment triangle
// are projected from the receiver onto the segment and OR-ed into a mask of cells.
// The includer declares `line_lights`, `shadow_atlas`, `line_shadow_pyramid`,
// SAMPLER_NEAREST_CLAMP, M_PI and the half types.

// Texels of depth error still counted as the receiver's own surface.
#define LINE_SHADOW_PLANE_TEXELS 1.5
// Neighbouring depths within this many texels are joined into one surface.
#define LINE_SHADOW_CONTINUITY 6.0

struct LineShadowContext {
	vec3 receiver; // Shadow-local: the segment runs along +Z, centered on the origin.
	vec3 normal;
	float z_far;
	vec2 texel_size; // One atlas texel in atlas UV.
	float jitter;
	float row_jitter;
	float stride_jitter;
	uint section_base; // First per-viewpoint record in the line light array.
	uint sections; // How many viewpoints.
	float seg_length;
};

// Interleaved gradient noise, as quick_hash() in the forward pass.
float line_shadow_hash(vec2 pos) {
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	return fract(magic.z * fract(dot(pos, magic.xy)));
}

// `pixel` is the pixel center, as gl_FragCoord.xy.
LineShadowContext line_shadow_begin(uint idx, vec3 vertex, vec3 normal, float taa_frame_count, vec2 pixel, vec2 atlas_texel_size) {
	LineShadowContext ctx;
	ctx.texel_size = atlas_texel_size;
	ctx.section_base = uint(line_lights.data[idx].pad[0]);
	ctx.sections = max(uint(line_lights.data[idx].pad[1]), 1u);
	ctx.seg_length = length(line_lights.data[idx].area_width);
	ctx.z_far = 1.0 / max(line_lights.data[idx].cone_attenuation, 1e-9);
	ctx.receiver = (line_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;
	ctx.normal = normalize(mat3(line_lights.data[idx].shadow_matrix) * normal);
	// The normal bias is an angle, so it scales with distance. Lifting the receiver off
	// its surface also lets the walk skip that surface in whole blocks.
	float dist = length(vec2(length(ctx.receiver.xy), max(abs(ctx.receiver.z) - 0.5 * ctx.seg_length, 0.0)));
	ctx.receiver += ctx.normal * (line_lights.data[idx].shadow_normal_bias * dist);
	ctx.jitter = line_shadow_hash(pixel + vec2(taa_frame_count * 5.588238));
	ctx.row_jitter = line_shadow_hash(pixel.yx + vec2(17.0 + taa_frame_count * 3.371));
	// R2 sequence: a lattice unrelated to the two above.
	ctx.stride_jitter = fract(dot(pixel, vec2(0.7548777, 0.5698403)) + taa_frame_count * 0.618034);
	return ctx;
}

#include "line_light_cells_inc.glsl"

// Cells hidden per quarter from packed contact hits: 4 bits per quarter, 15 is all.
uvec4 line_shadow_contact_cells(uint hits, float jitter) {
	uvec4 cells;
	for (int k = 0; k < 4; k++) {
		cells[k] = uint(min(int(float((hits >> uint(4 * k)) & 15u) * (float(LINE_SHADOW_CELLS / 4u) / 15.0) + jitter), int(LINE_SHADOW_CELLS / 4u)));
	}
	return cells;
}

// Marks the cells the surface piece from `a` to `b` hides. Slice coordinates relative to
// the viewpoint: x from the line, y along it. `ha`, `hb` are heights above the
// receiver's plane; `rv` is the receiver's slice position and the viewpoint offset.
void _line_shadow_segment(vec2 a, vec2 b, float ha, float hb, vec3 rv, LineShadowCells cd, LineShadowCells cs, inout uvec2 md, inout uvec2 ms) {
	// Anything below the receiver's plane only hides directions below its horizon.
	if (max(ha, hb) <= 0.0) {
		return;
	}
	float t0 = 0.0;
	float t1 = 1.0;
	if (ha < 0.0) {
		t0 = ha / (ha - hb);
	} else if (hb < 0.0) {
		t1 = ha / (ha - hb);
	}
	// Only what is between the line and the receiver.
	float ga = 0.999 * rv.x - a.x;
	float gb = 0.999 * rv.x - b.x;
	if (max(ga, gb) <= 0.0) {
		return;
	}
	if (ga < 0.0) {
		t0 = max(t0, ga / (ga - gb));
	} else if (gb < 0.0) {
		t1 = min(t1, ga / (ga - gb));
	}
	if (t1 <= t0) {
		return;
	}
	vec2 p0 = mix(a, b, t0);
	vec2 p1 = mix(a, b, t1);
	// Project from the receiver onto the line.
	float e0 = rv.y + (p0.y - rv.y) * rv.x / (rv.x - p0.x) + rv.z;
	float e1 = rv.y + (p1.y - rv.y) * rv.x / (rv.x - p1.x) + rv.z;
	float u0 = min(e0, e1);
	float u1 = max(e0, e1);
	md |= line_shadow_cell_bits(cd, u0, u1);
	ms |= line_shadow_cell_bits(cs, u0, u1);
}

// Distance along the ray at `theta` from the viewpoint to where it leaves the triangle
// between receiver `r` and the axis from `a` to `b` (viewpoint on [a, b]).
float _line_shadow_exit(vec2 r, float a, float b, float theta_r, float theta) {
	float e = theta < theta_r ? b : a;
	return e * r.x / (r.x * cos(theta) - (r.y - e) * sin(theta));
}

// Adds to `md`/`ms` the cells of `own_d`/`own_s` hidden in viewpoint `view`'s map.
void _line_shadow_walk(LineShadowContext ctx, uint view, float v, LineShadowCells cd, LineShadowCells cs, uvec2 own_d, uvec2 own_s, inout uvec2 md, inout uvec2 ms) {
	uint record = ctx.section_base + view;
	vec4 rect = line_lights.data[record].atlas_rect;
	ivec2 origin = ivec2(round(rect.xy / ctx.texel_size));
	int size = int(round(rect.z / ctx.texel_size.x));
	float delta = M_PI / float(size);

	vec3 r = ctx.receiver - vec3(0.0, 0.0, v);
	float rho_r = length(r.xy);
	if (rho_r < 1e-5) {
		return;
	}

	// One of the two rows either side of the receiver's azimuth, picked stochastically.
	float ring_f = atan(r.y, r.x) / delta - 0.5;
	float ring_lo = floor(ring_f);
	int ring = (int(ring_lo) + (ring_f - ring_lo > ctx.row_jitter ? 1 : 0) + 4 * size) % (2 * size);
	float phi = (float(ring) + 0.5) * delta;
	origin.y += ring % size;
	if (ring >= size) {
		origin += ivec2(round(line_lights.data[record].direction.xy / ctx.texel_size));
	}

	vec2 n2 = vec2(dot(vec2(cos(phi), sin(phi)), ctx.normal.xy), ctx.normal.z);
	float h0 = dot(r, ctx.normal);
	vec3 rv = vec3(rho_r, r.z, v);
	float half_tan = tan(0.5 * delta);
	float tol_scale = LINE_SHADOW_PLANE_TEXELS * delta;
	float cont_scale = LINE_SHADOW_CONTINUITY * delta;

	// Triangle to the still undecided cells, widened to the viewpoint. Blocks whose
	// nearest depth lies beyond it are skipped.
	float ua = 1e9;
	float ub = -1e9;
	if (own_d != uvec2(0u)) {
		ua = line_shadow_cell_u(cd, float(line_mask_lsb(own_d)));
		ub = line_shadow_cell_u(cd, float(line_mask_msb(own_d)));
	}
	if (own_s != uvec2(0u)) {
		ua = min(ua, line_shadow_cell_u(cs, float(line_mask_lsb(own_s))));
		ub = max(ub, line_shadow_cell_u(cs, float(line_mask_msb(own_s))));
	}
	float tri_a = min(ua - v, 0.0);
	float tri_b = max(ub - v, 0.0);
	vec2 rs = rv.xy;
	float r_len = length(rs);
	float theta_r = atan(rho_r, r.z);
	// Even pyramid levels span 4 to 1024 texels; none may span two slots.
	int top = size >= 16 ? min(findMSB(size) - 2, 8) & ~1 : -2;

	// Each step tests a block or reads a texel, which keeps divergence low.
	uvec2 hd = uvec2(0u);
	uvec2 hs = uvec2(0u);
	bool prev_ok = false;
	vec2 prev_x = vec2(0.0);
	float prev_h = 0.0;
	float prev_depth = 0.0;
	int i = 0;
	int level = top;
	// Leaf runs include one neighbour either side, so gaps to them are filled.
	int leaf_start = 0;
	int leaf_own_end = top < 0 ? size : 0;
	int leaf_end = top < 0 ? size + 1 : 0;
	[[dont_unroll]] while (i < size) {
		if (i >= leaf_end) {
			int block = 4 << level;
			float nearest = (1.0 - texelFetch(sampler2D(line_shadow_pyramid, SAMPLER_NEAREST_CLAMP), (origin + ivec2(i, 0)) >> (level + 2), level).r) * ctx.z_far;
			// Widened by a texel for patches. The exit distance is convex along each
			// edge, so its maximum is at an end or at the receiver.
			float t0 = max(float(i - 1) * delta, 0.0);
			float t1 = min(float(i + block + 1) * delta, M_PI);
			float reach = max(_line_shadow_exit(rs, tri_a, tri_b, theta_r, t0), _line_shadow_exit(rs, tri_a, tri_b, theta_r, t1));
			if (t0 <= theta_r && theta_r <= t1) {
				reach = max(reach, r_len);
			}
			// Chords between neighbouring texels dip in by up to 1 - cos(delta / 2).
			if (nearest * cos(0.5 * delta) > 1.001 * reach) {
				i += block;
				prev_ok = false;
				// Back up to the coarsest level this block boundary starts.
				while (level < top && (i & ((16 << level) - 1)) == 0) {
					level += 2;
				}
			} else if (level > 0) {
				level -= 2;
			} else {
				leaf_start = i;
				leaf_own_end = i + 4;
				leaf_end = i + 5;
				i = max(i - 1, 0);
				prev_ok = false;
			}
			continue;
		}

		float stored = texelFetch(sampler2D(shadow_atlas, SAMPLER_NEAREST_CLAMP), origin + ivec2(i, 0), 0).r;
		if (stored > 0.0) {
			float theta = (float(i) + 0.5) * delta;
			vec2 dir = vec2(sin(theta), cos(theta));
			float depth = (1.0 - stored) * ctx.z_far;
			vec2 x = depth * dir;
			float nd = dot(dir, n2);
			float h = dot(x, n2) - h0 - tol_scale * depth * sqrt(max(1.0 - nd * nd, 0.0));
			// The texel's own patch, then the gap to its neighbour if joined.
			vec2 t = vec2(dir.y, -dir.x) * (depth * half_tan);
			float ht = dot(t, n2);
			bool joined = prev_ok && abs(depth - prev_depth) <= cont_scale * min(depth, prev_depth);
			bool own = i >= leaf_start && i < leaf_own_end;
			[[dont_unroll]] for (int e = own ? 0 : 1; e < (joined ? 2 : 1); e++) {
				_line_shadow_segment(e == 0 ? x - t : prev_x, e == 0 ? x + t : x, e == 0 ? h - ht : prev_h, e == 0 ? h + ht : h, rv, cd, cs, hd, hs);
			}
			prev_ok = true;
			prev_x = x;
			prev_h = h;
			prev_depth = depth;
		} else {
			prev_ok = false; // Nothing recorded.
		}
		i++;
		if (i == leaf_end) {
			i = leaf_own_end;
			leaf_end = i;
			if ((hd & own_d) == own_d && (hs & own_s) == own_s) {
				break; // Everything this viewpoint answers for is already hidden.
			}
			level = 0;
			while (level < top && (i & ((16 << level) - 1)) == 0) {
				level += 2;
			}
		}
	}
	md |= hd & own_d;
	ms |= hs & own_s;
}

// The next viewpoint after `k` that is used and has a slot in the atlas, or -1.
int _line_shadow_next_view(LineShadowContext ctx, int k, uint stride) {
	for (uint j = uint(k + 1); j < ctx.sections; j++) {
		if ((j % stride == 0u || j == ctx.sections - 1u) && line_lights.data[ctx.section_base + j].atlas_rect.z > 0.0) {
			return int(j);
		}
	}
	return -1;
}

// The cells of `cd` (and of `cs`, if `do_specular`) hidden from the receiver.
void line_shadow_mask(LineShadowContext ctx, LineShadowCells cd, LineShadowCells cs, bool do_diffuse, bool do_specular, out uvec2 md, out uvec2 ms) {
	md = uvec2(0u);
	ms = uvec2(0u);
	float half_len = 0.5 * ctx.seg_length;
	float spacing = ctx.sections > 1u ? ctx.seg_length / float(ctx.sections - 1u) : 0.0;

	// Viewpoints within a third of the receiver's distance judge cells about equally
	// well, so far away only every stride'th is used, then only the middle one.
	// Dithered, so the switches cannot seam.
	uint stride = 1u;
	uint first = 0u;
	if (ctx.sections > 2u) {
		float reach = length(ctx.receiver.xy) / 3.0;
		uint middle = (ctx.sections - 1u) / 2u;
		float middle_reach = 0.5 * ctx.seg_length + abs(float(middle) * spacing - 0.5 * ctx.seg_length);
		if (log2(max(reach / middle_reach, 1e-6)) >= ctx.stride_jitter && line_lights.data[ctx.section_base + middle].atlas_rect.z > 0.0) {
			first = middle;
		} else {
			float n = floor(log2(max(reach / spacing, 1.0)) - ctx.stride_jitter);
			stride = min(1u << uint(clamp(n, 0.0, 5.0)), ctx.sections - 1u);
		}
	}

	// Each viewpoint owns the cells between its used neighbours; slotless ones hand
	// theirs over.
	int prev = -1;
	int k = first > 0u ? int(first) : _line_shadow_next_view(ctx, -1, stride);
	for (uint it = 0u; it < ctx.sections && k >= 0; it++) {
		int next = first > 0u ? -1 : _line_shadow_next_view(ctx, k, stride);
		float v = ctx.sections > 1u ? -half_len + float(k) * spacing : 0.0;
		float lo = prev >= 0 ? -half_len + float(prev) * spacing : -ctx.seg_length;
		float hi = next >= 0 ? -half_len + float(next) * spacing : ctx.seg_length;
		uvec2 own_d = do_diffuse ? line_shadow_cell_bits(cd, lo, hi) & ~md : uvec2(0u);
		uvec2 own_s = do_specular ? line_shadow_cell_bits(cs, lo, hi) & ~ms : uvec2(0u);
		if ((own_d | own_s) != uvec2(0u)) {
			_line_shadow_walk(ctx, uint(k), v, cd, cs, own_d, own_s, md, ms);
		}
		prev = k;
		k = next;
	}
}

// The specular lobe's integrand at a point `q` of the segment: the clamped cosine
// about +Y in cosine space, times the segment's density there.
float _line_shadow_specular_integrand(mat3 cos_xform, vec3 q, vec3 ct_n, float r_min_cos_sq) {
	vec3 lc = cos_xform * q;
	float lc2 = dot(lc, lc);
	// Direction from the unclamped vector, magnitude from the clamped one.
	vec3 wc = lc * inversesqrt(max(lc2, 1e-24));
	return max(wc.y, 0.0) * length(cross(wc, ct_n)) / max(lc2, r_min_cos_sq);
}

// Per-lobe visibility as a ratio estimator over the cells, weighted by f / pdf.
// View space, shading point at the origin: `po_w` is the closest point of the line,
// `wt` its direction, `d` its distance, `l1`/`l2`/`l_center` arclengths. `cos_xform`
// is zero when there is no specular lobe. `contact` is from line_shadow_contact_cells().
void line_shadow_visibility(uint idx, LineShadowContext ctx, vec3 normal, vec3 eye_vec, float alpha, vec3 po_w, vec3 wt,
		float d, float l1, float l2, float l_center, mat3 cos_xform, float ltc_min_radius, uvec4 contact, bool do_diffuse,
		out half r_vis_diffuse, inout half r_vis_specular) {
	half opacity = half(line_lights.data[idx].shadow_opacity);
	float half_len = 0.5 * (l2 - l1);

	vec3 ct = cos_xform * wt;
	bool do_specular = dot(ct, ct) > 1e-30;
	vec3 ct_n = do_specular ? normalize(ct) : vec3(0.0);
	float r_min_cos_sq = max(ltc_min_radius * ltc_min_radius, 1e-12);
	// Mirror point on the line and the lobe's footprint there.
	vec3 refl = reflect(-eye_vec, normal);
	float a = dot(wt, refl);
	float lm = clamp(abs(a) < 0.999 ? a * dot(po_w, refl) / ((1.0 - a) * (1.0 + a)) : 0.0, l1, l2);
	float w = max(2.0 * alpha * length(po_w + wt * lm), 1e-3);

	// Cells are placed in the light's frame, u = l - l_center.
	LineShadowCells cd = line_shadow_cells(-l_center, d, half_len, ctx.jitter);
	LineShadowCells cs = line_shadow_cells(lm - l_center, w, half_len, fract(ctx.jitter + 0.5));
	uvec2 md;
	uvec2 ms;
	line_shadow_mask(ctx, cd, cs, do_diffuse, do_specular, md, ms);
	// Screen-space contact hits: this many cells from the start of each quarter.
	LineShadowCells quarters = cd;
	quarters.jitter = 0.0;
	const int span = int(LINE_SHADOW_CELLS) / 4;
	for (int k = 0; k < 4; k++) {
		int n = int(contact[k]);
		if (n > 0) {
			md |= line_mask_range(k * span, k * span + n - 1);
			ms |= line_shadow_cell_bits(cs, line_shadow_cell_u(quarters, float(k * span)), line_shadow_cell_u(quarters, float(k * span + n)));
		}
	}
	if ((md | ms) == uvec2(0u)) {
		r_vis_diffuse = half(1.0);
		r_vis_specular = half(1.0);
		return;
	}

	// Accumulated as loss so a fully lit receiver gives exactly 1.0.
	float wd_sum = 0.0;
	float d_loss = 0.0;
	float ws_sum = 0.0;
	float s_loss = 0.0;
	[[dont_unroll]] for (uint i = 0u; i < LINE_SHADOW_CELLS; i++) {
		float li = line_shadow_cell_u(cd, float(i)) + l_center;
		float wd = max(dot(normal, (po_w + wt * li) * inversesqrt(d * d + li * li)), 0.0);
		wd_sum += wd;
		d_loss += line_mask_test(md, i) ? wd : 0.0;
		if (do_specular) {
			float dl = line_shadow_cell_u(cs, float(i)) + l_center - lm;
			float r2 = w * w + dl * dl;
			float ws = _line_shadow_specular_integrand(cos_xform, po_w + wt * (lm + dl), ct_n, r_min_cos_sq) / max(w * inversesqrt(r2) / r2, 1e-20);
			ws_sum += ws;
			s_loss += line_mask_test(ms, i) ? ws : 0.0;
		}
	}

	half vis_d = wd_sum > 0.0 ? half(1.0) - opacity * half(d_loss / wd_sum) : half(1.0);
	// Without the diffuse walk, a lobe with no weight keeps the caller's specular visibility.
	half vis_s = ws_sum > 0.0 ? half(1.0) - opacity * half(s_loss / ws_sum) : (do_diffuse ? vis_d : r_vis_specular);

	// A lobe with no weight above the horizon borrows the other's visibility.
	r_vis_diffuse = wd_sum > 0.0 ? vis_d : vis_s;
	r_vis_specular = vis_s;
}
