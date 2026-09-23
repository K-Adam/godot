// Line light shadows: each viewpoint's polar map stores the plane through the receiver
// and the line as one row. Surfaces found there inside the receiver-segment triangle
// are projected from the receiver onto the segment and OR-ed into a mask of cells.
// The includer declares `line_lights`, `shadow_atlas`, `line_shadow_pyramid`,
// SAMPLER_NEAREST_CLAMP, M_PI and the half types.

// Texels of depth error still counted as the receiver's own surface.
#define LINE_SHADOW_PLANE_TEXELS 1.5
// How far off straight a block's pieces may bend and still be marked as a whole.
#define LINE_SHADOW_MARK_TEXELS 1.5
// Smaller maps gain nothing from it.
#define LINE_SHADOW_PIECES_MIN_SIZE 1024

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
	// Blocks whose patch at their nearest depth projects onto fewer cells than this are
	// marked as that patch instead of being walked. It fills gaps in blockers, so it is
	// only for receivers too coarse to show them, such as fog cells; 0 walks everything.
	float coarse_cells;
	// Scales the distance within which viewpoints judge alike; above 1 uses fewer.
	float reach_scale;
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
	ctx.coarse_cells = 0.0;
	ctx.reach_scale = 1.0;
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

#ifdef LINE_SHADOW_BLOCKER
// Sum of (blocker's distance from the line / receiver's) over newly hidden diffuse cells,
// and their count: how far the penumbra spreads (screen-space pass only).
vec2 line_shadow_blocker;
#endif

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
	uvec2 bits = line_shadow_cell_bits(cd, u0, u1);
#ifdef LINE_SHADOW_BLOCKER
	uvec2 fresh = bits & ~md;
	float n = float(bitCount(fresh.x) + bitCount(fresh.y));
	line_shadow_blocker += n * vec2(0.5 * (p0.x + p1.x) / rv.x, 1.0);
#endif
	md |= bits;
	ms |= line_shadow_cell_bits(cs, u0, u1);
}

// Texel `j` as the leaves see it: (point, height above the receiver's plane, depth or 0).
vec4 _line_shadow_texel(LineShadowContext ctx, ivec2 origin, int j, float delta, vec2 n2, float h0, float tol_scale) {
	float stored = texelFetch(sampler2D(shadow_atlas, SAMPLER_NEAREST_CLAMP), origin + ivec2(j, 0), 0).r;
	float theta = (float(j) + 0.5) * delta;
	vec2 dir = vec2(sin(theta), cos(theta));
	float depth = (1.0 - stored) * ctx.z_far;
	vec2 x = depth * dir;
	float nd = dot(dir, n2);
	return vec4(x, dot(x, n2) - h0 - tol_scale * depth * sqrt(max(1.0 - nd * nd, 0.0)), stored > 0.0 ? depth : 0.0);
}

// Distance along the ray at `theta` from the viewpoint to where it leaves the triangle
// between receiver `r` and the axis from `a` to `b` (viewpoint on [a, b]).
float _line_shadow_exit(vec2 r, float a, float b, float theta_r, float theta) {
	float e = theta < theta_r ? b : a;
	return e * r.x / (r.x * cos(theta) - (r.y - e) * sin(theta));
}

// Where the slice point `p` projects onto the line, seen from the receiver.
float _line_shadow_project(vec3 rv, vec2 p) {
	return rv.y + (p.y - rv.y) * rv.x / (rv.x - p.x) + rv.z;
}

// Adds to `md`/`ms` the cells of `own_d`/`own_s` hidden in viewpoint `view`'s map.
void _line_shadow_walk(LineShadowContext ctx, uint view, float v, LineShadowCells cd, LineShadowCells cs, uvec2 own_d, uvec2 own_s, inout uvec2 md, inout uvec2 ms) {
	uint record = ctx.section_base + view;
	vec4 rect = line_lights.data[record].atlas_rect;
	ivec2 origin = ivec2(round(rect.xy / ctx.texel_size));
	int size = int(round(rect.z / ctx.texel_size.x));
	int atlas_size = int(round(1.0 / ctx.texel_size.x));
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
	// Blocks of even levels span 4 to 1024 texels; none may span two slots.
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
	// Runs over straight pieces jump from each piece's first texel to its last.
	int split = 0;
	int jump_at = -1;
	int jump_to = 0;
	bool jumped = false;
	[[dont_unroll]] while (i < size) {
		if (i >= leaf_end) {
			int block = 4 << level;
			// Negative if a row of the block has straight pieces.
			float stored_nearest = texelFetch(sampler2D(line_shadow_pyramid, SAMPLER_NEAREST_CLAMP), (origin + ivec2(i, 0)) >> (level + 2), level).r;
			float nearest = (1.0 - abs(stored_nearest)) * ctx.z_far;
			// Widened by a texel for patches. The exit distance is convex along each
			// edge, so its maximum is at an end or at the receiver.
			float t0 = max(float(i - 1) * delta, 0.0);
			float t1 = min(float(i + block + 1) * delta, M_PI);
			float exit0 = _line_shadow_exit(rs, tri_a, tri_b, theta_r, t0);
			float exit1 = _line_shadow_exit(rs, tri_a, tri_b, theta_r, t1);
			float reach = max(exit0, exit1);
			bool holds_receiver = t0 <= theta_r && theta_r <= t1;
			if (holds_receiver) {
				reach = max(reach, r_len);
			}
			// A block whose patch at its nearest depth projects onto only a cell or so of
			// the segment is taken for that patch, which is most blocks near the line.
			// Deeper texels of the block are taken for the patch too, which errs towards
			// shadow where a blocker has gaps.
			bool coarse = false;
			// An edge the ray runs parallel to, or away from, gives no exit.
			if (ctx.coarse_cells > 0.0 && !holds_receiver && exit0 > 0.0 && exit1 > 0.0 && max(exit0, exit1) < 1e8) {
				float ea = _line_shadow_project(rv, vec2(sin(t0), cos(t0)) * min(nearest, exit0));
				float eb = _line_shadow_project(rv, vec2(sin(t1), cos(t1)) * min(nearest, exit1));
				float e_lo = min(ea, eb);
				float e_hi = max(ea, eb);
				float span = 0.0;
				if (own_d != uvec2(0u)) {
					span = line_shadow_cell_of(cd, e_hi) - line_shadow_cell_of(cd, e_lo);
				}
				if (own_s != uvec2(0u)) {
					span = max(span, line_shadow_cell_of(cs, e_hi) - line_shadow_cell_of(cs, e_lo));
				}
				coarse = span < ctx.coarse_cells;
			}
			// Chords between neighbouring texels dip in by up to 1 - cos(delta / 2).
			bool skip = nearest * cos(0.5 * delta) > 1.001 * reach;
			if (coarse && !skip) {
				// The plane's tolerance widens with the patch, so a receiver's own surface
				// seen across the block is not taken for a blocker.
				float ta = float(i) * delta;
				float tb = float(i + block) * delta;
				vec2 xa = nearest * vec2(sin(ta), cos(ta));
				vec2 xb = nearest * vec2(sin(tb), cos(tb));
				float tol = tol_scale * float(block) * nearest;
				_line_shadow_segment(xa, xb, dot(xa, n2) - h0 - tol, dot(xb, n2) - h0 - tol, rv, cd, cs, hd, hs);
				if ((hd & own_d) == own_d && (hs & own_s) == own_s) {
					break; // Everything this viewpoint answers for is already hidden.
				}
			}
			if (skip || coarse) {
				i += block;
				prev_ok = false;
				// Back up to the coarsest level this block boundary starts.
				while (level < top && (i & ((16 << level) - 1)) == 0) {
					level += 2;
				}
			} else if (level > 0 && size >= LINE_SHADOW_PIECES_MIN_SIZE) {
				uint code = stored_nearest < 0.0 ? uint(texelFetch(sampler2D(line_shadow_pyramid, SAMPLER_NEAREST_CLAMP), line_shadow_row_texel(origin.x, origin.y, size, level, i / block, atlas_size), 0).r) : 0u;
				int k = code >= 32u ? int(code >> 4u) - 2 : block - 1;
				// Its end texels are walked if each piece is, by its bend and half a texel,
				// either past the receiver's plane or side, or clear of both: then its
				// chord hides what its texels do.
				float bend = float(code & 15u) * 0.25 * delta;
				// A second piece that is not there stays past.
				bvec2 past = bvec2(true);
				bvec2 beyond = bvec2(true);
				bvec2 clear = bvec2(bend <= LINE_SHADOW_MARK_TEXELS * delta);
				// The leaves' tolerance vanishes along the plane's normal.
				float first_side = 0.0;
				bvec2 first_past = bvec2(false);
				[[dont_unroll]] for (int e = 0; e < (code == 0u ? 0 : (code < 32u ? 2 : 4)); e++) {
					vec4 p = _line_shadow_texel(ctx, origin, e == 0 ? i : (e == 1 ? i + k : (e == 2 ? i + k + 1 : i + block - 1)), delta, n2, h0, tol_scale);
					float margin = (bend + 0.5 * delta) * p.w;
					float g = 0.999 * rho_r - p.x;
					float side = p.x * n2.y - p.y * n2.x;
					bvec2 is_past_both = bvec2(p.z < -margin, dot(p.xy, n2) - h0 < -margin);
					bool piece_b = e >= 2;
					bool is_past = side * first_side < 0.0 ? first_past.y && is_past_both.y : first_past.x && is_past_both.x;
					if ((e & 1) == 0) {
						first_side = side;
						first_past = is_past_both;
						is_past = true;
					}
					bool is_beyond = g < -margin;
					bool is_clear = p.z > margin && g > margin;
					past = piece_b ? bvec2(past.x, past.y && is_past) : bvec2(past.x && is_past, past.y);
					beyond = piece_b ? bvec2(beyond.x, beyond.y && is_beyond) : bvec2(beyond.x && is_beyond, beyond.y);
					clear = piece_b ? bvec2(clear.x, clear.y && is_clear) : bvec2(clear.x && is_clear, clear.y);
				}
				if (code > 0u && all(bvec2(past.x || beyond.x || clear.x, past.y || beyond.y || clear.y))) {
					leaf_start = i;
					leaf_own_end = i + block;
					leaf_end = i + block + 1;
					// A piece of one texel has nothing to jump over.
					split = k;
					jump_at = k > 0 ? i : (k + 1 < block - 1 ? i + k + 1 : -1);
					jump_to = k > 0 ? i + k : i + block - 1;
					i = max(i - 1, 0);
					prev_ok = false;
				} else {
					level -= 2;
				}
			} else if (level > 0) {
				level -= 2;
			} else {
				leaf_start = i;
				leaf_own_end = i + 4;
				leaf_end = i + 5;
				jump_at = -1;
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
			bool joined = prev_ok && (jumped || abs(depth - prev_depth) <= cont_scale * min(depth, prev_depth));
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
		jumped = i == jump_at;
		if (jumped) {
			bool to_second = i == leaf_start && leaf_start + split + 1 < leaf_own_end - 1;
			i = jump_to;
			jump_at = to_second ? leaf_start + split + 1 : -1;
			jump_to = leaf_own_end - 1;
		} else {
			i++;
		}
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
		float reach = ctx.reach_scale * length(ctx.receiver.xy) / 3.0;
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
		// A lobe with no weight keeps the caller's visibility, as it does below.
		r_vis_specular = do_diffuse || do_specular ? half(1.0) : r_vis_specular;
		return;
	}
	// Nothing visible: every cell below would count as lost, as long as one of them has
	// weight at all. Cells run along the line in order and the weight's sign follows a
	// plane, so the outermost two decide.
	if (do_diffuse && md == uvec2(0xFFFFFFFFu) && (!do_specular || ms == uvec2(0xFFFFFFFFu)) &&
			max(dot(normal, po_w + wt * (line_shadow_cell_u(cd, 0.0) + l_center)),
					dot(normal, po_w + wt * (line_shadow_cell_u(cd, float(LINE_SHADOW_CELLS - 1u)) + l_center))) > 0.0) {
		r_vis_diffuse = half(1.0) - opacity;
		r_vis_specular = half(1.0) - opacity;
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
