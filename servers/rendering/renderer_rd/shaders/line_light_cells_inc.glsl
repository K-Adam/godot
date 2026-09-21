// Cells along a line light that its shadow mask marks as hidden. They are equal
// shares of the density w / (w^2 + x^2)^(3/2), x = u - center, i.e. uniform in
// f(x) = x / sqrt(w^2 + x^2), each sampled at a jittered spot.

#define LINE_SHADOW_CELLS 64u

struct LineShadowCells {
	float center;
	float width;
	float f_lo;
	float scale; // Cells per unit of f.
	float jitter;
};

LineShadowCells line_shadow_cells(float center, float width, float half_length, float jitter) {
	LineShadowCells c;
	c.center = center;
	c.width = width;
	c.jitter = jitter;
	float a = -half_length - center;
	float b = half_length - center;
	c.f_lo = a * inversesqrt(width * width + a * a);
	c.scale = float(LINE_SHADOW_CELLS) / max(b * inversesqrt(width * width + b * b) - c.f_lo, 1e-7);
	return c;
}

// The point at fractional cell index `i`, measured from the segment's center.
float line_shadow_cell_u(LineShadowCells c, float i) {
	float a = c.f_lo + (i + c.jitter) / c.scale;
	// (1 - a) * (1 + a) avoids cancellation near the ends.
	return c.center + c.width * a * inversesqrt(max((1.0 - a) * (1.0 + a), 1e-12));
}

// Fractional cell index of the point u, such that cell i is sampled at exactly i.
float line_shadow_cell_of(LineShadowCells c, float u) {
	float x = clamp(u - c.center, -1e4, 1e4);
	return (x * inversesqrt(c.width * c.width + x * x) - c.f_lo) * c.scale - c.jitter;
}

// Masks are two words; every shift stays within 0..31.
uint _line_mask_word(int a, int b) {
	return (0xFFFFFFFFu >> uint(31 - b)) & (0xFFFFFFFFu << uint(a));
}

// Cells a to b inclusive, 0 <= a <= b < 64.
uvec2 line_mask_range(int a, int b) {
	uvec2 m = uvec2(0u);
	if (a < 32) {
		m.x = _line_mask_word(a, min(b, 31));
	}
	if (b >= 32) {
		m.y = _line_mask_word(max(a, 32) - 32, b - 32);
	}
	return m;
}

bool line_mask_test(uvec2 m, uint i) {
	return ((m[i >> 5u] >> (i & 31u)) & 1u) != 0u;
}

int line_mask_lsb(uvec2 m) {
	return m.x != 0u ? findLSB(m.x) : 32 + findLSB(m.y);
}

int line_mask_msb(uvec2 m) {
	return m.y != 0u ? 32 + findMSB(m.y) : findMSB(m.x);
}

// The cells sampled within [u0, u1].
uvec2 line_shadow_cell_bits(LineShadowCells c, float u0, float u1) {
	int a = max(int(ceil(clamp(line_shadow_cell_of(c, u0), -1.0, 64.0))), 0);
	int b = min(int(floor(clamp(line_shadow_cell_of(c, u1), -1.0, 64.0))), 63);
	return b < a ? uvec2(0u) : line_mask_range(a, b);
}
