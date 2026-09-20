# LTC lookup tables

`ltc_lut1.dds` / `ltc_lut2.dds` — isotropic GGX fit by Eric Heitz,
<https://eheitzresearch.wordpress.com/415-2/>. 64x64 RGBA32F, uploaded as two 2D
textures.

`ltc_aniso_lut.bin` — anisotropic GGX fit from *Bringing Linearly Transformed
Cosines to Anisotropic GGX* (Aakash KT, Eric Heitz, Jonathan Dupuy, P. J.
Narayanan, I3D 2022), MIT licensed,
<https://github.com/AakashKT/LTC-Anisotropic> (`LUT/alpha_lambda_theta_phi.npy`).

The source table is 8x8x8x8 of 3x3 matrices indexed by roughness, roughness
ratio, view elevation and view azimuth. It is repacked here into a single
8x8x192 RGBA16F volume so that it costs one binding and six filtered fetches:

* x = view azimuth, y = view elevation (both hardware filtered),
* z = `matrix row * 64 + roughness slice * 8 + roughness ratio`, so the hardware
  also filters the ratio while the shader blends the roughness slices by hand,
* rgb = one row of the matrix, a = unused.

To regenerate from the upstream `.npy`:

```python
import numpy as np
a = np.load("alpha_lambda_theta_phi.npy")  # [alpha][lambda][theta][phi][3][3]
out = np.zeros((192, 8, 8, 4), dtype=np.float16)
for r in range(3):
    for al in range(8):
        for lm in range(8):
            out[r * 64 + al * 8 + lm, :, :, 0:3] = a[al, lm, :, :, r, :]
out.tofile("ltc_aniso_lut.bin")
```
