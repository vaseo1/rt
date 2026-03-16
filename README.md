# rt — Real-Time Path Tracer

Metal ray tracing engine for Apple Silicon. Loads Quake 1 BSP maps, renders with Monte Carlo path tracing, uses SVGF denoising for still frames, and keeps MetalFX temporal filtering available while moving.

## Quick Start

```bash
./fetch_assets.sh                    # download Quake shareware pak0.pak (once)
brew install open-image-denoise      # optional: enables the OIDN accumulation denoiser
xcodebuild -project rt.xcodeproj -scheme rt build
open ~/Library/Developer/Xcode/DerivedData/rt-*/Build/Products/Debug/rt.app
```

Requires macOS 13+, Apple Silicon, Xcode with Metal Toolchain. XcodeGen generates the `.xcodeproj` from `project.yml`. Assets (`.pak`, `.bsp`) are gitignored.

## Startup Arguments

```bash
open ~/Library/Developer/Xcode/DerivedData/rt-*/Build/Products/Debug/rt.app --args --start-pos 512 32 -768
open ~/Library/Developer/Xcode/DerivedData/rt-*/Build/Products/Debug/rt.app --args --verify --verify-frames 96 --start-pos=512,32,-768
open ~/Library/Developer/Xcode/DerivedData/rt-*/Build/Products/Debug/rt.app --args --verify --look-at-water --highlight-water
open ~/Library/Developer/Xcode/DerivedData/rt-*/Build/Products/Debug/rt.app --args --render-mode svgf
open ~/Library/Developer/Xcode/DerivedData/rt-*/Build/Products/Debug/rt.app --args --render-mode accumulation --denoise-method eaw
open ~/Library/Developer/Xcode/DerivedData/rt-*/Build/Products/Debug/rt.app --args --verify --render-mode accumulation --verify-checkpoints 1,2,4,8,16,32,64 --verify-sweep-denoise --verify-output-dir build/verify-accum-sweep
open ~/Library/Developer/Xcode/DerivedData/rt-*/Build/Products/Debug/rt.app --args --verify --render-mode accumulation --verify-reference-frames 256 --verify-checkpoints 8,16,32,64 --verify-sweep-denoise --verify-output-dir build/verify-accum-ref
```

- Default startup camera pose is `p480,50,150 | v0/0` as shown in the window title.
- `--start-pos x y z` or `--start-pos=x,y,z`: override the BSP spawn position with engine-space coordinates after the map loads.
- `--look-at x y z` or `--look-at=x,y,z`: orient the camera toward a target point after the start position is applied.
- `--look-at-water`: aim the camera at the detected water surface; if no explicit start position is given, move to a top-down framing position above it.
- `--highlight-water`: force liquid materials to bright magenta in the render so they are obvious in screenshots and verify captures.
- `--render-mode auto|raw|accumulation|svgf|metalfx`: choose the presentation path explicitly. `auto` uses SVGF while still and MetalFX while moving.
- `--denoise-method none|oidn|svgfplus|eaw|bilateral|nlm`: accumulation-only denoiser selection. Ignored outside `accumulation` mode.
- `--verify`, `--verify-frames`, `--verify-output`: single-shot verification flow.
- `--verify-checkpoints 1,2,4,8,...`: capture multiple frame checkpoints in one verify run.
- `--verify-output-dir path`: directory for multi-capture verify runs.
- `--verify-sweep-denoise`: in accumulation mode, sweep all accumulation denoisers during one verify run.
- `--verify-reference-frames N`: in accumulation mode, first capture a long-run reference image at `N` frames, then reset and compare later checkpoints against it.
- `--verify-reference-denoise-method none|oidn|svgfplus|eaw|bilateral|nlm`: optional reference capture denoiser, default `none`.
- When a BSP contains water, the renderer logs a suggested camera position just above the largest horizontal water surface.

## Controls

| Key | Action |
|-----|--------|
| WASD | Move |
| Shift + WASD | Move faster |
| Mouse | Look |
| Space | Up |
| M | Cycle render mode |
| N | Cycle accumulation denoiser |
| Escape | Release cursor / Quit |

## Architecture

```
src/
  main.swift                  entry point
  App/
    AppDelegate.swift         NSWindow + Metal device + mouse capture
    GameView.swift            CAMetalLayer + CVDisplayLink + input
  Renderer/
    Renderer.swift            orchestrator: scene load, adaptive res, accumulation
    PathTracer.swift          path trace, bloom, tonemap, texture management, GPU encode
    AccelStructure.swift      MTLAccelerationStructure build + compact
  Scene/
    BSPLoader.swift           Quake BSP v29 parser + PAKLoader
    SceneGeometry.swift       MTLBuffers for vertices, indices, materials
    Camera.swift              FPS camera, view/proj matrices, Halton jitter
  Upscaler/
    MetalFXUpscaler.swift     MTLFXTemporalScaler wrapper
shaders/
  Common.metal                shared structs (Vertex, Material, Uniforms), RNG, sampling
  PathTrace.metal             path trace kernel (texturing, NEE, DOF, emissives)
  Accumulate.metal            temporal accumulation + ACES tonemap kernels
  Bloom.metal                 HDR bright-pass, blur, and bloom composite kernels
```

## GPU Pipeline

```
pathTraceKernel → SVGF or MetalFX → bloom → tonemapKernel → blit to drawable
```

- **Path trace**: 1 spp/frame, Lambertian BRDF, cosine-weighted hemisphere, Russian roulette after bounce 3. Outputs HDR color + depth + motion vectors plus normal/albedo G-buffers.
- **SVGF**: temporal reprojection with history rejection, luminance moments, then 4 a-trous passes over depth, normal, luminance, and albedo.
- **Bloom**: thresholds bright HDR highlights, blurs at half resolution, composites before tonemap.
- **Tonemap**: ACES filmic + sRGB gamma, firefly clamp at luminance 50.

## Asset Pipeline

```
fetch_assets.sh → quake106.zip → lha extract → pak0.pak
                                                  ↓
PAKLoader → BSP bytes → BSPLoader → BSPData → SceneGeometry → AccelStructure
```

Fallback: Cornell box test scene if no assets found.

## Accumulation

Full resolution, 8 bounces, 1 spp per frame — always. Samples accumulate via running average `mix(history, current, 1/N)`. Camera movement resets accumulation. Frame semaphore (2 in-flight) prevents GPU saturation from blocking the main thread, ensuring input is always responsive.

When `accumulation` mode is active, `N` cycles accumulation-only denoisers on top of the running average:

- `None`: plain accumulation.
- `OIDN`: CPU-side Open Image Denoise on the accumulated HDR buffer. Best suited to static/offline use.
- `SVGF+`: variance-guided static A-Trous filtering on top of accumulated radiance.
- `EAW`: multi-pass edge-aware wavelet filtering.
- `Bilateral`: cross-bilateral post-filter guided by depth, normal, and albedo.
- `NLM`: non-local means post-filter for heavier static noise cleanup.

Reference-based verify runs report additional image-error metrics against the captured reference frame. In accumulation mode these now use the pre-tonemap linear HDR source when available, with LDR screenshot-space metrics kept only as a fallback:

- `hdr_ref_mae`: mean absolute linear RGB error.
- `hdr_ref_rmse`: root-mean-square linear RGB error.
- `hdr_ref_peak_psnr`: PSNR using the reference-frame peak radiance as the signal peak.
- `hdr_ref_luma_rmse`: linear luminance-space RMSE.
- `hdr_ref_rel_luma_rmse`: luminance RMSE normalized by local reference luminance.

## Key Implementation Details

- **Coord system**: Quake Z-up → engine Y-up via `SIMD3(x, z, -y)`
- **Texture pipeline**: BSP miptex level 0 pixels are palette-decoded to RGBA and bilinearly sampled in the path tracer
- **Emissives**: textures prefixed `light`, `*lava`, `*teleport`, `flame` become colored area lights with emissiveStrength 25.0
- **Smooth normals**: spatial hash welding with angle-weighted averaging and a 60° hard-edge threshold
- **Depth of field**: thin-lens ray generation with aperture disk sampling and focus-distance plane intersection
- **Struct layout**: Swift `Uniforms` uses `(Float, Float, Float)` tuples to match Metal `packed_float3`
- **Binary reads**: `memcpy`-based to avoid alignment traps on arbitrary BSP offsets
- **Retina**: drawable size = window size × `backingScaleFactor`

## Build Config

| Property | Value |
|----------|-------|
| Target | macOS 13.0 |
| Arch | arm64 |
| Swift | 5.9 |
| Signing | ad-hoc |
| Bundle ID | `dev.rt.pathtracer` |
| Generator | XcodeGen (`project.yml`) |
