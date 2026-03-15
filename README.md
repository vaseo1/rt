# rt — Real-Time Path Tracer

Metal ray tracing engine for Apple Silicon. Loads Quake 1 BSP maps, renders with Monte Carlo path tracing + temporal accumulation for converging global illumination.

## Quick Start

```bash
./fetch_assets.sh                    # download Quake shareware pak0.pak (once)
xcodebuild -project rt.xcodeproj -scheme rt build
open ~/Library/Developer/Xcode/DerivedData/rt-*/Build/Products/Debug/rt.app
```

Requires macOS 13+, Apple Silicon, Xcode with Metal Toolchain. XcodeGen generates the `.xcodeproj` from `project.yml`. Assets (`.pak`, `.bsp`) are gitignored.

## Controls

| Key | Action |
|-----|--------|
| WASD | Move |
| Mouse | Look |
| Space / Shift | Up / Down |
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
    PathTracer.swift          3 compute pipelines, texture management, GPU encode
    AccelStructure.swift      MTLAccelerationStructure build + compact
  Scene/
    BSPLoader.swift           Quake BSP v29 parser + PAKLoader
    SceneGeometry.swift       MTLBuffers for vertices, indices, materials
    Camera.swift              FPS camera, view/proj matrices, Halton jitter
  Upscaler/
    MetalFXUpscaler.swift     MTLFXTemporalScaler wrapper
shaders/
  Common.metal                shared structs (Vertex, Material, Uniforms), RNG, sampling
  PathTrace.metal             path trace kernel (bounces, Russian roulette, emissives)
  Accumulate.metal            temporal accumulation + ACES tonemap kernels
```

## GPU Pipeline

```
pathTraceKernel → accumulateKernel → tonemapKernel → blit to drawable
```

- **Path trace**: 1 spp/frame, Lambertian BRDF, cosine-weighted hemisphere, Russian roulette after bounce 3. Outputs HDR color + depth + motion vectors.
- **Accumulate**: running average `mix(history, current, 1/N)`. Resets on camera move.
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

## Key Implementation Details

- **Coord system**: Quake Z-up → engine Y-up via `SIMD3(x, z, -y)`
- **No texture sampling**: BSP texture names → approximate albedo colors
- **Emissives**: textures prefixed `light`, `*lava`, `*teleport`, `flame` → emissiveStrength 5.0
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
