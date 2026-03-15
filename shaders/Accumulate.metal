#include <metal_stdlib>
#include "Common.metal"

using namespace metal;

// ─── Temporal Accumulation Kernel ────────────────────────────────────────────
//
// Running average: history = history * ((N-1)/N) + current * (1/N)
// When N=1 (just moved), outputs current frame directly.
// As N increases (stationary), noise converges toward ground truth GI.

kernel void accumulateKernel(
    uint2                             tid           [[thread_position_in_grid]],
    constant Uniforms&                uniforms      [[buffer(0)]],
    texture2d<float, access::read>    currentFrame  [[texture(0)]],
    texture2d<float, access::read>    historyFrame  [[texture(1)]],
    texture2d<float, access::write>   outputFrame   [[texture(2)]]
) {
    if (tid.x >= uniforms.renderWidth || tid.y >= uniforms.renderHeight) return;

    float4 current = currentFrame.read(tid);
    float  N       = float(uniforms.accumulationCount);

    float4 result;
    if (N <= 1.0f) {
        // First frame or just moved — no history to blend
        result = current;
    } else {
        float4 history = historyFrame.read(tid);
        float  weight  = 1.0f / N;
        result = mix(history, current, weight);
    }

    outputFrame.write(result, tid);
}

// ─── Tonemap + Display Kernel ────────────────────────────────────────────────
//
// Reads accumulated HDR buffer, applies ACES filmic tonemapping + sRGB gamma,
// writes to displayable output.

kernel void tonemapKernel(
    uint2                             tid          [[thread_position_in_grid]],
    constant Uniforms&                uniforms     [[buffer(0)]],
    texture2d<float, access::read>    hdrInput     [[texture(0)]],
    texture2d<float, access::write>   ldrOutput    [[texture(1)]]
) {
    uint2 outputSize = uint2(uniforms.outputWidth, uniforms.outputHeight);
    if (tid.x >= outputSize.x || tid.y >= outputSize.y) return;

    // If render resolution != output resolution, do nearest-neighbor sample
    float2 scale = float2(uniforms.renderWidth, uniforms.renderHeight) /
                   float2(outputSize);
    uint2 srcCoord = uint2(float2(tid) * scale);
    srcCoord = min(srcCoord, uint2(uniforms.renderWidth - 1, uniforms.renderHeight - 1));

    constexpr float exposure = 4.0f;
    float3 hdr = hdrInput.read(srcCoord).rgb * exposure;

    // Clamp fireflies
    float lum = dot(hdr, float3(0.2126f, 0.7152f, 0.0722f));
    if (lum > 50.0f) hdr *= 50.0f / lum;

    float3 tonemapped = acesFilmic(hdr);
    float3 srgb = linearToSRGB(tonemapped);

    ldrOutput.write(float4(srgb, 1.0f), tid);
}
