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

    uint2 inputSize = uint2(hdrInput.get_width(), hdrInput.get_height());

    // If input resolution != output resolution, do nearest-neighbor sample.
    float2 scale = float2(inputSize) / float2(outputSize);
    uint2 srcCoord = uint2(float2(tid) * scale);
    srcCoord = min(srcCoord, inputSize - 1);

    float exposure = max(uniforms.exposure, 1e-3f);
    float3 hdr = hdrInput.read(srcCoord).rgb * exposure;

    // Clamp fireflies
    float lum = dot(hdr, float3(0.2126f, 0.7152f, 0.0722f));
    if (lum > 50.0f) hdr *= 50.0f / lum;

    float3 tonemapped = acesFilmic(hdr);
    float3 srgb = linearToSRGB(tonemapped);

    ldrOutput.write(float4(srgb, 1.0f), tid);
}

kernel void measureExposureKernel(
    uint2                             tid                 [[thread_position_in_grid]],
    texture2d<float, access::read>    hdrInput            [[texture(0)]],
    device float*                     averageLuminanceOut [[buffer(0)]]
) {
    if (tid.x != 0 || tid.y != 0) return;

    uint width = hdrInput.get_width();
    uint height = hdrInput.get_height();
    uint sampleCountX = min(width, 64u);
    uint sampleCountY = min(height, 64u);
    float logLuminanceSum = 0.0f;
    uint sampleCount = 0u;

    for (uint sampleY = 0; sampleY < sampleCountY; sampleY++) {
        float v = (float(sampleY) + 0.5f) / float(sampleCountY);
        uint y = min(uint(v * float(height)), height - 1u);

        for (uint sampleX = 0; sampleX < sampleCountX; sampleX++) {
            float u = (float(sampleX) + 0.5f) / float(sampleCountX);
            uint x = min(uint(u * float(width)), width - 1u);
            float3 hdr = hdrInput.read(uint2(x, y)).rgb;
            float lum = clamp(luminance(hdr), 1e-4f, 64.0f);
            logLuminanceSum += log(lum);
            sampleCount += 1u;
        }
    }

    float averageLogLuminance = logLuminanceSum / max(float(sampleCount), 1.0f);
    averageLuminanceOut[0] = exp(averageLogLuminance);
}
