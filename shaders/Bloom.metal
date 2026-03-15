#include <metal_stdlib>
#include "Common.metal"

using namespace metal;

struct BloomUniforms {
    uint  sourceWidth;
    uint  sourceHeight;
    uint  outputWidth;
    uint  outputHeight;
    uint  bloomWidth;
    uint  bloomHeight;
    float threshold;
    float softKnee;
    float intensity;
    float blurScale;
    float directionX;
    float directionY;
    float padding0;
    float padding1;
};

constexpr sampler bloomSampler(coord::normalized,
                               address::clamp_to_edge,
                               filter::linear);

inline float bloomContribution(float3 hdr, float threshold, float softKnee) {
    float brightness = luminance(hdr);
    float knee = max(threshold * softKnee, 1e-5f);
    float soft = clamp(brightness - threshold + knee, 0.0f, 2.0f * knee);
    soft = (soft * soft) / max(4.0f * knee, 1e-5f);
    float contribution = max(soft, brightness - threshold);
    return contribution / max(brightness, 1e-4f);
}

kernel void extractBloomKernel(
    uint2                              tid          [[thread_position_in_grid]],
    constant BloomUniforms&            uniforms     [[buffer(0)]],
    texture2d<float, access::sample>   hdrInput     [[texture(0)]],
    texture2d<float, access::write>    bloomOutput  [[texture(1)]]
) {
    if (tid.x >= uniforms.bloomWidth || tid.y >= uniforms.bloomHeight) return;

    float2 bloomSize = float2(max(uniforms.bloomWidth, 1u), max(uniforms.bloomHeight, 1u));
    float2 sourceTexel = 1.0f / float2(max(uniforms.sourceWidth, 1u), max(uniforms.sourceHeight, 1u));
    float2 uv = (float2(tid) + 0.5f) / bloomSize;

    float3 hdr = float3(0.0f);
    hdr += hdrInput.sample(bloomSampler, uv + sourceTexel * float2(-0.5f, -0.5f)).rgb;
    hdr += hdrInput.sample(bloomSampler, uv + sourceTexel * float2(0.5f, -0.5f)).rgb;
    hdr += hdrInput.sample(bloomSampler, uv + sourceTexel * float2(-0.5f, 0.5f)).rgb;
    hdr += hdrInput.sample(bloomSampler, uv + sourceTexel * float2(0.5f, 0.5f)).rgb;
    hdr *= 0.25f;

    float contribution = bloomContribution(hdr, uniforms.threshold, uniforms.softKnee);
    bloomOutput.write(float4(hdr * contribution, 1.0f), tid);
}

kernel void blurBloomKernel(
    uint2                              tid          [[thread_position_in_grid]],
    constant BloomUniforms&            uniforms     [[buffer(0)]],
    texture2d<float, access::sample>   bloomInput   [[texture(0)]],
    texture2d<float, access::write>    bloomOutput  [[texture(1)]]
) {
    if (tid.x >= uniforms.bloomWidth || tid.y >= uniforms.bloomHeight) return;

    constexpr float weights[5] = {
        0.227027f,
        0.1945946f,
        0.1216216f,
        0.054054f,
        0.016216f,
    };

    float2 bloomSize = float2(max(uniforms.bloomWidth, 1u), max(uniforms.bloomHeight, 1u));
    float2 uv = (float2(tid) + 0.5f) / bloomSize;
    float2 axis = float2(uniforms.directionX, uniforms.directionY) * uniforms.blurScale / bloomSize;

    float3 blurred = bloomInput.sample(bloomSampler, uv).rgb * weights[0];
    for (uint tap = 1; tap < 5; tap++) {
        float2 offset = axis * float(tap);
        blurred += bloomInput.sample(bloomSampler, uv + offset).rgb * weights[tap];
        blurred += bloomInput.sample(bloomSampler, uv - offset).rgb * weights[tap];
    }

    bloomOutput.write(float4(blurred, 1.0f), tid);
}

kernel void compositeBloomKernel(
    uint2                              tid             [[thread_position_in_grid]],
    constant BloomUniforms&            uniforms        [[buffer(0)]],
    texture2d<float, access::sample>   hdrInput        [[texture(0)]],
    texture2d<float, access::sample>   bloomInput      [[texture(1)]],
    texture2d<float, access::write>    compositeOutput [[texture(2)]]
) {
    if (tid.x >= uniforms.outputWidth || tid.y >= uniforms.outputHeight) return;

    float2 outputSize = float2(max(uniforms.outputWidth, 1u), max(uniforms.outputHeight, 1u));
    float2 uv = (float2(tid) + 0.5f) / outputSize;

    float3 hdr = hdrInput.sample(bloomSampler, uv).rgb;
    float3 bloom = bloomInput.sample(bloomSampler, uv).rgb;
    compositeOutput.write(float4(hdr + bloom * uniforms.intensity, 1.0f), tid);
}