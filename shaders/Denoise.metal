#include <metal_stdlib>
#include "Common.metal"

using namespace metal;

// ─── Clear a half-float texture to zero ──────────────────────────────────────

kernel void clearR16Kernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::write> output [[texture(0)]]
) {
    if (tid.x >= output.get_width() || tid.y >= output.get_height()) return;
    output.write(float4(0.0f), tid);
}

struct SVGFATrousUniforms {
    uint renderWidth;
    uint renderHeight;
    uint stepWidth;
    uint passIndex;
    float colorPhiScale;
    float normalPhi;
    float depthPhi;
    float albedoPhi;
};

constant float kATrousKernel[5] = {
    1.0f / 16.0f,
    1.0f / 4.0f,
    3.0f / 8.0f,
    1.0f / 4.0f,
    1.0f / 16.0f
};

constexpr sampler linearClampSampler(coord::normalized,
                                     address::clamp_to_edge,
                                     filter::linear);

inline float3 clampFireflies(float3 color, float maxLuminance) {
    float lum = luminance(color);
    if (lum > maxLuminance) {
        color *= maxLuminance / lum;
    }
    return color;
}

inline float3 clampHistoryToNeighborhood(uint2 tid,
                                         texture2d<float, access::read> currentColor,
                                         float3 historyRadiance,
                                         float sigmaScale) {
    uint width = currentColor.get_width();
    uint height = currentColor.get_height();

    float3 m1 = float3(0.0f);
    float3 m2 = float3(0.0f);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int sampleX = clamp(int(tid.x) + dx, 0, int(width) - 1);
            int sampleY = clamp(int(tid.y) + dy, 0, int(height) - 1);
            uint2 sCoord = uint2(sampleX, sampleY);

            float3 radiance = currentColor.read(sCoord).rgb;
            m1 += radiance;
            m2 += radiance * radiance;
        }
    }

    float3 mean = m1 / 9.0f;
    float3 variance = max(m2 / 9.0f - mean * mean, float3(0.0f));
    float3 sigma = sqrt(variance + 1e-5f) * sigmaScale;
    float3 colorMin = mean - sigma;
    float3 colorMax = mean + sigma;
    return clamp(historyRadiance, colorMin, colorMax);
}

inline bool isHistoryValid(float2 prevUV,
                           float currentDepth,
                           float4 currentNormal,
                           float4 currentAlbedo,
                           float prevDepth,
                           float4 prevNormal,
                           float4 prevAlbedo) {
    bool insideViewport = all(prevUV >= float2(0.0f)) && all(prevUV <= float2(1.0f));
    if (!insideViewport) {
        return false;
    }

    bool currentHasSurface = currentNormal.w > 0.5f;
    bool previousHasSurface = prevNormal.w > 0.5f;

    if (currentHasSurface != previousHasSurface) {
        return false;
    }

    if (!currentHasSurface && !previousHasSurface) {
        return true;
    }

    float normalSimilarity = dot(normalize(currentNormal.xyz), normalize(prevNormal.xyz));
    float albedoDelta = length(currentAlbedo.rgb - prevAlbedo.rgb);
    float depthDelta = abs(currentDepth - prevDepth);
    float depthThreshold = 0.0035f + 0.02f * max(currentDepth, prevDepth);

    return normalSimilarity > 0.85f &&
           albedoDelta < 0.2f &&
           depthDelta < depthThreshold;
}

kernel void svgfTemporalKernel(
    uint2 tid [[thread_position_in_grid]],
    constant Uniforms& uniforms [[buffer(0)]],
    texture2d<float, access::read> currentColor [[texture(0)]],
    texture2d<float, access::read> currentDepth [[texture(1)]],
    texture2d<float, access::read> currentMotion [[texture(2)]],
    texture2d<float, access::read> currentNormal [[texture(3)]],
    texture2d<float, access::read> currentAlbedo [[texture(4)]],
    texture2d<float, access::sample> historyColor [[texture(5)]],
    texture2d<float, access::sample> historyMoments [[texture(6)]],
    texture2d<float, access::sample> historyLength [[texture(7)]],
    texture2d<float, access::sample> historyDepth [[texture(8)]],
    texture2d<float, access::sample> historyNormal [[texture(9)]],
    texture2d<float, access::sample> historyAlbedo [[texture(10)]],
    texture2d<float, access::write> temporalColorOut [[texture(11)]],
    texture2d<float, access::write> momentsOut [[texture(12)]],
    texture2d<float, access::write> historyLengthOut [[texture(13)]]
) {
    if (tid.x >= uniforms.renderWidth || tid.y >= uniforms.renderHeight) {
        return;
    }

    float3 current = currentColor.read(tid).rgb;
    float currentDepthValue = currentDepth.read(tid).x;
    float2 motion = currentMotion.read(tid).xy;
    float4 currentNormalValue = currentNormal.read(tid);
    float4 currentAlbedoValue = currentAlbedo.read(tid);

    // Firefly suppression
    current = clampFireflies(current, 25.0f);

    // Temporal accumulation in RADIANCE space (demodulation only for spatial filter)
    float radLuminance = luminance(current);
    float2 currentMoments = float2(radLuminance, radLuminance * radLuminance);

    if (uniforms.accumulationCount <= 1) {
        temporalColorOut.write(float4(current, 1.0f), tid);
        momentsOut.write(float4(currentMoments, 0.0f, 0.0f), tid);
        historyLengthOut.write(float4(1.0f, 0.0f, 0.0f, 0.0f), tid);
        return;
    }

    float2 prevPixel = float2(tid) + motion;
    float2 prevUV = (prevPixel + 0.5f) / float2(uniforms.renderWidth, uniforms.renderHeight);

    float4 previousColorValue = historyColor.sample(linearClampSampler, prevUV);
    float2 previousMomentsValue = historyMoments.sample(linearClampSampler, prevUV).xy;
    float previousHistoryLength = historyLength.sample(linearClampSampler, prevUV).x;
    float previousDepthValue = historyDepth.sample(linearClampSampler, prevUV).x;
    float4 previousNormalValue = historyNormal.sample(linearClampSampler, prevUV);
    float4 previousAlbedoValue = historyAlbedo.sample(linearClampSampler, prevUV);

    bool historyValid = uniforms.accumulationCount > 1 && previousHistoryLength > 0.0f;
    historyValid = historyValid && isHistoryValid(prevUV,
                                                  currentDepthValue,
                                                  currentNormalValue,
                                                  currentAlbedoValue,
                                                  previousDepthValue,
                                                  previousNormalValue,
                                                  previousAlbedoValue);

    float historyLengthValue = 1.0f;
    float3 temporalRadiance = current;
    float2 temporalMoments = currentMoments;

    if (historyValid) {
        historyLengthValue = min(previousHistoryLength + 1.0f, 64.0f);
        float alpha = 1.0f / historyLengthValue;

        // Only clamp history for early frames; once converged, let temporal accumulate freely
        float3 history = previousColorValue.rgb;
        if (previousHistoryLength < 8.0f) {
            float clampSigmaScale = 1.0f + previousHistoryLength * 0.25f;
            history = clampHistoryToNeighborhood(tid, currentColor, history, clampSigmaScale);
        }

        float momentsAlpha = max(alpha, 0.1f);
        temporalRadiance = mix(history, current, alpha);
        temporalMoments = mix(previousMomentsValue, currentMoments, momentsAlpha);
    }

    temporalColorOut.write(float4(temporalRadiance, 1.0f), tid);
    momentsOut.write(float4(temporalMoments, 0.0f, 0.0f), tid);
    historyLengthOut.write(float4(historyLengthValue, 0.0f, 0.0f, 0.0f), tid);
}

kernel void svgfATrousKernel(
    uint2 tid [[thread_position_in_grid]],
    constant SVGFATrousUniforms& uniforms [[buffer(0)]],
    texture2d<float, access::read> sourceColor [[texture(0)]],
    texture2d<float, access::read> momentsTexture [[texture(1)]],
    texture2d<float, access::read> depthTexture [[texture(2)]],
    texture2d<float, access::read> normalTexture [[texture(3)]],
    texture2d<float, access::read> albedoTexture [[texture(4)]],
    texture2d<float, access::write> outputTexture [[texture(5)]],
    texture2d<float, access::read> historyLengthTexture [[texture(6)]]
) {
    if (tid.x >= uniforms.renderWidth || tid.y >= uniforms.renderHeight) {
        return;
    }

    float3 centerColor = sourceColor.read(tid).rgb;
    float2 centerMoments = momentsTexture.read(tid).xy;
    float centerDepth = depthTexture.read(tid).x;
    float4 centerNormal = normalTexture.read(tid);
    float4 centerAlbedo = albedoTexture.read(tid);
    bool centerHasSurface = centerNormal.w > 0.5f;

    // Adaptive spatial filter strength based on temporal convergence
    float historyLength = historyLengthTexture.read(tid).x;
    float adaptiveScale = clamp(3.0f / max(historyLength, 1.0f), 0.0f, 2.0f);

    // Skip spatial filter when temporal has sufficiently converged (>6 frames)
    if (adaptiveScale < 0.4f) {
        outputTexture.write(float4(centerColor, 1.0f), tid);
        return;
    }

    float variance = max(centerMoments.y - centerMoments.x * centerMoments.x, 0.0f);
    float colorPhi = max(uniforms.colorPhiScale * adaptiveScale * sqrt(variance + 1e-4f), 2.5e-3f);

    float3 filteredColor = float3(0.0f);
    float accumulatedWeight = 0.0f;

    for (int kernelY = -2; kernelY <= 2; kernelY++) {
        for (int kernelX = -2; kernelX <= 2; kernelX++) {
            int sampleX = clamp(int(tid.x) + kernelX * int(uniforms.stepWidth),
                                0,
                                int(uniforms.renderWidth) - 1);
            int sampleY = clamp(int(tid.y) + kernelY * int(uniforms.stepWidth),
                                0,
                                int(uniforms.renderHeight) - 1);
            uint2 sampleCoord = uint2(sampleX, sampleY);

            float3 sampleColor = sourceColor.read(sampleCoord).rgb;
            float sampleDepth = depthTexture.read(sampleCoord).x;
            float4 sampleNormal = normalTexture.read(sampleCoord);
            float4 sampleAlbedo = albedoTexture.read(sampleCoord);
            bool sampleHasSurface = sampleNormal.w > 0.5f;

            float kernelWeight = kATrousKernel[kernelX + 2] * kATrousKernel[kernelY + 2];

            float depthWeight = 1.0f;
            float normalWeight = 1.0f;
            float albedoWeight = 1.0f;
            if (centerHasSurface && sampleHasSurface) {
                depthWeight = exp(-abs(sampleDepth - centerDepth) * uniforms.depthPhi / max(float(uniforms.stepWidth), 1.0f));
                normalWeight = pow(saturate(dot(normalize(sampleNormal.xyz), normalize(centerNormal.xyz))),
                                   uniforms.normalPhi);
                albedoWeight = exp(-length(sampleAlbedo.rgb - centerAlbedo.rgb) * uniforms.albedoPhi);
            } else if (centerHasSurface != sampleHasSurface) {
                depthWeight = 0.0f;
                normalWeight = 0.0f;
                albedoWeight = 0.0f;
            }

            float colorDelta = abs(luminance(sampleColor) - luminance(centerColor));
            float colorWeight = exp(-colorDelta / colorPhi);
            float sampleWeight = kernelWeight * depthWeight * normalWeight * albedoWeight * colorWeight;

            filteredColor += sampleColor * sampleWeight;
            accumulatedWeight += sampleWeight;
        }
    }

    float safeWeight = max(accumulatedWeight, 1e-4f);
    outputTexture.write(float4(filteredColor / safeWeight, 1.0f), tid);
}

// ─── Demodulate: radiance / albedo → irradiance ──────────────────────────────

kernel void svgfDemodulateKernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::read> radianceTexture [[texture(0)]],
    texture2d<float, access::read> normalTexture [[texture(1)]],
    texture2d<float, access::read> albedoTexture [[texture(2)]],
    texture2d<float, access::write> outputTexture [[texture(3)]]
) {
    uint width = outputTexture.get_width();
    uint height = outputTexture.get_height();
    if (tid.x >= width || tid.y >= height) return;

    float3 rad = radianceTexture.read(tid).rgb;
    float4 normalValue = normalTexture.read(tid);
    bool hasSurface = normalValue.w > 0.5f;

    float3 albedo = hasSurface ? max(albedoTexture.read(tid).rgb, float3(0.03f)) : float3(1.0f);
    float3 irr = rad / albedo;

    outputTexture.write(float4(irr, 1.0f), tid);
}

// ─── Remodulate: irradiance × albedo → final radiance ────────────────────────

kernel void svgfRemodulateKernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::read> irradiance [[texture(0)]],
    texture2d<float, access::read> normalTexture [[texture(1)]],
    texture2d<float, access::read> albedoTexture [[texture(2)]],
    texture2d<float, access::write> outputTexture [[texture(3)]]
) {
    uint width = outputTexture.get_width();
    uint height = outputTexture.get_height();
    if (tid.x >= width || tid.y >= height) return;

    float3 irr = irradiance.read(tid).rgb;
    float4 normalValue = normalTexture.read(tid);
    bool hasSurface = normalValue.w > 0.5f;

    float3 albedo = hasSurface ? max(albedoTexture.read(tid).rgb, float3(0.03f)) : float3(1.0f);
    float3 result = irr * albedo;

    outputTexture.write(float4(result, 1.0f), tid);
}