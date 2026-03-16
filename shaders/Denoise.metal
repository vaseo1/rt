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

// ─── Post-filter uniforms (shared by EAW, Bilateral, NLM) ───────────────────

struct PostFilterUniforms {
    uint renderWidth;
    uint renderHeight;
    uint stepWidth;
    uint passIndex;
    float colorPhiScale;
    float normalPhi;
    float depthPhi;
    float albedoPhi;
    uint accumulationCount;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// ─── EAW: Edge-Aware Wavelet (A-Trous without convergence cutoff) ───────────
//
// Same A-Trous 5×5 filter as svgfATrousKernel but without history-length-based
// adaptive scaling or convergence cutoff. Filter strength adapts to accumulation
// count instead: stronger at low spp, gentler at high spp. Intended as a
// post-filter on the accumulated image.

kernel void postATrousKernel(
    uint2 tid [[thread_position_in_grid]],
    constant PostFilterUniforms& uniforms [[buffer(0)]],
    texture2d<float, access::read> sourceColor [[texture(0)]],
    texture2d<float, access::read> depthTexture [[texture(1)]],
    texture2d<float, access::read> normalTexture [[texture(2)]],
    texture2d<float, access::read> albedoTexture [[texture(3)]],
    texture2d<float, access::write> outputTexture [[texture(4)]]
) {
    if (tid.x >= uniforms.renderWidth || tid.y >= uniforms.renderHeight) return;

    float3 centerColor = sourceColor.read(tid).rgb;
    float centerDepth = depthTexture.read(tid).x;
    float4 centerNormal = normalTexture.read(tid);
    float4 centerAlbedo = albedoTexture.read(tid);
    bool centerHasSurface = centerNormal.w > 0.5f;

    // Adaptive strength: stronger at low accumulation, gentler when converged
    float accumScale = clamp(8.0f / max(float(uniforms.accumulationCount), 1.0f), 0.15f, 2.0f);
    float centerLum = luminance(centerColor);
    float colorPhi = max(uniforms.colorPhiScale * accumScale * sqrt(max(centerLum, 0.0f) + 1e-4f), 2.5e-3f);

    float3 filteredColor = float3(0.0f);
    float accumulatedWeight = 0.0f;

    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int sx = clamp(int(tid.x) + kx * int(uniforms.stepWidth), 0, int(uniforms.renderWidth) - 1);
            int sy = clamp(int(tid.y) + ky * int(uniforms.stepWidth), 0, int(uniforms.renderHeight) - 1);
            uint2 sc = uint2(sx, sy);

            float3 sampleColor = sourceColor.read(sc).rgb;
            float sampleDepth = depthTexture.read(sc).x;
            float4 sampleNormal = normalTexture.read(sc);
            float4 sampleAlbedo = albedoTexture.read(sc);
            bool sampleHasSurface = sampleNormal.w > 0.5f;

            float kernelWeight = kATrousKernel[kx + 2] * kATrousKernel[ky + 2];

            float depthWeight = 1.0f;
            float normalWeight = 1.0f;
            float albedoWeight = 1.0f;
            if (centerHasSurface && sampleHasSurface) {
                depthWeight = exp(-abs(sampleDepth - centerDepth) * uniforms.depthPhi / max(float(uniforms.stepWidth), 1.0f));
                normalWeight = pow(saturate(dot(normalize(sampleNormal.xyz), normalize(centerNormal.xyz))), uniforms.normalPhi);
                albedoWeight = exp(-length(sampleAlbedo.rgb - centerAlbedo.rgb) * uniforms.albedoPhi);
            } else if (centerHasSurface != sampleHasSurface) {
                depthWeight = 0.0f;
                normalWeight = 0.0f;
                albedoWeight = 0.0f;
            }

            float colorDelta = abs(luminance(sampleColor) - centerLum);
            float colorWeight = exp(-colorDelta / colorPhi);
            float w = kernelWeight * depthWeight * normalWeight * albedoWeight * colorWeight;

            filteredColor += sampleColor * w;
            accumulatedWeight += w;
        }
    }

    outputTexture.write(float4(filteredColor / max(accumulatedWeight, 1e-4f), 1.0f), tid);
}

// ─── Cross-Bilateral Filter ─────────────────────────────────────────────────
//
// G-buffer-guided Gaussian bilateral filter with configurable radius.
// Weights: Gaussian_spatial × depth × normal × albedo × color similarity.
// Better edge preservation than A-Trous for fine geometric detail.

struct CrossBilateralUniforms {
    uint renderWidth;
    uint renderHeight;
    int radius;          // kernel half-size (e.g. 5 → 11×11)
    uint accumulationCount;
    float sigmaSpatial;  // spatial Gaussian sigma
    float sigmaColor;    // color similarity sigma
    float depthPhi;
    float normalPhi;
    float albedoPhi;
    float _pad0;
    float _pad1;
    float _pad2;
};

kernel void crossBilateralKernel(
    uint2 tid [[thread_position_in_grid]],
    constant CrossBilateralUniforms& uniforms [[buffer(0)]],
    texture2d<float, access::read> sourceColor [[texture(0)]],
    texture2d<float, access::read> depthTexture [[texture(1)]],
    texture2d<float, access::read> normalTexture [[texture(2)]],
    texture2d<float, access::read> albedoTexture [[texture(3)]],
    texture2d<float, access::write> outputTexture [[texture(4)]]
) {
    if (tid.x >= uniforms.renderWidth || tid.y >= uniforms.renderHeight) return;

    float3 centerColor = sourceColor.read(tid).rgb;
    float centerDepth = depthTexture.read(tid).x;
    float4 centerNormal = normalTexture.read(tid);
    float4 centerAlbedo = albedoTexture.read(tid);
    bool centerHasSurface = centerNormal.w > 0.5f;
    float centerLum = luminance(centerColor);

    float invSigmaSpatial2 = -0.5f / (uniforms.sigmaSpatial * uniforms.sigmaSpatial);
    float invSigmaColor2 = -0.5f / (uniforms.sigmaColor * uniforms.sigmaColor);

    float3 filteredColor = float3(0.0f);
    float totalWeight = 0.0f;

    int r = uniforms.radius;

    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int sx = clamp(int(tid.x) + dx, 0, int(uniforms.renderWidth) - 1);
            int sy = clamp(int(tid.y) + dy, 0, int(uniforms.renderHeight) - 1);
            uint2 sc = uint2(sx, sy);

            float3 sampleColor = sourceColor.read(sc).rgb;
            float sampleDepth = depthTexture.read(sc).x;
            float4 sampleNormal = normalTexture.read(sc);
            float4 sampleAlbedo = albedoTexture.read(sc);
            bool sampleHasSurface = sampleNormal.w > 0.5f;

            // Spatial weight
            float dist2 = float(dx * dx + dy * dy);
            float wSpatial = exp(dist2 * invSigmaSpatial2);

            // Color weight
            float colorDelta = abs(luminance(sampleColor) - centerLum);
            float wColor = exp(colorDelta * colorDelta * invSigmaColor2);

            // G-buffer weights
            float wDepth = 1.0f;
            float wNormal = 1.0f;
            float wAlbedo = 1.0f;
            if (centerHasSurface && sampleHasSurface) {
                wDepth = exp(-abs(sampleDepth - centerDepth) * uniforms.depthPhi);
                wNormal = pow(saturate(dot(normalize(sampleNormal.xyz), normalize(centerNormal.xyz))), uniforms.normalPhi);
                wAlbedo = exp(-length(sampleAlbedo.rgb - centerAlbedo.rgb) * uniforms.albedoPhi);
            } else if (centerHasSurface != sampleHasSurface) {
                wSpatial = 0.0f;
            }

            float w = wSpatial * wColor * wDepth * wNormal * wAlbedo;
            filteredColor += sampleColor * w;
            totalWeight += w;
        }
    }

    outputTexture.write(float4(filteredColor / max(totalWeight, 1e-4f), 1.0f), tid);
}

// ─── Non-Local Means Filter ─────────────────────────────────────────────────
//
// Patch-based denoiser: for each pixel, searches a window, compares patches,
// and averages weighted by patch similarity. G-buffer pre-rejection trims the
// search space. Best quality for MC noise but expensive (~11k taps/pixel).

struct NLMUniforms {
    uint renderWidth;
    uint renderHeight;
    int searchRadius;    // search window half-size (e.g. 10 → 21×21)
    int patchRadius;     // patch half-size (e.g. 2 → 5×5)
    float h;             // filter strength (higher = more smoothing)
    float depthRejectThreshold;
    float normalRejectDot;
    float albedoRejectDelta;
    uint accumulationCount;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

kernel void nlmDenoiseKernel(
    uint2 tid [[thread_position_in_grid]],
    constant NLMUniforms& uniforms [[buffer(0)]],
    texture2d<float, access::read> sourceColor [[texture(0)]],
    texture2d<float, access::read> depthTexture [[texture(1)]],
    texture2d<float, access::read> normalTexture [[texture(2)]],
    texture2d<float, access::read> albedoTexture [[texture(3)]],
    texture2d<float, access::write> outputTexture [[texture(4)]]
) {
    if (tid.x >= uniforms.renderWidth || tid.y >= uniforms.renderHeight) return;

    int w = int(uniforms.renderWidth);
    int h = int(uniforms.renderHeight);
    int sr = uniforms.searchRadius;
    int pr = uniforms.patchRadius;

    float3 centerColor = sourceColor.read(tid).rgb;
    float centerDepth = depthTexture.read(tid).x;
    float4 centerNormal = normalTexture.read(tid);
    float4 centerAlbedo = albedoTexture.read(tid);
    bool centerHasSurface = centerNormal.w > 0.5f;

    float invH2 = 1.0f / (uniforms.h * uniforms.h);
    int patchSize = (2 * pr + 1);
    float invPatchArea = 1.0f / float(patchSize * patchSize);

    float3 filteredColor = float3(0.0f);
    float totalWeight = 0.0f;

    for (int sy = -sr; sy <= sr; sy++) {
        for (int sx = -sr; sx <= sr; sx++) {
            int qx = clamp(int(tid.x) + sx, 0, w - 1);
            int qy = clamp(int(tid.y) + sy, 0, h - 1);
            uint2 qc = uint2(qx, qy);

            // G-buffer pre-rejection
            if (centerHasSurface) {
                float qDepth = depthTexture.read(qc).x;
                float4 qNormal = normalTexture.read(qc);
                float4 qAlbedo = albedoTexture.read(qc);
                bool qHasSurface = qNormal.w > 0.5f;

                if (!qHasSurface) continue;
                if (abs(qDepth - centerDepth) > uniforms.depthRejectThreshold) continue;
                if (dot(normalize(qNormal.xyz), normalize(centerNormal.xyz)) < uniforms.normalRejectDot) continue;
                if (length(qAlbedo.rgb - centerAlbedo.rgb) > uniforms.albedoRejectDelta) continue;
            }

            // Patch distance
            float patchDist2 = 0.0f;
            for (int py = -pr; py <= pr; py++) {
                for (int px = -pr; px <= pr; px++) {
                    int cx = clamp(int(tid.x) + px, 0, w - 1);
                    int cy = clamp(int(tid.y) + py, 0, h - 1);
                    int rx = clamp(qx + px, 0, w - 1);
                    int ry = clamp(qy + py, 0, h - 1);

                    float3 cp = sourceColor.read(uint2(cx, cy)).rgb;
                    float3 rp = sourceColor.read(uint2(rx, ry)).rgb;
                    float3 diff = cp - rp;
                    patchDist2 += dot(diff, diff);
                }
            }

            float avgDist2 = patchDist2 * invPatchArea;
            float weight = exp(-max(avgDist2 * invH2, 0.0f));

            float3 qColor = sourceColor.read(qc).rgb;
            filteredColor += qColor * weight;
            totalWeight += weight;
        }
    }

    outputTexture.write(float4(filteredColor / max(totalWeight, 1e-4f), 1.0f), tid);
}