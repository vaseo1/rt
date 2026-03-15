#ifndef COMMON_METAL
#define COMMON_METAL

#include <metal_stdlib>
using namespace metal;

// ─── Vertex layout matching Swift side ───────────────────────────────────────

struct Vertex {
    packed_float3 position;
    packed_float3 normal;
    packed_float2 uv;
    uint          materialIndex;
};

// ─── Material descriptor ─────────────────────────────────────────────────────

struct Material {
    packed_float3 albedo;
    float         emissiveStrength; // 0 = non-emissive, >0 = light source
};

// ─── Per-frame uniforms ──────────────────────────────────────────────────────

struct Uniforms {
    float4x4 inverseViewProjection;
    float4x4 previousViewProjection;
    packed_float3 cameraPosition;
    uint     frameIndex;          // total frames rendered
    uint     accumulationCount;   // how many frames accumulated (resets on move)
    uint     samplesPerPixel;     // spp this frame
    uint     maxBounces;
    float    jitterX;
    float    jitterY;
    uint     renderWidth;
    uint     renderHeight;
    uint     outputWidth;
    uint     outputHeight;
};

// ─── Random number generation (PCG) ─────────────────────────────────────────

inline uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct RNG {
    uint state;

    RNG(uint2 pixelCoord, uint frameIndex, uint sampleIndex) {
        state = pcg_hash(pixelCoord.x + pixelCoord.y * 16384u + frameIndex * 16384u * 16384u + sampleIndex);
    }

    float next() {
        state = pcg_hash(state);
        return float(state) / float(0xFFFFFFFFu);
    }

    float2 next2() {
        return float2(next(), next());
    }
};

// ─── Sampling helpers ────────────────────────────────────────────────────────

inline float3 cosineWeightedHemisphere(float2 u, float3 normal) {
    // Cosine-weighted hemisphere sampling
    float r = sqrt(u.x);
    float theta = 2.0f * M_PI_F * u.y;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0f, 1.0f - u.x));

    // Build tangent frame from normal
    float3 up = abs(normal.y) < 0.999f ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    return normalize(tangent * x + bitangent * y + normal * z);
}

// ─── Tonemapping ─────────────────────────────────────────────────────────────

inline float3 acesFilmic(float3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

inline float3 linearToSRGB(float3 linear) {
    return pow(linear, float3(1.0f / 2.2f));
}

#endif // COMMON_METAL
