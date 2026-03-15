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
    float         emissiveStrength;
    packed_float3 emissiveColor;
    float         roughness;
    float         metallic;
    float         transmissive;
    float         ior;
    uint          surfaceType;
    uint          textureOffset;    // into texture data buffer (0xFFFFFFFF = none)
    uint          textureWidth;
    uint          textureHeight;
    uint          _pad0;
};

enum SurfaceType : uint {
    SurfaceTypeDiffuse = 0u,
    SurfaceTypeMetal = 1u,
    SurfaceTypeLiquid = 2u,
    SurfaceTypeEmissive = 3u,
};

// ─── Light (emissive triangle for NEE) ───────────────────────────────────────

struct Light {
    packed_float3 v0;
    float         area;
    packed_float3 edge1;
    float         _pad0;
    packed_float3 edge2;
    float         _pad1;
    packed_float3 emission;
    float         _pad2;
    packed_float3 normal;
    float         _pad3;
};

// ─── Per-frame uniforms ──────────────────────────────────────────────────────

struct Uniforms {
    float4x4 currentViewProjection;
    float4x4 inverseViewProjection;
    float4x4 previousViewProjection;
    packed_float3 cameraPosition;
    uint     frameIndex;          // total frames rendered
    packed_float3 cameraRight;
    uint     accumulationCount;   // how many frames accumulated (resets on move)
    packed_float3 cameraUp;
    uint     samplesPerPixel;     // spp this frame
    packed_float3 cameraForward;
    uint     maxBounces;
    float    jitterX;
    float    jitterY;
    uint     renderWidth;
    uint     renderHeight;
    uint     outputWidth;
    uint     outputHeight;
    float    aperture;
    float    focusDistance;
    uint     lightCount;
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
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

inline void buildTangentFrame(float3 normal, thread float3& tangent, thread float3& bitangent) {
    float3 up = abs(normal.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(up, normal));
    bitangent = cross(normal, tangent);
}

inline float3 cosineWeightedHemisphere(float2 u, float3 normal) {
    // Cosine-weighted hemisphere sampling
    float r = sqrt(u.x);
    float theta = 2.0f * M_PI_F * u.y;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0f, 1.0f - u.x));

    float3 tangent;
    float3 bitangent;
    buildTangentFrame(normal, tangent, bitangent);

    return normalize(tangent * x + bitangent * y + normal * z);
}

inline float cosineHemispherePdf(float ndotl) {
    return ndotl > 0.0f ? ndotl / M_PI_F : 0.0f;
}

inline float dielectricF0(float ior) {
    float eta = max(1.0f, ior);
    float ratio = (eta - 1.0f) / (eta + 1.0f);
    return ratio * ratio;
}

inline float3 materialF0(float3 albedo, float metallic, float ior) {
    return mix(float3(dielectricF0(ior)), albedo, saturate(metallic));
}

inline float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0f - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

inline float ggxDistribution(float ndoth, float roughness) {
    float alpha = max(0.045f, roughness * roughness);
    float a2 = alpha * alpha;
    float denom = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / max(M_PI_F * denom * denom, 1e-5f);
}

inline float geometrySchlickGGX(float ndotx, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) * 0.125f;
    return ndotx / max(ndotx * (1.0f - k) + k, 1e-5f);
}

inline float geometrySmith(float ndotv, float ndotl, float roughness) {
    return geometrySchlickGGX(ndotv, roughness) * geometrySchlickGGX(ndotl, roughness);
}

inline float3 sampleGGXHalfVector(float3 normal, float roughness, float2 u) {
    float alpha = max(0.045f, roughness * roughness);
    float a2 = alpha * alpha;
    float phi = 2.0f * M_PI_F * u.x;
    float cosTheta = sqrt((1.0f - u.y) / max(1.0f + (a2 - 1.0f) * u.y, 1e-5f));
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

    float3 tangent;
    float3 bitangent;
    buildTangentFrame(normal, tangent, bitangent);

    float3 halfVector = tangent * (cos(phi) * sinTheta) +
                        bitangent * (sin(phi) * sinTheta) +
                        normal * cosTheta;
    return normalize(halfVector);
}

inline float ggxPdf(float3 normal, float3 viewDir, float3 lightDir, float roughness) {
    float3 halfVector = viewDir + lightDir;
    float halfLength2 = dot(halfVector, halfVector);
    if (halfLength2 <= 1e-6f) {
        return 0.0f;
    }

    halfVector *= rsqrt(halfLength2);
    float ndoth = saturate(dot(normal, halfVector));
    float vdoth = saturate(dot(viewDir, halfVector));
    if (ndoth <= 0.0f || vdoth <= 0.0f) {
        return 0.0f;
    }

    return ggxDistribution(ndoth, roughness) * ndoth / max(4.0f * vdoth, 1e-5f);
}

inline float3 evaluateCookTorranceBRDF(float3 albedo, float roughness, float metallic,
                                       float ior, float3 normal, float3 viewDir,
                                       float3 lightDir) {
    float ndotv = saturate(dot(normal, viewDir));
    float ndotl = saturate(dot(normal, lightDir));
    if (ndotv <= 0.0f || ndotl <= 0.0f) {
        return float3(0.0f);
    }

    float3 halfVector = viewDir + lightDir;
    float halfLength2 = dot(halfVector, halfVector);
    if (halfLength2 <= 1e-6f) {
        return float3(0.0f);
    }
    halfVector *= rsqrt(halfLength2);

    float ndoth = saturate(dot(normal, halfVector));
    float vdoth = saturate(dot(viewDir, halfVector));
    float3 F = fresnelSchlick(vdoth, materialF0(albedo, metallic, ior));
    float D = ggxDistribution(ndoth, roughness);
    float G = geometrySmith(ndotv, ndotl, roughness);
    float3 specular = (D * G * F) / max(4.0f * ndotv * ndotl, 1e-4f);
    float3 diffuseWeight = (1.0f - F) * (1.0f - saturate(metallic));
    float3 diffuse = diffuseWeight * albedo / M_PI_F;
    return diffuse + specular;
}

inline float luminance(float3 value) {
    return dot(value, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float2 concentricSampleDisk(float2 u) {
    float2 offset = 2.0f * u - 1.0f;

    if (offset.x == 0.0f && offset.y == 0.0f) {
        return float2(0.0f);
    }

    float radius;
    float theta;

    if (abs(offset.x) > abs(offset.y)) {
        radius = offset.x;
        theta = (M_PI_F / 4.0f) * (offset.y / offset.x);
    } else {
        radius = offset.y;
        theta = (M_PI_F / 2.0f) - (M_PI_F / 4.0f) * (offset.x / offset.y);
    }

    return radius * float2(cos(theta), sin(theta));
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
