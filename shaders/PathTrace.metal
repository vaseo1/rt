#include <metal_stdlib>
#include <metal_raytracing>
#include "Common.metal"

using namespace metal;
using namespace raytracing;

// ─── Path Tracing Compute Kernel ─────────────────────────────────────────────
//
// Dispatched per-pixel at the current internal render resolution.
// Traces paths through the scene, accumulating radiance via Monte Carlo
// integration with cosine-weighted hemisphere sampling (Lambertian BRDF)
// and Russian roulette termination.

kernel void pathTraceKernel(
    uint2                                     tid               [[thread_position_in_grid]],
    constant Uniforms&                        uniforms          [[buffer(0)]],
    primitive_acceleration_structure           accelStructure    [[buffer(1)]],
    device   Vertex*                          vertices          [[buffer(2)]],
    device   uint*                            indices           [[buffer(3)]],
    constant Material*                        materials         [[buffer(4)]],
    texture2d<float, access::write>           colorOutput       [[texture(0)]],
    texture2d<float, access::write>           depthOutput       [[texture(1)]],
    texture2d<float, access::write>           motionOutput      [[texture(2)]]
) {
    if (tid.x >= uniforms.renderWidth || tid.y >= uniforms.renderHeight) return;

    float2 pixelSize = float2(1.0f / float(uniforms.renderWidth),
                              1.0f / float(uniforms.renderHeight));
    float2 uv = (float2(tid) + 0.5f) * pixelSize;

    float3 totalRadiance = float3(0.0f);
    float  totalDepth    = 0.0f;

    for (uint s = 0; s < uniforms.samplesPerPixel; s++) {
        RNG rng(tid, uniforms.frameIndex, s);

        // Jittered pixel coordinate for anti-aliasing
        float2 jitter = float2(uniforms.jitterX, uniforms.jitterY);
        float2 sampleUV = uv + (jitter + rng.next2() - 0.5f) * pixelSize;

        // Map to NDC [-1, 1]
        float2 ndc = sampleUV * 2.0f - 1.0f;
        ndc.y = -ndc.y; // flip Y for Metal convention

        // Generate primary ray from inverse view-projection
        float4 nearPoint = uniforms.inverseViewProjection * float4(ndc, 0.0f, 1.0f);
        float4 farPoint  = uniforms.inverseViewProjection * float4(ndc, 1.0f, 1.0f);
        nearPoint /= nearPoint.w;
        farPoint  /= farPoint.w;

        float3 rayOrigin    = nearPoint.xyz;
        float3 rayDirection = normalize(farPoint.xyz - nearPoint.xyz);

        // ── Path trace with multiple bounces ──

        float3 radiance   = float3(0.0f);
        float3 throughput  = float3(1.0f);
        float  primaryDepth = 1e10f;

        ray r;
        r.origin       = rayOrigin;
        r.direction    = rayDirection;
        r.min_distance = 0.001f;
        r.max_distance = 1e10f;

        intersector<triangle_data> intersector;
        intersector.accept_any_intersection(false);

        for (uint bounce = 0; bounce < uniforms.maxBounces; bounce++) {
            auto intersection = intersector.intersect(r, accelStructure);

            if (intersection.type == intersection_type::none) {
                // Sky: simple gradient
                float t = 0.5f * (r.direction.y + 1.0f);
                float3 skyColor = mix(float3(0.8f, 0.85f, 0.9f),
                                      float3(0.3f, 0.5f, 0.8f), t);
                radiance += throughput * skyColor * 0.8f;
                break;
            }

            float hitDistance = intersection.distance;

            if (bounce == 0) {
                primaryDepth = hitDistance;
            }

            // Get triangle data
            uint primitiveIndex = intersection.primitive_id;
            float2 barycentrics = intersection.triangle_barycentric_coord;

            uint i0 = indices[primitiveIndex * 3 + 0];
            uint i1 = indices[primitiveIndex * 3 + 1];
            uint i2 = indices[primitiveIndex * 3 + 2];

            Vertex v0 = vertices[i0];
            Vertex v1 = vertices[i1];
            Vertex v2 = vertices[i2];

            // Interpolate normal
            float3 n0 = float3(v0.normal);
            float3 n1 = float3(v1.normal);
            float3 n2 = float3(v2.normal);
            float3 hitNormal = normalize(
                n0 * (1.0f - barycentrics.x - barycentrics.y) +
                n1 * barycentrics.x +
                n2 * barycentrics.y
            );

            // Get material
            Material mat = materials[v0.materialIndex];
            float3 albedo = float3(mat.albedo);

            // Ambient hemisphere light for indoor scene base illumination
            if (bounce == 0) {
                float hemiLight = max(0.0f, dot(hitNormal, float3(0, 1, 0))) * 0.6f + 0.4f;
                radiance += albedo * hemiLight;
            }

            // Handle emissive surfaces (lights in Quake maps)
            if (mat.emissiveStrength > 0.0f) {
                radiance += throughput * albedo * mat.emissiveStrength;
                break;
            }

            // Lambertian BRDF: throughput *= albedo
            throughput *= albedo;

            // Russian roulette after bounce 3
            if (bounce >= 3) {
                float p = max(throughput.x, max(throughput.y, throughput.z));
                if (rng.next() > p) break;
                throughput /= p;
            }

            // Sample next direction (cosine-weighted hemisphere)
            float3 hitPoint = r.origin + r.direction * hitDistance;
            float3 newDir = cosineWeightedHemisphere(rng.next2(), hitNormal);

            r.origin    = hitPoint + hitNormal * 0.001f;
            r.direction = newDir;
        }

        totalRadiance += radiance;
        totalDepth    += primaryDepth;
    }

    float invSpp = 1.0f / float(uniforms.samplesPerPixel);
    float3 finalColor = totalRadiance * invSpp;
    float  finalDepth = totalDepth * invSpp;

    colorOutput.write(float4(finalColor, 1.0f), tid);
    depthOutput.write(float4(finalDepth, 0.0f, 0.0f, 1.0f), tid);

    // ── Motion vectors for MetalFX ──
    // Reproject current pixel through previous frame's VP matrix
    float2 ndcCurrent = uv * 2.0f - 1.0f;
    ndcCurrent.y = -ndcCurrent.y;
    float4 worldPos = uniforms.inverseViewProjection * float4(ndcCurrent, finalDepth / 1000.0f, 1.0f);
    worldPos /= worldPos.w;
    float4 prevClip = uniforms.previousViewProjection * worldPos;
    float2 prevUV = (prevClip.xy / prevClip.w) * 0.5f + 0.5f;
    prevUV.y = 1.0f - prevUV.y;
    float2 motion = (prevUV - uv) * float2(uniforms.renderWidth, uniforms.renderHeight);
    motionOutput.write(float4(motion, 0.0f, 1.0f), tid);
}
