#include <metal_stdlib>
#include <metal_raytracing>
#include "Common.metal"

using namespace metal;
using namespace raytracing;

// ─── Texture sampling with bilinear filtering ────────────────────────────────

inline float3 sampleTextureBilinear(device uchar4* texData, uint offset,
                                    uint width, uint height, float2 uv) {
    float u = fract(uv.x);
    float v = fract(uv.y);
    float fu = u * float(width);
    float fv = v * float(height);

    int ix = int(fu);
    int iy = int(fv);
    int ix1 = (ix + 1) % int(width);
    int iy1 = (iy + 1) % int(height);
    float fx = fu - float(ix);
    float fy = fv - float(iy);

    float3 c00 = pow(float3(texData[offset + uint(iy)  * width + uint(ix) ].xyz) / 255.0f, float3(2.2f));
    float3 c10 = pow(float3(texData[offset + uint(iy)  * width + uint(ix1)].xyz) / 255.0f, float3(2.2f));
    float3 c01 = pow(float3(texData[offset + uint(iy1) * width + uint(ix) ].xyz) / 255.0f, float3(2.2f));
    float3 c11 = pow(float3(texData[offset + uint(iy1) * width + uint(ix1)].xyz) / 255.0f, float3(2.2f));

    return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

// ─── Path Tracing Compute Kernel ─────────────────────────────────────────────

kernel void pathTraceKernel(
    uint2                                     tid               [[thread_position_in_grid]],
    constant Uniforms&                        uniforms          [[buffer(0)]],
    primitive_acceleration_structure           accelStructure    [[buffer(1)]],
    device   Vertex*                          vertices          [[buffer(2)]],
    device   uint*                            indices           [[buffer(3)]],
    constant Material*                        materials         [[buffer(4)]],
    device   uchar4*                          textureData       [[buffer(5)]],
    device   Light*                           lights            [[buffer(6)]],
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

        float3 cameraPosition = float3(uniforms.cameraPosition);
        float3 cameraRight = normalize(float3(uniforms.cameraRight));
        float3 cameraUp = normalize(float3(uniforms.cameraUp));
        float3 cameraForward = normalize(float3(uniforms.cameraForward));

        float3 rayOrigin    = cameraPosition;
        float3 rayDirection = normalize(farPoint.xyz - cameraPosition);

        // ── Depth of Field (thin lens model) ──
        if (uniforms.aperture > 0.0f) {
            float focusT = uniforms.focusDistance / max(1e-4f, dot(rayDirection, cameraForward));
            float3 focusPoint = cameraPosition + rayDirection * focusT;
            float2 lensSample = concentricSampleDisk(rng.next2()) * uniforms.aperture;

            rayOrigin = cameraPosition + cameraRight * lensSample.x + cameraUp * lensSample.y;
            rayDirection = normalize(focusPoint - rayOrigin);
        }

        // ── Path trace with multiple bounces ──

        float3 radiance   = float3(0.0f);
        float3 throughput  = float3(1.0f);
        float  primaryDepth = 1e10f;

        ray r;
        r.origin       = rayOrigin;
        r.direction    = rayDirection;
        r.min_distance = 0.001f;
        r.max_distance = 1e10f;

        intersector<triangle_data> inter;
        inter.accept_any_intersection(false);



        for (uint bounce = 0; bounce < uniforms.maxBounces; bounce++) {
            auto intersection = inter.intersect(r, accelStructure);

            if (intersection.type == intersection_type::none) {
                // Sky gradient
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

            float bw = 1.0f - barycentrics.x - barycentrics.y;

            // Interpolate normal
            float3 hitNormal = normalize(
                float3(v0.normal) * bw +
                float3(v1.normal) * barycentrics.x +
                float3(v2.normal) * barycentrics.y
            );

            float3 hitPoint = r.origin + r.direction * hitDistance;

            // Get material
            Material mat = materials[v0.materialIndex];
            float3 albedo = float3(mat.albedo);

            // Sample texture if available
            if (mat.textureWidth > 0 && mat.textureOffset != 0xFFFFFFFF) {
                float2 hitUV = float2(v0.uv) * bw +
                               float2(v1.uv) * barycentrics.x +
                               float2(v2.uv) * barycentrics.y;
                albedo = sampleTextureBilinear(textureData, mat.textureOffset,
                                              mat.textureWidth, mat.textureHeight, hitUV);
            }

            // Handle emissive surfaces
            if (mat.emissiveStrength > 0.0f) {
                // With NEE: only count direct visibility (bounce 0)
                // Without NEE: count on all bounces
                if (bounce == 0 || uniforms.lightCount == 0) {
                    radiance += throughput * float3(mat.emissiveColor) * mat.emissiveStrength;
                }
                break;
            }

            // ── Next Event Estimation (direct light sampling) ──
            if (uniforms.lightCount > 0) {
                uint lightIdx = min(uint(rng.next() * float(uniforms.lightCount)),
                                    uniforms.lightCount - 1);
                Light light = lights[lightIdx];

                // Sample random point on light triangle
                float2 uLight = rng.next2();
                if (uLight.x + uLight.y > 1.0f) {
                    uLight = 1.0f - uLight;
                }
                float3 lightPoint = float3(light.v0) +
                                    uLight.x * float3(light.edge1) +
                                    uLight.y * float3(light.edge2);

                float3 toLight = lightPoint - hitPoint;
                float lightDist = length(toLight);
                float3 lightDir = toLight / lightDist;

                float cosTheta = dot(hitNormal, lightDir);
                float cosLight = abs(dot(float3(light.normal), lightDir));

                if (cosTheta > 0.0f && cosLight > 0.0f) {
                    // Shadow ray
                    ray shadowRay;
                    shadowRay.origin = hitPoint + hitNormal * 0.001f;
                    shadowRay.direction = lightDir;
                    shadowRay.min_distance = 0.001f;
                    shadowRay.max_distance = lightDist - 0.002f;

                    intersector<triangle_data> shadowInter;
                    shadowInter.accept_any_intersection(true);
                    auto shadowHit = shadowInter.intersect(shadowRay, accelStructure);

                    if (shadowHit.type == intersection_type::none) {
                        float3 emission = float3(light.emission);
                        float lightArea = light.area;
                        float pdf = 1.0f / (float(uniforms.lightCount) * lightArea);
                        float G = cosTheta * cosLight / (lightDist * lightDist);
                        float3 brdf = albedo / M_PI_F;
                        radiance += throughput * emission * brdf * G / pdf;
                    }
                }
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
