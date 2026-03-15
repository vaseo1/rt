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

inline float3 perturbLiquidNormal(float3 normal, float3 hitPoint, float2 uv, float roughness) {
    float3 tangent;
    float3 bitangent;
    buildTangentFrame(normal, tangent, bitangent);

    float wave0 = sin(dot(hitPoint.xz, float2(0.065f, 0.041f)) + uv.x * 18.0f + uv.y * 11.0f);
    float wave1 = cos(dot(hitPoint.xz, float2(-0.038f, 0.057f)) - uv.x * 9.0f + uv.y * 14.0f);
    float wave2 = sin(dot(hitPoint.xz, float2(0.022f, -0.071f)) + uv.x * 5.0f - uv.y * 7.0f);

    float waveStrength = mix(0.035f, 0.12f, saturate(roughness * 6.0f));
    float3 perturbed = normal +
                       tangent * (wave0 * waveStrength + wave2 * (waveStrength * 0.5f)) +
                       bitangent * (wave1 * waveStrength);
    return normalize(perturbed);
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
    texture2d<float, access::write>           motionOutput      [[texture(2)]],
    texture2d<float, access::write>           normalOutput      [[texture(3)]],
    texture2d<float, access::write>           albedoOutput      [[texture(4)]]
) {
    if (tid.x >= uniforms.renderWidth || tid.y >= uniforms.renderHeight) return;

    float2 pixelSize = float2(1.0f / float(uniforms.renderWidth),
                              1.0f / float(uniforms.renderHeight));
    float2 uv = (float2(tid) + 0.5f) * pixelSize;

    float3 totalRadiance = float3(0.0f);
    float  totalDepth    = 0.0f;
    float2 totalMotion   = float2(0.0f);
    float3 totalNormal   = float3(0.0f);
    float3 totalAlbedo   = float3(0.0f);
    float  totalRoughness = 0.0f;
    uint   motionSamples = 0;
    uint   surfaceSamples = 0;

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

        float3 radiance = float3(0.0f);
        float3 throughput = float3(1.0f);
        float  primaryDepth = 1.0f;
        float2 primaryMotion = float2(0.0f);
        float3 primaryNormal = float3(0.0f);
        float3 primaryAlbedo = float3(0.0f);
        float  primaryRoughness = 0.0f;
        bool   hasPrimaryHit = false;
        bool   previousBounceWasSpecular = false;

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
                radiance += throughput * sampleProceduralEnvironment(r.direction, uniforms);
                break;
            }

            float hitDistance = intersection.distance;

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
            bool frontFace = dot(hitNormal, r.direction) < 0.0f;
            if (!frontFace) {
                hitNormal = -hitNormal;
            }

            if (bounce == 0) {
                float4 currentClip = uniforms.currentViewProjection * float4(hitPoint, 1.0f);
                float4 prevClip = uniforms.previousViewProjection * float4(hitPoint, 1.0f);

                if (abs(currentClip.w) > 1e-5f) {
                    float currentNdcDepth = currentClip.z / currentClip.w;
                    primaryDepth = saturate(currentNdcDepth * 0.5f + 0.5f);
                }

                if (abs(currentClip.w) > 1e-5f && abs(prevClip.w) > 1e-5f) {
                    float2 currentUV = (currentClip.xy / currentClip.w) * 0.5f + 0.5f;
                    float2 prevUV = (prevClip.xy / prevClip.w) * 0.5f + 0.5f;
                    currentUV.y = 1.0f - currentUV.y;
                    prevUV.y = 1.0f - prevUV.y;
                    primaryMotion = (prevUV - currentUV) * float2(uniforms.renderWidth, uniforms.renderHeight);
                }

                hasPrimaryHit = true;
            }

            // Get material
            Material mat = materials[v0.materialIndex];
            float3 albedo = float3(mat.albedo);
            float2 hitUV = float2(v0.uv) * bw +
                           float2(v1.uv) * barycentrics.x +
                           float2(v2.uv) * barycentrics.y;

            // Sample texture if available
            if (mat.textureWidth > 0 && mat.textureOffset != 0xFFFFFFFF) {
                albedo = sampleTextureBilinear(textureData, mat.textureOffset,
                                              mat.textureWidth, mat.textureHeight, hitUV);
            }

            float roughness = clamp(mat.roughness, 0.045f, 1.0f);
            float metallic = saturate(mat.metallic);
            float transmissive = saturate(mat.transmissive);
            float ior = max(mat.ior, 1.0f);
            float3 viewDir = -r.direction;
            float3 emission = float3(mat.emissiveColor) * mat.emissiveStrength;
            bool highlightLiquid = ((uniforms.debugFlags & 1u) != 0u) &&
                                   (mat.surfaceType == SurfaceTypeLiquid);
            if (highlightLiquid) {
                albedo = float3(1.0f, 0.0f, 1.0f);
                roughness = 0.85f;
                metallic = 0.0f;
                transmissive = 0.0f;
                ior = 1.0f;
                emission = float3(5.0f, 0.0f, 5.0f);
            }
            bool isTransmissive = transmissive > 0.0f;

            if (bounce == 0) {
                primaryNormal = hitNormal;
                primaryAlbedo = saturate(albedo);
                primaryRoughness = roughness;
            }

            if (highlightLiquid) {
                radiance += throughput * emission;
                break;
            }

            // Handle emissive surfaces
            if (mat.emissiveStrength > 0.0f) {
                if (bounce == 0 || previousBounceWasSpecular || uniforms.lightCount == 0) {
                    radiance += throughput * emission;
                }
                break;
            }

            float ndotv = saturate(dot(hitNormal, viewDir));
            if (ndotv <= 0.0f) {
                break;
            }

            // ── Next Event Estimation (direct light sampling) ──
            if (uniforms.lightCount > 0 && !isTransmissive) {
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
                        float3 brdf = evaluateCookTorranceBRDF(albedo, roughness, metallic,
                                                              ior, hitNormal, viewDir, lightDir);
                        radiance += throughput * emission * brdf * G / pdf;
                    }
                }
            }

            if (isTransmissive) {
                float3 liquidNormal = perturbLiquidNormal(hitNormal, hitPoint, hitUV, roughness);
                if (dot(liquidNormal, viewDir) <= 0.0f) {
                    liquidNormal = hitNormal;
                }

                float liquidRoughness = clamp(roughness, 0.04f, 0.18f);
                float3 microNormal = sampleGGXHalfVector(liquidNormal, liquidRoughness, rng.next2());
                if (dot(microNormal, viewDir) <= 0.0f) {
                    microNormal = liquidNormal;
                }

                float etaI = frontFace ? 1.0f : ior;
                float etaT = frontFace ? ior : 1.0f;
                float etaRatio = etaI / etaT;
                float liquidNdotV = saturate(dot(microNormal, viewDir));
                float reflectance = dielectricReflectance(liquidNdotV, etaI, etaT);
                float3 reflectedDir = reflect(r.direction, microNormal);
                float3 refractedDir = refract(r.direction, microNormal, etaRatio);
                bool hasRefraction = dot(refractedDir, refractedDir) > 1e-6f;

                bool sampleReflection = !hasRefraction || rng.next() < reflectance;
                float3 newDir = sampleReflection ? normalize(reflectedDir) : normalize(refractedDir);

                if (!sampleReflection) {
                    float transmissionChance = max(1.0f - reflectance, 1e-3f);
                    float3 transmissionTint = transmissionTintFromAlbedo(albedo);
                    throughput *= (transmissionTint * transmissive) / transmissionChance;
                }

                previousBounceWasSpecular = true;

                if (bounce >= 3) {
                    float p = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.95f);
                    if (rng.next() > p) break;
                    throughput /= p;
                }

                float3 normalOffset = dot(newDir, hitNormal) >= 0.0f ? hitNormal : -hitNormal;
                r.origin = hitPoint + normalOffset * 0.001f;
                r.direction = newDir;
                continue;
            }

            float3 Fview = fresnelSchlick(ndotv, materialF0(albedo, metallic, ior));
            float diffuseWeight = max((1.0f - metallic) * (1.0f - transmissive), 0.0f);
            float specularWeight = max(luminance(Fview), 0.001f);
            float specularChance = 1.0f;
            float diffuseChance = 0.0f;

            if (diffuseWeight > 0.0f) {
                specularChance = specularWeight / (specularWeight + diffuseWeight);
                specularChance = clamp(specularChance, 0.12f, 0.88f);
                diffuseChance = 1.0f - specularChance;
            }

            bool sampleSpecular = diffuseChance <= 0.0f || rng.next() < specularChance;
            float3 newDir;
            if (sampleSpecular) {
                float3 halfVector = sampleGGXHalfVector(hitNormal, roughness, rng.next2());
                newDir = normalize(reflect(-viewDir, halfVector));
            } else {
                newDir = cosineWeightedHemisphere(rng.next2(), hitNormal);
            }

            float ndotl = saturate(dot(hitNormal, newDir));
            if (ndotl <= 0.0f) {
                break;
            }

            float pdfDiffuse = cosineHemispherePdf(ndotl);
            float pdfSpecular = ggxPdf(hitNormal, viewDir, newDir, roughness);
            float totalPdf = diffuseChance * pdfDiffuse + specularChance * pdfSpecular;
            if (totalPdf <= 1e-6f) {
                break;
            }

            float3 brdf = evaluateCookTorranceBRDF(albedo, roughness, metallic,
                                                  ior, hitNormal, viewDir, newDir);
            throughput *= brdf * (ndotl / totalPdf);
            previousBounceWasSpecular = sampleSpecular && roughness < 0.2f;

            // Russian roulette after bounce 3
            if (bounce >= 3) {
                float p = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.95f);
                if (rng.next() > p) break;
                throughput /= p;
            }

            r.origin    = hitPoint + hitNormal * 0.001f;
            r.direction = newDir;
        }

        totalRadiance += radiance;
        totalDepth    += primaryDepth;
        if (hasPrimaryHit) {
            totalMotion += primaryMotion;
            motionSamples += 1;
            totalNormal += primaryNormal;
            totalAlbedo += primaryAlbedo;
            totalRoughness += primaryRoughness;
            surfaceSamples += 1;
        }
    }

    float invSpp = 1.0f / float(uniforms.samplesPerPixel);
    float3 finalColor = totalRadiance * invSpp;
    float  finalDepth = totalDepth * invSpp;
    float2 finalMotion = motionSamples > 0 ? totalMotion / float(motionSamples) : float2(0.0f);
    float3 finalNormal = surfaceSamples > 0 ? normalize(totalNormal / float(surfaceSamples)) : float3(0.0f);
    float3 finalAlbedo = surfaceSamples > 0 ? totalAlbedo / float(surfaceSamples) : float3(0.0f);
    float  finalRoughness = surfaceSamples > 0 ? totalRoughness / float(surfaceSamples) : 0.0f;
    float  hasSurface = surfaceSamples > 0 ? 1.0f : 0.0f;

    colorOutput.write(float4(finalColor, 1.0f), tid);
    depthOutput.write(float4(finalDepth, 0.0f, 0.0f, 1.0f), tid);
    motionOutput.write(float4(finalMotion, 0.0f, 1.0f), tid);
    normalOutput.write(float4(finalNormal, hasSurface), tid);
    albedoOutput.write(float4(finalAlbedo, finalRoughness), tid);
}
