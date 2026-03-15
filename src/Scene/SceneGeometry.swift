import Metal
import simd

// ─── GPU-side vertex/material layout (matches Common.metal) ──────────────────

struct GPUVertex {
    var position: (Float, Float, Float)  // packed_float3
    var normal: (Float, Float, Float)    // packed_float3
    var uv: (Float, Float)              // packed_float2
    var materialIndex: UInt32
}

struct GPUMaterial {
    var albedo: (Float, Float, Float)
    var emissiveStrength: Float
    var emissiveColor: (Float, Float, Float)
    var textureOffset: UInt32
    var textureWidth: UInt32
    var textureHeight: UInt32
    var _pad0: UInt32 = 0
    var _pad1: UInt32 = 0
}

struct GPULight {
    var v0: (Float, Float, Float)
    var area: Float
    var edge1: (Float, Float, Float)
    var _pad0: Float = 0
    var edge2: (Float, Float, Float)
    var _pad1: Float = 0
    var emission: (Float, Float, Float)
    var _pad2: Float = 0
    var normal: (Float, Float, Float)
    var _pad3: Float = 0
}

// ─── Scene Geometry ──────────────────────────────────────────────────────────
//
// Holds Metal buffers for the loaded BSP geometry.

class SceneGeometry {
    let device: MTLDevice
    private(set) var vertexBuffer: MTLBuffer?
    private(set) var indexBuffer: MTLBuffer?
    private(set) var materialBuffer: MTLBuffer?
    private(set) var textureDataBuffer: MTLBuffer?
    private(set) var lightBuffer: MTLBuffer?

    private(set) var vertexCount: Int = 0
    private(set) var indexCount: Int = 0
    private(set) var triangleCount: Int = 0
    private(set) var lightCount: Int = 0

    init(device: MTLDevice) {
        self.device = device
    }

    func loadFromBSP(_ bspData: BSPData) {
        // Convert parsed vertices to GPU layout
        var gpuVertices = bspData.vertices.map { v in
            GPUVertex(
                position: (v.position.x, v.position.y, v.position.z),
                normal: (v.normal.x, v.normal.y, v.normal.z),
                uv: (v.uv.x, v.uv.y),
                materialIndex: v.materialIndex
            )
        }

        // Build texture atlas: pack all material textures into a flat RGBA buffer
        var texturePixels: [UInt8] = []
        var textureOffsets: [(offset: UInt32, width: UInt32, height: UInt32)] = []
        for mat in bspData.materials {
            if let pixels = mat.texturePixels, mat.textureWidth > 0, mat.textureHeight > 0 {
                let offset = UInt32(texturePixels.count / 4) // in uchar4 units
                textureOffsets.append((offset, UInt32(mat.textureWidth), UInt32(mat.textureHeight)))
                texturePixels.append(contentsOf: pixels)
            } else {
                textureOffsets.append((UInt32.max, 0, 0))
            }
        }

        var gpuMaterials = bspData.materials.enumerated().map { (i, m) in
            let texInfo = i < textureOffsets.count ? textureOffsets[i] : (UInt32.max, UInt32(0), UInt32(0))
            return GPUMaterial(
                albedo: (m.albedo.x, m.albedo.y, m.albedo.z),
                emissiveStrength: m.emissiveStrength,
                emissiveColor: (m.emissiveColor.x, m.emissiveColor.y, m.emissiveColor.z),
                textureOffset: texInfo.0,
                textureWidth: texInfo.1,
                textureHeight: texInfo.2
            )
        }

        var indices = bspData.indices

        vertexCount = gpuVertices.count
        indexCount = indices.count
        triangleCount = indexCount / 3

        vertexBuffer = device.makeBuffer(
            bytes: &gpuVertices,
            length: MemoryLayout<GPUVertex>.stride * gpuVertices.count,
            options: .storageModeShared
        )

        indexBuffer = device.makeBuffer(
            bytes: &indices,
            length: MemoryLayout<UInt32>.stride * indices.count,
            options: .storageModeShared
        )

        materialBuffer = device.makeBuffer(
            bytes: &gpuMaterials,
            length: MemoryLayout<GPUMaterial>.stride * gpuMaterials.count,
            options: .storageModeShared
        )

        // Create texture data buffer
        if !texturePixels.isEmpty {
            textureDataBuffer = device.makeBuffer(
                bytes: &texturePixels,
                length: texturePixels.count,
                options: .storageModeShared
            )
        }
        if textureDataBuffer == nil {
            textureDataBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
        }

        // Build light list from emissive triangles
        var lights: [GPULight] = []
        for tri in 0..<triangleCount {
            let i0 = Int(indices[tri * 3 + 0])
            let i1 = Int(indices[tri * 3 + 1])
            let i2 = Int(indices[tri * 3 + 2])

            let v0 = gpuVertices[i0]
            let matIdx = Int(v0.materialIndex)
            guard matIdx < bspData.materials.count else { continue }
            let mat = bspData.materials[matIdx]
            guard mat.emissiveStrength > 0 else { continue }

            let p0 = SIMD3<Float>(v0.position.0, v0.position.1, v0.position.2)
            let p1 = SIMD3<Float>(gpuVertices[i1].position.0, gpuVertices[i1].position.1, gpuVertices[i1].position.2)
            let p2 = SIMD3<Float>(gpuVertices[i2].position.0, gpuVertices[i2].position.1, gpuVertices[i2].position.2)

            let e1 = p1 - p0
            let e2 = p2 - p0
            let crossP = cross(e1, e2)
            let area = length(crossP) * 0.5
            guard area > 0.001 else { continue }

            let v0Normal = SIMD3<Float>(v0.normal.0, v0.normal.1, v0.normal.2)
            let v1Normal = SIMD3<Float>(gpuVertices[i1].normal.0, gpuVertices[i1].normal.1, gpuVertices[i1].normal.2)
            let v2Normal = SIMD3<Float>(gpuVertices[i2].normal.0, gpuVertices[i2].normal.1, gpuVertices[i2].normal.2)
            let shadingNormal = normalize(v0Normal + v1Normal + v2Normal)

            var n = normalize(crossP)
            if dot(n, shadingNormal) < 0 {
                n = -n
            }

            let emission = mat.emissiveColor * mat.emissiveStrength

            lights.append(GPULight(
                v0: (p0.x, p0.y, p0.z), area: area,
                edge1: (e1.x, e1.y, e1.z),
                edge2: (e2.x, e2.y, e2.z),
                emission: (emission.x, emission.y, emission.z),
                normal: (n.x, n.y, n.z)
            ))
        }

        lightCount = lights.count
        if !lights.isEmpty {
            lightBuffer = device.makeBuffer(
                bytes: &lights,
                length: MemoryLayout<GPULight>.stride * lights.count,
                options: .storageModeShared
            )
        }
        if lightBuffer == nil {
            lightBuffer = device.makeBuffer(length: MemoryLayout<GPULight>.stride, options: .storageModeShared)
        }

        print("[Scene] GPU buffers: \(vertexCount) verts, \(triangleCount) tris, \(lightCount) lights")
        if !texturePixels.isEmpty {
            print("[Scene] Texture atlas: \(texturePixels.count / 4) pixels (\(texturePixels.count / 1024) KB)")
        }
    }

    // Create a position-only buffer for acceleration structure building
    // (MTLAccelerationStructure wants a simple float3 position buffer)
    func makePositionBuffer() -> MTLBuffer? {
        guard let vb = vertexBuffer else { return nil }
        let verts = vb.contents().bindMemory(to: GPUVertex.self, capacity: vertexCount)

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(vertexCount)
        for i in 0..<vertexCount {
            let v = verts[i]
            positions.append(SIMD3<Float>(v.position.0, v.position.1, v.position.2))
        }

        return device.makeBuffer(
            bytes: &positions,
            length: MemoryLayout<SIMD3<Float>>.stride * positions.count,
            options: .storageModeShared
        )
    }
}
