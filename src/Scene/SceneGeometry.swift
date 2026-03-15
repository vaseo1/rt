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
    var albedo: (Float, Float, Float)    // packed_float3
    var emissiveStrength: Float
}

// ─── Scene Geometry ──────────────────────────────────────────────────────────
//
// Holds Metal buffers for the loaded BSP geometry.

class SceneGeometry {
    let device: MTLDevice
    private(set) var vertexBuffer: MTLBuffer?
    private(set) var indexBuffer: MTLBuffer?
    private(set) var materialBuffer: MTLBuffer?

    private(set) var vertexCount: Int = 0
    private(set) var indexCount: Int = 0
    private(set) var triangleCount: Int = 0

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

        var gpuMaterials = bspData.materials.map { m in
            GPUMaterial(
                albedo: (m.albedo.x, m.albedo.y, m.albedo.z),
                emissiveStrength: m.emissiveStrength
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

        print("[Scene] GPU buffers created: \(vertexCount) verts, \(triangleCount) tris")
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
