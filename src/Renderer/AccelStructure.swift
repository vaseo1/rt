import Metal

// ─── Acceleration Structure Builder ──────────────────────────────────────────
//
// Builds a Metal ray tracing acceleration structure (BVH) from triangle geometry.
// Uses primitive acceleration structure with triangle geometry descriptors.
// Static scene → .fastTrace usage, built once + compacted.

class AccelStructureBuilder {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private(set) var accelerationStructure: MTLAccelerationStructure?

    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
    }

    func build(scene: SceneGeometry) {
        guard let positionBuffer = scene.makePositionBuffer(),
              let indexBuffer = scene.indexBuffer else {
            print("[Accel] ERROR: Missing geometry buffers")
            return
        }

        // Describe the triangle geometry
        let geometryDesc = MTLAccelerationStructureTriangleGeometryDescriptor()
        geometryDesc.vertexBuffer = positionBuffer
        geometryDesc.vertexStride = MemoryLayout<SIMD3<Float>>.stride
        geometryDesc.vertexFormat = .float3
        geometryDesc.indexBuffer = indexBuffer
        geometryDesc.indexType = .uint32
        geometryDesc.triangleCount = scene.triangleCount

        // Primitive acceleration structure (bottom-level)
        let accelDesc = MTLPrimitiveAccelerationStructureDescriptor()
        accelDesc.geometryDescriptors = [geometryDesc]

        // Query sizes needed
        let sizes = device.accelerationStructureSizes(descriptor: accelDesc)

        // Allocate the acceleration structure
        guard let accelStructure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize) else {
            print("[Accel] ERROR: Failed to allocate acceleration structure")
            return
        }

        // Allocate scratch buffer
        guard let scratchBuffer = device.makeBuffer(length: sizes.buildScratchBufferSize,
                                                     options: .storageModePrivate) else {
            print("[Accel] ERROR: Failed to allocate scratch buffer")
            return
        }

        // Build
        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeAccelerationStructureCommandEncoder() else {
            print("[Accel] ERROR: Failed to create command encoder")
            return
        }

        encoder.build(accelerationStructure: accelStructure,
                      descriptor: accelDesc,
                      scratchBuffer: scratchBuffer,
                      scratchBufferOffset: 0)
        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        // Compact the structure to reduce memory usage
        if let compacted = compact(original: accelStructure) {
            self.accelerationStructure = compacted
        } else {
            self.accelerationStructure = accelStructure
        }

        let sizeMB = Double(self.accelerationStructure!.size) / (1024 * 1024)
        print(String(format: "[Accel] Built: %.2f MB", sizeMB))
    }

    private func compact(original: MTLAccelerationStructure) -> MTLAccelerationStructure? {
        // Query compacted size
        let sizeBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                           options: .storageModeShared)!

        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeAccelerationStructureCommandEncoder() else {
            return nil
        }

        encoder.writeCompactedSize(accelerationStructure: original,
                                   buffer: sizeBuffer,
                                   offset: 0)
        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        let compactedSize = sizeBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee

        guard compactedSize > 0,
              let compacted = device.makeAccelerationStructure(size: Int(compactedSize)) else {
            return nil
        }

        guard let copyCmd = commandQueue.makeCommandBuffer(),
              let copyEncoder = copyCmd.makeAccelerationStructureCommandEncoder() else {
            return nil
        }

        copyEncoder.copyAndCompact(sourceAccelerationStructure: original,
                                   destinationAccelerationStructure: compacted)
        copyEncoder.endEncoding()
        copyCmd.commit()
        copyCmd.waitUntilCompleted()

        return compacted
    }
}
