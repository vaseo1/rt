import Metal
import simd

private struct BloomSettings {
    var threshold: Float = 1.1
    var softKnee: Float = 0.75
    var intensity: Float = 0.085
    var blurScale: Float = 1.35
}

private struct BloomUniforms {
    var sourceWidth: UInt32
    var sourceHeight: UInt32
    var outputWidth: UInt32
    var outputHeight: UInt32
    var bloomWidth: UInt32
    var bloomHeight: UInt32
    var threshold: Float
    var softKnee: Float
    var intensity: Float
    var blurScale: Float
    var directionX: Float
    var directionY: Float
    var padding0: Float
    var padding1: Float
}

// ─── GPU Uniforms (must match Common.metal layout exactly) ───────────────────

struct Uniforms {
    var currentViewProjection: float4x4
    var inverseViewProjection: float4x4
    var previousViewProjection: float4x4
    var cameraPosition: (Float, Float, Float)
    var frameIndex: UInt32
    var cameraRight: (Float, Float, Float)
    var accumulationCount: UInt32
    var cameraUp: (Float, Float, Float)
    var samplesPerPixel: UInt32
    var cameraForward: (Float, Float, Float)
    var maxBounces: UInt32
    var jitterX: Float
    var jitterY: Float
    var renderWidth: UInt32
    var renderHeight: UInt32
    var outputWidth: UInt32
    var outputHeight: UInt32
    var aperture: Float
    var focusDistance: Float
    var lightCount: UInt32
    var debugFlags: UInt32
    var environmentTint: (Float, Float, Float)
    var environmentIntensity: Float
    var environmentSunDirection: (Float, Float, Float)
    var environmentSunIntensity: Float
    var exposure: Float
    var exposurePadding0: Float
    var exposurePadding1: Float
    var exposurePadding2: Float
}

// ─── Path Tracer Pipeline ────────────────────────────────────────────────────
//
// Manages Metal compute pipeline states and dispatches the path tracing,
// accumulation, and tonemapping kernels.

class PathTracer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let bloomSettings = BloomSettings()

    private var pathTracePipeline: MTLComputePipelineState!
    private var accumulatePipeline: MTLComputePipelineState!
    private var measureExposurePipeline: MTLComputePipelineState!
    private var extractBloomPipeline: MTLComputePipelineState!
    private var blurBloomPipeline: MTLComputePipelineState!
    private var compositeBloomPipeline: MTLComputePipelineState!
    private var tonemapPipeline: MTLComputePipelineState!
    private let exposureReadbackBuffer: MTLBuffer

    // Render targets at current internal resolution
    private(set) var colorTexture: MTLTexture?
    private(set) var depthTexture: MTLTexture?
    private(set) var motionTexture: MTLTexture?
    private(set) var historyTexture: MTLTexture?
    private(set) var accumulatedTexture: MTLTexture?
    private(set) var bloomTexture: MTLTexture?
    private(set) var bloomScratchTexture: MTLTexture?
    private(set) var bloomCompositeTexture: MTLTexture?
    private(set) var tonemappedTexture: MTLTexture?

    private var currentRenderWidth: Int = 0
    private var currentRenderHeight: Int = 0
    private var outputWidth: Int = 0
    private var outputHeight: Int = 0

    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
        guard let exposureReadbackBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride,
                                                             options: .storageModeShared) else {
            fatalError("Failed to allocate exposure readback buffer")
        }
        self.exposureReadbackBuffer = exposureReadbackBuffer
        exposureReadbackBuffer.contents().assumingMemoryBound(to: Float.self).pointee = 0.18
        buildPipelines()
    }

    private func buildPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to load Metal library. Ensure .metal files are in the Xcode target.")
        }

        func makePipeline(_ name: String) -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                fatalError("Metal function '\(name)' not found")
            }
            do {
                return try device.makeComputePipelineState(function: function)
            } catch {
                fatalError("Failed to create pipeline '\(name)': \(error)")
            }
        }

        pathTracePipeline = makePipeline("pathTraceKernel")
        accumulatePipeline = makePipeline("accumulateKernel")
        measureExposurePipeline = makePipeline("measureExposureKernel")
        extractBloomPipeline = makePipeline("extractBloomKernel")
        blurBloomPipeline = makePipeline("blurBloomKernel")
        compositeBloomPipeline = makePipeline("compositeBloomKernel")
        tonemapPipeline = makePipeline("tonemapKernel")
    }

    func ensureTextures(renderWidth: Int, renderHeight: Int,
                        outputWidth: Int, outputHeight: Int) -> Bool {
        if renderWidth == currentRenderWidth && renderHeight == currentRenderHeight
            && outputWidth == self.outputWidth && outputHeight == self.outputHeight {
            return false
        }

        currentRenderWidth = renderWidth
        currentRenderHeight = renderHeight
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight

        func makeTexture(_ w: Int, _ h: Int, format: MTLPixelFormat,
                         usage: MTLTextureUsage = [.shaderRead, .shaderWrite]) -> MTLTexture {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: w, height: h, mipmapped: false)
            desc.usage = usage
            desc.storageMode = .private
            return device.makeTexture(descriptor: desc)!
        }

        colorTexture       = makeTexture(renderWidth, renderHeight, format: .rgba32Float)
        depthTexture        = makeTexture(renderWidth, renderHeight, format: .r32Float)
        motionTexture       = makeTexture(renderWidth, renderHeight, format: .rg16Float)
        historyTexture      = makeTexture(renderWidth, renderHeight, format: .rgba32Float)
        accumulatedTexture  = makeTexture(renderWidth, renderHeight, format: .rgba32Float)
        let bloomWidth = max(1, (outputWidth + 1) / 2)
        let bloomHeight = max(1, (outputHeight + 1) / 2)
        bloomTexture = makeTexture(bloomWidth, bloomHeight, format: .rgba16Float)
        bloomScratchTexture = makeTexture(bloomWidth, bloomHeight, format: .rgba16Float)
        bloomCompositeTexture = makeTexture(outputWidth, outputHeight, format: .rgba32Float)
        let tonemapDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: outputWidth, height: outputHeight, mipmapped: false)
        tonemapDesc.usage = [.shaderRead, .shaderWrite, .renderTarget]
        tonemapDesc.storageMode = .shared
        tonemappedTexture = device.makeTexture(descriptor: tonemapDesc)

        return true
    }

    func encodePathTrace(commandBuffer: MTLCommandBuffer,
                         uniforms: inout Uniforms,
                         accelStructure: MTLAccelerationStructure,
                         scene: SceneGeometry) {

        guard let colorTex = colorTexture,
              let depthTex = depthTexture,
              let motionTex = motionTexture,
              let vertexBuf = scene.vertexBuffer,
              let indexBuf = scene.indexBuffer,
              let materialBuf = scene.materialBuffer,
              let texDataBuf = scene.textureDataBuffer,
              let lightBuf = scene.lightBuffer else {
            return
        }

        let renderW = Int(uniforms.renderWidth)
        let renderH = Int(uniforms.renderHeight)

        // ── 1. Path Trace ──

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Path Trace"
            encoder.setComputePipelineState(pathTracePipeline)
            encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
            encoder.setAccelerationStructure(accelStructure, bufferIndex: 1)
            encoder.useResource(accelStructure, usage: .read)
            encoder.setBuffer(vertexBuf, offset: 0, index: 2)
            encoder.setBuffer(indexBuf, offset: 0, index: 3)
            encoder.setBuffer(materialBuf, offset: 0, index: 4)
            encoder.setBuffer(texDataBuf, offset: 0, index: 5)
            encoder.setBuffer(lightBuf, offset: 0, index: 6)
            encoder.setTexture(colorTex, index: 0)
            encoder.setTexture(depthTex, index: 1)
            encoder.setTexture(motionTex, index: 2)

            let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
            let gridSize = MTLSize(width: renderW, height: renderH, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }

    func encodeAccumulation(commandBuffer: MTLCommandBuffer,
                            uniforms: inout Uniforms) {
        guard let colorTex = colorTexture,
              let historyTex = historyTexture,
              let accumTex = accumulatedTexture else {
            return
        }

        let renderW = Int(uniforms.renderWidth)
        let renderH = Int(uniforms.renderHeight)

        // ── 2. Accumulate ──

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Accumulate"
            encoder.setComputePipelineState(accumulatePipeline)
            encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
            encoder.setTexture(colorTex, index: 0)
            encoder.setTexture(historyTex, index: 1)
            encoder.setTexture(accumTex, index: 2)

            let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
            let gridSize = MTLSize(width: renderW, height: renderH, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }

    func copyAccumulationToHistory(commandBuffer: MTLCommandBuffer) {
        guard let historyTex = historyTexture,
              let accumTex = accumulatedTexture else {
            return
        }

        // Copy accumulated → history for next frame
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "Copy History"
            blit.copy(from: accumTex, to: historyTex)
            blit.endEncoding()
        }
    }

    func encodeExposureMeasurement(commandBuffer: MTLCommandBuffer,
                                   sourceTexture: MTLTexture) {
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Measure Exposure"
            encoder.setComputePipelineState(measureExposurePipeline)
            encoder.setTexture(sourceTexture, index: 0)
            encoder.setBuffer(exposureReadbackBuffer, offset: 0, index: 0)

            let gridSize = MTLSize(width: 1, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: 1, height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }

    func readMeasuredAverageLuminance() -> Float {
        let value = exposureReadbackBuffer.contents().assumingMemoryBound(to: Float.self).pointee
        return max(value, 1e-4)
    }

    func encodeBloom(commandBuffer: MTLCommandBuffer,
                     sourceTexture: MTLTexture) -> MTLTexture? {
        guard bloomSettings.intensity > 0,
              let bloomTexture,
              let bloomScratchTexture,
              let bloomCompositeTexture else {
            return nil
        }

        var bloomUniforms = BloomUniforms(
            sourceWidth: UInt32(sourceTexture.width),
            sourceHeight: UInt32(sourceTexture.height),
            outputWidth: UInt32(bloomCompositeTexture.width),
            outputHeight: UInt32(bloomCompositeTexture.height),
            bloomWidth: UInt32(bloomTexture.width),
            bloomHeight: UInt32(bloomTexture.height),
            threshold: bloomSettings.threshold,
            softKnee: bloomSettings.softKnee,
            intensity: bloomSettings.intensity,
            blurScale: bloomSettings.blurScale,
            directionX: 0,
            directionY: 0,
            padding0: 0,
            padding1: 0
        )

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Bloom Extract"
            encoder.setComputePipelineState(extractBloomPipeline)
            encoder.setBytes(&bloomUniforms, length: MemoryLayout<BloomUniforms>.size, index: 0)
            encoder.setTexture(sourceTexture, index: 0)
            encoder.setTexture(bloomTexture, index: 1)

            let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
            let gridSize = MTLSize(width: bloomTexture.width, height: bloomTexture.height, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        bloomUniforms.directionX = 1
        bloomUniforms.directionY = 0

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Bloom Blur Horizontal"
            encoder.setComputePipelineState(blurBloomPipeline)
            encoder.setBytes(&bloomUniforms, length: MemoryLayout<BloomUniforms>.size, index: 0)
            encoder.setTexture(bloomTexture, index: 0)
            encoder.setTexture(bloomScratchTexture, index: 1)

            let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
            let gridSize = MTLSize(width: bloomScratchTexture.width, height: bloomScratchTexture.height, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        bloomUniforms.directionX = 0
        bloomUniforms.directionY = 1

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Bloom Blur Vertical"
            encoder.setComputePipelineState(blurBloomPipeline)
            encoder.setBytes(&bloomUniforms, length: MemoryLayout<BloomUniforms>.size, index: 0)
            encoder.setTexture(bloomScratchTexture, index: 0)
            encoder.setTexture(bloomTexture, index: 1)

            let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
            let gridSize = MTLSize(width: bloomTexture.width, height: bloomTexture.height, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Bloom Composite"
            encoder.setComputePipelineState(compositeBloomPipeline)
            encoder.setBytes(&bloomUniforms, length: MemoryLayout<BloomUniforms>.size, index: 0)
            encoder.setTexture(sourceTexture, index: 0)
            encoder.setTexture(bloomTexture, index: 1)
            encoder.setTexture(bloomCompositeTexture, index: 2)

            let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
            let gridSize = MTLSize(width: bloomCompositeTexture.width, height: bloomCompositeTexture.height, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        return bloomCompositeTexture
    }

    func encodeTonemap(commandBuffer: MTLCommandBuffer,
                       uniforms: inout Uniforms,
                       sourceTexture: MTLTexture) {
        guard let tonemapTex = tonemappedTexture else {
            return
        }

        let outW = Int(uniforms.outputWidth)
        let outH = Int(uniforms.outputHeight)

        // ── 3. Tonemap ──

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Tonemap"
            encoder.setComputePipelineState(tonemapPipeline)
            encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
            encoder.setTexture(sourceTexture, index: 0)
            encoder.setTexture(tonemapTex, index: 1)

            let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
            let gridSize = MTLSize(width: outW, height: outH, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }
}
