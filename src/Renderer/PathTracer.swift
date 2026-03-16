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
    var previousJitterX: Float
    var previousJitterY: Float
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
    private let svgfStepWidths: [UInt32] = [1, 2]

    private var pathTracePipeline: MTLComputePipelineState!
    private var accumulatePipeline: MTLComputePipelineState!
    private var accumulateClampedPipeline: MTLComputePipelineState!
    private var svgfTemporalPipeline: MTLComputePipelineState!
    private var svgfATrousPipeline: MTLComputePipelineState!
    private var svgfRemodPipeline: MTLComputePipelineState!
    private var svgfDemodPipeline: MTLComputePipelineState!
    private var postATrousPipeline: MTLComputePipelineState!
    private var crossBilateralPipeline: MTLComputePipelineState!
    private var nlmPipeline: MTLComputePipelineState!
    private var clearR16Pipeline: MTLComputePipelineState!
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
    private(set) var normalTexture: MTLTexture?
    private(set) var albedoTexture: MTLTexture?
    private(set) var historyTexture: MTLTexture?
    private(set) var accumulatedTexture: MTLTexture?
    private(set) var svgfTemporalTexture: MTLTexture?
    private(set) var svgfMomentsTexture: MTLTexture?
    private(set) var svgfHistoryMomentsTexture: MTLTexture?
    private(set) var svgfHistoryLengthTexture: MTLTexture?
    private(set) var svgfHistoryLengthScratchTexture: MTLTexture?
    private(set) var svgfHistoryDepthTexture: MTLTexture?
    private(set) var svgfHistoryNormalTexture: MTLTexture?
    private(set) var svgfHistoryAlbedoTexture: MTLTexture?
    private(set) var svgfPingTexture: MTLTexture?
    private(set) var svgfPongTexture: MTLTexture?
    private(set) var svgfFilteredTexture: MTLTexture?
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

    private struct SVGFATrousUniforms {
        var renderWidth: UInt32
        var renderHeight: UInt32
        var stepWidth: UInt32
        var passIndex: UInt32
        var colorPhiScale: Float
        var normalPhi: Float
        var depthPhi: Float
        var albedoPhi: Float
    }

    private struct PostFilterUniforms {
        var renderWidth: UInt32
        var renderHeight: UInt32
        var stepWidth: UInt32
        var passIndex: UInt32
        var colorPhiScale: Float
        var normalPhi: Float
        var depthPhi: Float
        var albedoPhi: Float
        var accumulationCount: UInt32
        var _pad0: UInt32
        var _pad1: UInt32
        var _pad2: UInt32
    }

    private struct CrossBilateralUniforms {
        var renderWidth: UInt32
        var renderHeight: UInt32
        var radius: Int32
        var accumulationCount: UInt32
        var sigmaSpatial: Float
        var sigmaColor: Float
        var depthPhi: Float
        var normalPhi: Float
        var albedoPhi: Float
        var _pad0: Float
        var _pad1: Float
        var _pad2: Float
    }

    private struct NLMUniforms {
        var renderWidth: UInt32
        var renderHeight: UInt32
        var searchRadius: Int32
        var patchRadius: Int32
        var h: Float
        var depthRejectThreshold: Float
        var normalRejectDot: Float
        var albedoRejectDelta: Float
        var accumulationCount: UInt32
        var _pad0: UInt32
        var _pad1: UInt32
        var _pad2: UInt32
    }

    private let eawStepWidths: [UInt32] = [1, 2, 4, 8, 16]

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
        accumulateClampedPipeline = makePipeline("accumulateClampedKernel")
        svgfTemporalPipeline = makePipeline("svgfTemporalKernel")
        svgfATrousPipeline = makePipeline("svgfATrousKernel")
        svgfRemodPipeline = makePipeline("svgfRemodulateKernel")
        svgfDemodPipeline = makePipeline("svgfDemodulateKernel")
        postATrousPipeline = makePipeline("postATrousKernel")
        crossBilateralPipeline = makePipeline("crossBilateralKernel")
        nlmPipeline = makePipeline("nlmDenoiseKernel")
        clearR16Pipeline = makePipeline("clearR16Kernel")
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

        colorTexture = makeTexture(renderWidth, renderHeight, format: .rgba32Float)
        depthTexture = makeTexture(renderWidth, renderHeight, format: .r32Float)
        motionTexture = makeTexture(renderWidth, renderHeight, format: .rg16Float)
        normalTexture = makeTexture(renderWidth, renderHeight, format: .rgba16Float)
        albedoTexture = makeTexture(renderWidth, renderHeight, format: .rgba16Float)
        historyTexture = makeTexture(renderWidth, renderHeight, format: .rgba32Float)
        accumulatedTexture = makeTexture(renderWidth, renderHeight, format: .rgba32Float)
        svgfTemporalTexture = makeTexture(renderWidth, renderHeight, format: .rgba32Float)
        svgfMomentsTexture = makeTexture(renderWidth, renderHeight, format: .rg32Float)
        svgfHistoryMomentsTexture = makeTexture(renderWidth, renderHeight, format: .rg32Float)
        svgfHistoryLengthTexture = makeTexture(renderWidth, renderHeight, format: .r16Float)
        svgfHistoryLengthScratchTexture = makeTexture(renderWidth, renderHeight, format: .r16Float)
        svgfHistoryDepthTexture = makeTexture(renderWidth, renderHeight, format: .r32Float)
        svgfHistoryNormalTexture = makeTexture(renderWidth, renderHeight, format: .rgba16Float)
        svgfHistoryAlbedoTexture = makeTexture(renderWidth, renderHeight, format: .rgba16Float)
        svgfPingTexture = makeTexture(renderWidth, renderHeight, format: .rgba32Float)
        svgfPongTexture = makeTexture(renderWidth, renderHeight, format: .rgba32Float)
        svgfFilteredTexture = nil
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
              let normalTex = normalTexture,
              let albedoTex = albedoTexture,
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
            encoder.setTexture(normalTex, index: 3)
            encoder.setTexture(albedoTex, index: 4)

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

    func encodeClampedAccumulation(commandBuffer: MTLCommandBuffer,
                                   uniforms: inout Uniforms) {
        guard let colorTex = colorTexture,
              let historyTex = historyTexture,
              let accumTex = accumulatedTexture else {
            return
        }

        let renderW = Int(uniforms.renderWidth)
        let renderH = Int(uniforms.renderHeight)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Accumulate (Clamped)"
            encoder.setComputePipelineState(accumulateClampedPipeline)
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

    /// Post-filter: Demodulate → EAW (5-pass A-Trous) → Remodulate.
    /// Applied as a post-filter on the accumulated image.
    func encodePostATrous(commandBuffer: MTLCommandBuffer,
                          uniforms: inout Uniforms,
                          sourceTexture: MTLTexture) -> MTLTexture? {
        guard let depthTex = depthTexture,
              let normalTex = normalTexture,
              let albedoTex = albedoTexture,
              let pingTex = svgfPingTexture,
              let pongTex = svgfPongTexture else {
            return nil
        }

        let renderW = Int(uniforms.renderWidth)
        let renderH = Int(uniforms.renderHeight)
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let gridSize = MTLSize(width: renderW, height: renderH, depth: 1)

        // Demodulate: radiance → irradiance (into pingTex)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "EAW Demodulate"
            encoder.setComputePipelineState(svgfDemodPipeline)
            encoder.setTexture(sourceTexture, index: 0)
            encoder.setTexture(normalTex, index: 1)
            encoder.setTexture(albedoTex, index: 2)
            encoder.setTexture(pingTex, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        var filterUniforms = PostFilterUniforms(
            renderWidth: uniforms.renderWidth,
            renderHeight: uniforms.renderHeight,
            stepWidth: 1,
            passIndex: 0,
            colorPhiScale: 2.0,
            normalPhi: 128.0,
            depthPhi: 2.0,
            albedoPhi: 12.0,
            accumulationCount: uniforms.accumulationCount,
            _pad0: 0, _pad1: 0, _pad2: 0
        )

        var srcTex: MTLTexture = pingTex
        var dstTex: MTLTexture = pongTex

        for (passIndex, stepWidth) in eawStepWidths.enumerated() {
            filterUniforms.stepWidth = stepWidth
            filterUniforms.passIndex = UInt32(passIndex)

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "EAW A-Trous \(passIndex)"
                encoder.setComputePipelineState(postATrousPipeline)
                encoder.setBytes(&filterUniforms,
                                 length: MemoryLayout<PostFilterUniforms>.size,
                                 index: 0)
                encoder.setTexture(srcTex, index: 0)
                encoder.setTexture(depthTex, index: 1)
                encoder.setTexture(normalTex, index: 2)
                encoder.setTexture(albedoTex, index: 3)
                encoder.setTexture(dstTex, index: 4)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                encoder.endEncoding()
            }

            let prev = dstTex
            srcTex = prev
            dstTex = (prev === pingTex) ? pongTex : pingTex
        }

        // Remodulate: irradiance × albedo → radiance
        let remodTarget = dstTex
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "EAW Remodulate"
            encoder.setComputePipelineState(svgfRemodPipeline)
            encoder.setTexture(srcTex, index: 0)
            encoder.setTexture(normalTex, index: 1)
            encoder.setTexture(albedoTex, index: 2)
            encoder.setTexture(remodTarget, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        return remodTarget
    }

    /// Post-filter: Demodulate → Cross-Bilateral → Remodulate.
    func encodeCrossBilateral(commandBuffer: MTLCommandBuffer,
                              uniforms: inout Uniforms,
                              sourceTexture: MTLTexture) -> MTLTexture? {
        guard let depthTex = depthTexture,
              let normalTex = normalTexture,
              let albedoTex = albedoTexture,
              let pingTex = svgfPingTexture,
              let pongTex = svgfPongTexture else {
            return nil
        }

        let renderW = Int(uniforms.renderWidth)
        let renderH = Int(uniforms.renderHeight)
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let gridSize = MTLSize(width: renderW, height: renderH, depth: 1)

        // Demodulate
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Bilateral Demodulate"
            encoder.setComputePipelineState(svgfDemodPipeline)
            encoder.setTexture(sourceTexture, index: 0)
            encoder.setTexture(normalTex, index: 1)
            encoder.setTexture(albedoTex, index: 2)
            encoder.setTexture(pingTex, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        // Adaptive radius: larger when noisy (low spp), smaller when converged
        let accumCount = max(uniforms.accumulationCount, 1)
        let adaptiveRadius: Int32 = accumCount < 8 ? 5 : (accumCount < 32 ? 4 : 3)
        let adaptiveSigmaColor: Float = accumCount < 8 ? 0.3 : (accumCount < 32 ? 0.15 : 0.08)

        var bilateralUniforms = CrossBilateralUniforms(
            renderWidth: uniforms.renderWidth,
            renderHeight: uniforms.renderHeight,
            radius: adaptiveRadius,
            accumulationCount: uniforms.accumulationCount,
            sigmaSpatial: Float(adaptiveRadius) * 0.6,
            sigmaColor: adaptiveSigmaColor,
            depthPhi: 2.0,
            normalPhi: 128.0,
            albedoPhi: 12.0,
            _pad0: 0, _pad1: 0, _pad2: 0
        )

        // Bilateral filter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Cross-Bilateral"
            encoder.setComputePipelineState(crossBilateralPipeline)
            encoder.setBytes(&bilateralUniforms,
                             length: MemoryLayout<CrossBilateralUniforms>.size,
                             index: 0)
            encoder.setTexture(pingTex, index: 0)
            encoder.setTexture(depthTex, index: 1)
            encoder.setTexture(normalTex, index: 2)
            encoder.setTexture(albedoTex, index: 3)
            encoder.setTexture(pongTex, index: 4)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        // Remodulate
        let resultTex = pingTex
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Bilateral Remodulate"
            encoder.setComputePipelineState(svgfRemodPipeline)
            encoder.setTexture(pongTex, index: 0)
            encoder.setTexture(normalTex, index: 1)
            encoder.setTexture(albedoTex, index: 2)
            encoder.setTexture(resultTex, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        return resultTex
    }

    /// Post-filter: Demodulate → Non-Local Means → Remodulate.
    func encodeNLM(commandBuffer: MTLCommandBuffer,
                   uniforms: inout Uniforms,
                   sourceTexture: MTLTexture) -> MTLTexture? {
        guard let depthTex = depthTexture,
              let normalTex = normalTexture,
              let albedoTex = albedoTexture,
              let pingTex = svgfPingTexture,
              let pongTex = svgfPongTexture else {
            return nil
        }

        let renderW = Int(uniforms.renderWidth)
        let renderH = Int(uniforms.renderHeight)
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let gridSize = MTLSize(width: renderW, height: renderH, depth: 1)

        // Demodulate
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "NLM Demodulate"
            encoder.setComputePipelineState(svgfDemodPipeline)
            encoder.setTexture(sourceTexture, index: 0)
            encoder.setTexture(normalTex, index: 1)
            encoder.setTexture(albedoTex, index: 2)
            encoder.setTexture(pingTex, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        // Adaptive NLM parameters based on accumulation count
        let accumCount = max(uniforms.accumulationCount, 1)
        let adaptiveH: Float = accumCount < 8 ? 0.6 : (accumCount < 32 ? 0.3 : 0.15)
        let adaptiveSearchRadius: Int32 = accumCount < 16 ? 10 : 7

        var nlmUniforms = NLMUniforms(
            renderWidth: uniforms.renderWidth,
            renderHeight: uniforms.renderHeight,
            searchRadius: adaptiveSearchRadius,
            patchRadius: 2,
            h: adaptiveH,
            depthRejectThreshold: 0.05,
            normalRejectDot: 0.7,
            albedoRejectDelta: 0.25,
            accumulationCount: uniforms.accumulationCount,
            _pad0: 0, _pad1: 0, _pad2: 0
        )

        // NLM filter
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Non-Local Means"
            encoder.setComputePipelineState(nlmPipeline)
            encoder.setBytes(&nlmUniforms,
                             length: MemoryLayout<NLMUniforms>.size,
                             index: 0)
            encoder.setTexture(pingTex, index: 0)
            encoder.setTexture(depthTex, index: 1)
            encoder.setTexture(normalTex, index: 2)
            encoder.setTexture(albedoTex, index: 3)
            encoder.setTexture(pongTex, index: 4)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        // Remodulate
        let resultTex = pingTex
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "NLM Remodulate"
            encoder.setComputePipelineState(svgfRemodPipeline)
            encoder.setTexture(pongTex, index: 0)
            encoder.setTexture(normalTex, index: 1)
            encoder.setTexture(albedoTex, index: 2)
            encoder.setTexture(resultTex, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        return resultTex
    }

    func encodeSVGF(commandBuffer: MTLCommandBuffer,
                    uniforms: inout Uniforms) -> MTLTexture? {
        guard let colorTex = colorTexture,
              let depthTex = depthTexture,
              let motionTex = motionTexture,
              let normalTex = normalTexture,
              let albedoTex = albedoTexture,
              let temporalTex = svgfTemporalTexture,
              let momentsTex = svgfMomentsTexture,
              let historyMomentsTex = svgfHistoryMomentsTexture,
              let historyLengthTex = svgfHistoryLengthTexture,
              let historyLengthScratchTex = svgfHistoryLengthScratchTexture,
              let historyDepthTex = svgfHistoryDepthTexture,
              let historyNormalTex = svgfHistoryNormalTexture,
              let historyAlbedoTex = svgfHistoryAlbedoTexture,
              let historyTex = historyTexture,
              let pingTex = svgfPingTexture,
              let pongTex = svgfPongTexture else {
            return nil
        }

        let renderW = Int(uniforms.renderWidth)
        let renderH = Int(uniforms.renderHeight)
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let gridSize = MTLSize(width: renderW, height: renderH, depth: 1)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "SVGF Temporal"
            encoder.setComputePipelineState(svgfTemporalPipeline)
            encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
            encoder.setTexture(colorTex, index: 0)
            encoder.setTexture(depthTex, index: 1)
            encoder.setTexture(motionTex, index: 2)
            encoder.setTexture(normalTex, index: 3)
            encoder.setTexture(albedoTex, index: 4)
            encoder.setTexture(historyTex, index: 5)
            encoder.setTexture(historyMomentsTex, index: 6)
            encoder.setTexture(historyLengthTex, index: 7)
            encoder.setTexture(historyDepthTex, index: 8)
            encoder.setTexture(historyNormalTex, index: 9)
            encoder.setTexture(historyAlbedoTex, index: 10)
            encoder.setTexture(temporalTex, index: 11)
            encoder.setTexture(momentsTex, index: 12)
            encoder.setTexture(historyLengthScratchTex, index: 13)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        var filterUniforms = SVGFATrousUniforms(
            renderWidth: uniforms.renderWidth,
            renderHeight: uniforms.renderHeight,
            stepWidth: 1,
            passIndex: 0,
            colorPhiScale: 2.0,
            normalPhi: 128.0,
            depthPhi: 2.0,
            albedoPhi: 12.0
        )

        // Demodulate: temporal radiance → irradiance (into pingTex)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "SVGF Demodulate"
            encoder.setComputePipelineState(svgfDemodPipeline)
            encoder.setTexture(temporalTex, index: 0)
            encoder.setTexture(normalTex, index: 1)
            encoder.setTexture(albedoTex, index: 2)
            encoder.setTexture(pingTex, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        // A-Trous passes operate on irradiance
        var sourceTexture: MTLTexture = pingTex
        var targetTexture: MTLTexture = pongTex

        for (passIndex, stepWidth) in svgfStepWidths.enumerated() {
            filterUniforms.stepWidth = stepWidth
            filterUniforms.passIndex = UInt32(passIndex)

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "SVGF A-Trous \(passIndex)"
                encoder.setComputePipelineState(svgfATrousPipeline)
                encoder.setBytes(&filterUniforms,
                                 length: MemoryLayout<SVGFATrousUniforms>.size,
                                 index: 0)
                encoder.setTexture(sourceTexture, index: 0)
                encoder.setTexture(momentsTex, index: 1)
                encoder.setTexture(depthTex, index: 2)
                encoder.setTexture(normalTex, index: 3)
                encoder.setTexture(albedoTex, index: 4)
                encoder.setTexture(targetTexture, index: 5)
                encoder.setTexture(historyLengthScratchTex, index: 6)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                encoder.endEncoding()
            }

            let previousTarget = targetTexture
            sourceTexture = previousTarget
            targetTexture = (previousTarget === pingTex) ? pongTex : pingTex
        }

        // Remodulate: irradiance × albedo → radiance
        let remodSource = sourceTexture
        let remodTarget = targetTexture

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "SVGF Remodulate"
            encoder.setComputePipelineState(svgfRemodPipeline)
            encoder.setTexture(remodSource, index: 0)
            encoder.setTexture(normalTex, index: 1)
            encoder.setTexture(albedoTex, index: 2)
            encoder.setTexture(remodTarget, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        svgfFilteredTexture = remodTarget

        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "SVGF History Update"
            blit.copy(from: temporalTex, to: historyTex)
            blit.copy(from: momentsTex, to: historyMomentsTex)
            blit.copy(from: historyLengthScratchTex, to: historyLengthTex)
            blit.copy(from: depthTex, to: historyDepthTex)
            blit.copy(from: normalTex, to: historyNormalTex)
            blit.copy(from: albedoTex, to: historyAlbedoTex)
            blit.endEncoding()
        }

        return svgfFilteredTexture
    }

    /// Spatial-only denoise: demodulate → À-Trous → remodulate (no temporal pass).
    /// Suitable as a pre-filter before MetalFX temporal upscaling.
    func encodeSpatialDenoise(commandBuffer: MTLCommandBuffer,
                              uniforms: inout Uniforms,
                              sourceTexture: MTLTexture) -> MTLTexture? {
        guard let depthTex = depthTexture,
              let normalTex = normalTexture,
              let albedoTex = albedoTexture,
              let pingTex = svgfPingTexture,
              let pongTex = svgfPongTexture else {
            return nil
        }

        let renderW = Int(uniforms.renderWidth)
        let renderH = Int(uniforms.renderHeight)
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let gridSize = MTLSize(width: renderW, height: renderH, depth: 1)

        var filterUniforms = SVGFATrousUniforms(
            renderWidth: uniforms.renderWidth,
            renderHeight: uniforms.renderHeight,
            stepWidth: 1,
            passIndex: 0,
            colorPhiScale: 2.0,
            normalPhi: 128.0,
            depthPhi: 2.0,
            albedoPhi: 12.0
        )

        // Demodulate: radiance → irradiance (into pingTex)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Spatial Demodulate"
            encoder.setComputePipelineState(svgfDemodPipeline)
            encoder.setTexture(sourceTexture, index: 0)
            encoder.setTexture(normalTex, index: 1)
            encoder.setTexture(albedoTex, index: 2)
            encoder.setTexture(pingTex, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        // Clear history length scratch to zero so À-Trous applies full spatial filtering
        if let hlTex = svgfHistoryLengthScratchTexture {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Clear HistoryLength"
                encoder.setComputePipelineState(clearR16Pipeline)
                encoder.setTexture(hlTex, index: 0)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                encoder.endEncoding()
            }
        }

        // Clear moments so À-Trous uses colorPhiScale-driven filtering
        if let momentsTex = svgfMomentsTexture {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Clear Moments"
                encoder.setComputePipelineState(clearR16Pipeline)
                encoder.setTexture(momentsTex, index: 0)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                encoder.endEncoding()
            }
        }

        // À-Trous passes on irradiance (ping-pong)
        var srcTex: MTLTexture = pingTex
        var dstTex: MTLTexture = pongTex

        if let momentsTex = svgfMomentsTexture {
            let spatialStepWidths: [UInt32] = [1, 2]

            for (passIndex, stepWidth) in spatialStepWidths.enumerated() {
                filterUniforms.stepWidth = stepWidth
                filterUniforms.passIndex = UInt32(passIndex)

                if let encoder = commandBuffer.makeComputeCommandEncoder() {
                    encoder.label = "Spatial A-Trous \(passIndex)"
                    encoder.setComputePipelineState(svgfATrousPipeline)
                    encoder.setBytes(&filterUniforms,
                                     length: MemoryLayout<SVGFATrousUniforms>.size,
                                     index: 0)
                    encoder.setTexture(srcTex, index: 0)
                    encoder.setTexture(momentsTex, index: 1)
                    encoder.setTexture(depthTex, index: 2)
                    encoder.setTexture(normalTex, index: 3)
                    encoder.setTexture(albedoTex, index: 4)
                    encoder.setTexture(dstTex, index: 5)
                    // historyLength cleared to 0 → adaptiveScale = max
                    // which means full spatial filtering (no convergence cutoff).
                    if let hlTex = svgfHistoryLengthScratchTexture {
                        encoder.setTexture(hlTex, index: 6)
                    }
                    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                    encoder.endEncoding()
                }

                let previousTarget = dstTex
                srcTex = previousTarget
                dstTex = (previousTarget === pingTex) ? pongTex : pingTex
            }
        }

        // Remodulate: irradiance × albedo → radiance
        let remodSource = srcTex
        let remodTarget = dstTex

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "Spatial Remodulate"
            encoder.setComputePipelineState(svgfRemodPipeline)
            encoder.setTexture(remodSource, index: 0)
            encoder.setTexture(normalTex, index: 1)
            encoder.setTexture(albedoTex, index: 2)
            encoder.setTexture(remodTarget, index: 3)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        return remodTarget
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
