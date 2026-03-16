import AppKit
import Metal
import MetalFX
import QuartzCore
import simd

// ─── Renderer ────────────────────────────────────────────────────────────────
//
// Orchestrates the full render pipeline:
//   1. Path trace at full resolution, 8 bounces
//   2. Route through raw, accumulation, SVGF, or MetalFX based on render mode
//   3. Tonemap + present to drawable

class Renderer {
    private static let defaultStartupCameraPosition = SIMD3<Float>(480, 50, 150)
    private static let defaultStartupCameraYawDegrees: Float = 0
    private static let defaultStartupCameraPitchDegrees: Float = 0

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let camera = Camera()
    let verifyConfig: VerifyConfig
    private(set) var selectedRenderMode: RenderMode
    private(set) var activeRenderMode: RenderMode
    private(set) var selectedDenoiseMethod: DenoiseMethod

    private var pathTracer: PathTracer
    private var accelBuilder: AccelStructureBuilder
    private var scene: SceneGeometry
    private var upscaler: MetalFXUpscaler?
    private let oidnDenoiser = OIDNDenoiser()
    private(set) var sceneTitleLabel: String = "test"
    private let startupCameraPosition: SIMD3<Float>
    private let startupCameraYaw: Float
    private let startupCameraPitch: Float
    private let hasExplicitStartupCameraPosition: Bool
    private let startupLookAtPosition: SIMD3<Float>?
    private let lookAtWaterOnLoad: Bool
    private let highlightWater: Bool
    private var environmentSettings = EnvironmentSettings.default
    private var currentExposure: Float = 4.0
    private var measuredAverageLuminance: Float = 0.18
    private var lastExposureUpdateTime = CACurrentMediaTime()
    private let exposureKeyValue: Float = 0.18
    private let minAutoExposure: Float = 0.35
    private let maxAutoExposure: Float = 8.0
    private let darkToBrightAdaptationRate: Float = 3.0
    private let brightToDarkAdaptationRate: Float = 1.35
    private let metalFXRenderScale: Float = 0.67

    // Resolution
    private var outputWidth: Int = 2560
    private var outputHeight: Int = 1440

    // Accumulation
    private(set) var accumulationCount: UInt32 = 0
    private var frameIndex: UInt32 = 0

    // State
    private var sceneLoaded = false

    // Verification
    var pendingScreenshot = false
    private var verifyCompleted = false
    private(set) var captureInProgress = false
    private var submittedFrames: UInt32 = 0
    private var completedFrames: UInt32 = 0
    private var previousCameraMoving = false
    private var previousJitter = SIMD2<Float>(repeating: 0)
    private let verifyCheckpointFrames: [UInt32]
    private let verifySweepMethods: [DenoiseMethod]
    private let verifyReferenceFrame: UInt32?
    private let verifyReferenceDenoiseMethod: DenoiseMethod
    private var verifyCheckpointIndex = 0
    private var verifySweepIndex = 0
    private var verifyAllPassed = true
    private var verifyReferencePixels: [UInt8]?
    private var verifyReferenceHDRPixels: [Float]?
    private var verifyReferenceLabel: String?
    private var verifyReferenceReady = false
    private var verifyReferenceTransitionPending = false
    private let oidnQueue = DispatchQueue(label: "dev.rt.oidn", qos: .userInitiated, attributes: .concurrent)
    private var oidnLatestTexture: MTLTexture?
    private var oidnLatestFrame: UInt32 = 0
    private var oidnJobInFlight = false
    private var oidnJobStartTime: CFTimeInterval?
    private var oidnGeneration: UInt64 = 0
    private var oidnUseAuxiliaryInputs = true
    private var oidnRuntimeDisabledReason: String?
    private var hasLoggedOIDNAvailability = false
    private let shouldLogOIDNEvents: Bool
    private let oidnJobTimeout: CFTimeInterval = 8.0

    init(device: MTLDevice, view: GameView, launchConfig: LaunchConfig) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.verifyConfig = launchConfig.verifyConfig
        self.selectedRenderMode = launchConfig.renderMode
        self.activeRenderMode = launchConfig.renderMode
        let shouldSweepDenoise = launchConfig.verifyConfig.enabled
            && launchConfig.verifyConfig.sweepDenoiseMethods
            && launchConfig.renderMode == .accumulation

        let checkpointFrames = Array(Set(launchConfig.verifyConfig.checkpointFrames.filter { $0 > 0 })).sorted()
        self.verifyCheckpointFrames = checkpointFrames.isEmpty
            ? [max(launchConfig.verifyConfig.targetFrames, 1)]
            : checkpointFrames
        self.verifySweepMethods = shouldSweepDenoise ? DenoiseMethod.allCases : [launchConfig.denoiseMethod]
        self.verifyReferenceFrame = launchConfig.verifyConfig.referenceFrames
        self.verifyReferenceDenoiseMethod = launchConfig.verifyConfig.referenceDenoiseMethod
        self.selectedDenoiseMethod = launchConfig.verifyConfig.referenceFrames != nil
            ? launchConfig.verifyConfig.referenceDenoiseMethod
            : (shouldSweepDenoise
                ? (self.verifySweepMethods.first ?? launchConfig.denoiseMethod)
                : launchConfig.denoiseMethod)
        self.startupCameraPosition = launchConfig.startPosition ?? Self.defaultStartupCameraPosition
        self.startupCameraYaw = Self.defaultStartupCameraYawDegrees * .pi / 180.0
        self.startupCameraPitch = Self.defaultStartupCameraPitchDegrees * .pi / 180.0
        self.hasExplicitStartupCameraPosition = launchConfig.startPosition != nil
        self.startupLookAtPosition = launchConfig.lookAtPosition
        self.lookAtWaterOnLoad = launchConfig.lookAtWater
        self.highlightWater = launchConfig.highlightWater
        self.shouldLogOIDNEvents = launchConfig.scriptedOIDNRepro

        self.pathTracer = PathTracer(device: device, commandQueue: commandQueue)
        self.accelBuilder = AccelStructureBuilder(device: device, commandQueue: commandQueue)
        self.scene = SceneGeometry(device: device)
        self.upscaler = MetalFXUpscaler(device: device)

        let scale = view.window?.backingScaleFactor ?? 2.0
        outputWidth = Int(view.bounds.width * scale)
        outputHeight = Int(view.bounds.height * scale)

        loadScene()
    }

    var renderModeTitle: String {
        if selectedRenderMode == .auto {
            return "Auto/\(activeRenderMode.displayName)"
        }

        if selectedRenderMode != activeRenderMode {
            return "\(selectedRenderMode.displayName)->\(activeRenderMode.displayName)"
        }

        return activeRenderMode.displayName
    }

    var renderModeDetail: String {
        switch activeRenderMode {
        case .raw:
            return "1 spp"
        case .accumulation:
            let base = "\(max(accumulationCount, 1)) spp"
            if selectedDenoiseMethod != .none {
                return "\(base) + \(selectedDenoiseMethod.displayName)"
            }
            return base
        case .svgf, .metalfx, .metalfxSVGF:
            return "\(max(accumulationCount, 1)) history"
        case .auto:
            return "\(max(accumulationCount, 1)) history"
        }
    }

    var renderModeDetailCompact: String {
        switch activeRenderMode {
        case .raw:
            return "1spp"
        case .accumulation:
            let base = "\(max(accumulationCount, 1))spp"
            if selectedDenoiseMethod != .none {
                return "\(base)+\(selectedDenoiseMethod.displayName)"
            }
            return base
        case .svgf, .metalfx, .metalfxSVGF, .auto:
            return "\(max(accumulationCount, 1))h"
        }
    }

    private var oidnStatusTitle: String? {
        guard activeRenderMode == .accumulation,
              selectedDenoiseMethod == .oidn else {
            return nil
        }

        if !oidnDenoiser.isAvailable {
            return "OIDN unavailable"
        }

        if let oidnRuntimeDisabledReason {
            return oidnRuntimeDisabledReason
        }

        if camera.isMoving {
            return "OIDN moving"
        }

        let targetFrame = max(accumulationCount, 1)
        let hasPresentationTexture = currentOIDNPresentationTexture() != nil
        let elapsed = oidnJobElapsedTime

        if hasPresentationTexture {
            if oidnJobInFlight, oidnLatestFrame < targetFrame {
                return "\(oidnPipelineTitle) ready@\(oidnLatestFrame)spp -> \(targetFrame)spp"
            }
            return "\(oidnPipelineTitle) ready@\(oidnLatestFrame)spp"
        }

        if oidnJobInFlight {
            if let elapsed, elapsed >= oidnJobTimeout {
                return "\(oidnPipelineTitle) stalled@\(targetFrame)spp"
            }
            return "\(oidnPipelineTitle) pending@\(targetFrame)spp"
        }

        return "\(oidnPipelineTitle) waiting@\(targetFrame)spp"
    }

    private var oidnPipelineTitle: String {
        oidnUseAuxiliaryInputs ? "OIDN" : "OIDN(color)"
    }

    private var oidnJobElapsedTime: CFTimeInterval? {
        guard let oidnJobStartTime else {
            return nil
        }
        return CACurrentMediaTime() - oidnJobStartTime
    }

    var windowTitle: String {
        var components = [
            "RT Path Tracer",
            sceneTitleLabel,
            "p\(camera.compactPositionString)",
            "v\(camera.compactViewString)",
            "\(renderModeTitle) \(renderModeDetailCompact)"
        ]
        if let oidnStatusTitle {
            components.append(oidnStatusTitle)
        }
        return components.joined(separator: " | ")
    }

    func cycleRenderMode() {
        guard let currentIndex = RenderMode.allCases.firstIndex(of: selectedRenderMode) else {
            return
        }

        let nextIndex = RenderMode.allCases.index(after: currentIndex)
        selectedRenderMode = nextIndex == RenderMode.allCases.endIndex
            ? RenderMode.allCases[0]
            : RenderMode.allCases[nextIndex]
        activeRenderMode = selectedRenderMode
        resetTemporalState(syncCameraHistory: true)
        print("[Renderer] Render mode: \(selectedRenderMode.displayName)")
    }

    func cycleDenoiseMethod() {
        guard let currentIndex = DenoiseMethod.allCases.firstIndex(of: selectedDenoiseMethod) else {
            return
        }

        let nextIndex = DenoiseMethod.allCases.index(after: currentIndex)
        let nextMethod = nextIndex == DenoiseMethod.allCases.endIndex
            ? DenoiseMethod.allCases[0]
            : DenoiseMethod.allCases[nextIndex]
        selectedDenoiseMethod = nextMethod
        resetOIDNRuntimeConfiguration()

        if selectedRenderMode == .accumulation || activeRenderMode == .accumulation {
            resetTemporalState(syncCameraHistory: true)
        }

        print("[Renderer] Accumulation denoiser: \(selectedDenoiseMethod.displayName)")
    }

    // MARK: - Scene Loading

    private func loadScene() {
        // Try to find Quake assets
        let assetsDir = findAssetsDirectory()
        var bspData: BSPData?
        let requestedMapName = "e1m1"
        let requestedMapPath = "maps/\(requestedMapName).bsp"

        // Load Quake palette for texture decoding
        let pakPath = assetsDir.appendingPathComponent("pak0.pak")
        var palette: [UInt8]? = nil
        if FileManager.default.fileExists(atPath: pakPath.path) {
            if let paletteData = try? PAKLoader().extractFile(from: pakPath, name: "gfx/palette.lmp") {
                palette = [UInt8](paletteData)
                print("[Renderer] Loaded Quake palette (\(paletteData.count) bytes)")
            }
        }

        // 1. Try loading .bsp directly
        let bspPath = assetsDir.appendingPathComponent("\(requestedMapName).bsp")
        if FileManager.default.fileExists(atPath: bspPath.path) {
            print("[Renderer] Loading BSP from: \(bspPath.path)")
            do {
                bspData = try BSPLoader().load(from: bspPath, palette: palette)
                sceneTitleLabel = requestedMapName
            } catch {
                print("[Renderer] Failed to load BSP: \(error)")
            }
        }

        // 2. Try extracting from pak0.pak
        if bspData == nil {
            if FileManager.default.fileExists(atPath: pakPath.path) {
                print("[Renderer] Loading from PAK: \(pakPath.path)")
                do {
                    let pakData = try PAKLoader().extractBSP(from: pakPath, mapName: requestedMapPath)
                    bspData = try BSPLoader().load(data: pakData, palette: palette)
                    sceneTitleLabel = requestedMapName
                } catch {
                    print("[Renderer] Failed to load PAK: \(error)")
                }
            }
        }

        // 3. Fallback: generate a test scene
        if bspData == nil {
            print("[Renderer] No Quake assets found. Generating test scene.")
            print("[Renderer] Place pak0.pak or \(requestedMapName).bsp in: \(assetsDir.path)")
            bspData = generateTestScene()
            sceneTitleLabel = "test"
        }

        guard let data = bspData else { return }
        environmentSettings = data.environment

        // Upload to GPU
        scene.loadFromBSP(data)

        let envTint = environmentSettings.tint
        let sunDir = environmentSettings.sunDirection
        print(String(format: "[Renderer] Environment tint=(%.2f, %.2f, %.2f) intensity=%.2f sunDir=(%.2f, %.2f, %.2f) sunIntensity=%.2f",
                 envTint.x,
                 envTint.y,
                 envTint.z,
                 environmentSettings.intensity,
                 sunDir.x,
                 sunDir.y,
                 sunDir.z,
                 environmentSettings.sunIntensity))

        let waterFramingPosition = data.preferredLiquidSurface.map { liquidSurface in
            let suggestedCameraPosition = liquidSurface.position + liquidSurface.normal * 24.0
            print(String(format: "[Renderer] Water surface '%@' center=(%.1f, %.1f, %.1f) suggested-camera=(%.1f, %.1f, %.1f)",
                         liquidSurface.materialName,
                         liquidSurface.position.x,
                         liquidSurface.position.y,
                         liquidSurface.position.z,
                         suggestedCameraPosition.x,
                         suggestedCameraPosition.y,
                         suggestedCameraPosition.z))

            return liquidSurface.position + liquidSurface.normal * 96.0
        }

        if lookAtWaterOnLoad, !hasExplicitStartupCameraPosition, let waterFramingPosition {
            print(String(format: "[Renderer] Water framing camera=(%.1f, %.1f, %.1f)",
                         waterFramingPosition.x,
                         waterFramingPosition.y,
                         waterFramingPosition.z))
        }

        // Use the configured startup pose unless look-at-water requests its framing shot.
        if lookAtWaterOnLoad, !hasExplicitStartupCameraPosition, let waterFramingPosition {
            camera.position = waterFramingPosition
        } else {
            camera.position = startupCameraPosition
        }
        camera.yaw = startupCameraYaw
        camera.pitch = startupCameraPitch
        print(String(format: "[Renderer] Camera start=(%.1f, %.1f, %.1f) view=(%.0f, %.0f)",
                     camera.position.x,
                     camera.position.y,
                     camera.position.z,
                     camera.yaw * 180.0 / .pi,
                     camera.pitch * 180.0 / .pi))

        if lookAtWaterOnLoad, let liquidSurface = data.preferredLiquidSurface {
            camera.lookAt(target: liquidSurface.position)
            camera.focusDistance = length(liquidSurface.position - camera.position)
            print(String(format: "[Renderer] Camera looking at water center=(%.1f, %.1f, %.1f) focusDistance=%.1f",
                         liquidSurface.position.x,
                         liquidSurface.position.y,
                         liquidSurface.position.z,
                         camera.focusDistance))
        } else if let startupLookAtPosition {
            camera.lookAt(target: startupLookAtPosition)
            camera.focusDistance = length(startupLookAtPosition - camera.position)
            print(String(format: "[Renderer] Camera look-at target=(%.1f, %.1f, %.1f) focusDistance=%.1f",
                         startupLookAtPosition.x,
                         startupLookAtPosition.y,
                         startupLookAtPosition.z,
                         camera.focusDistance))
        }
        camera.syncHistory()

        // Build acceleration structure
        accelBuilder.build(scene: scene)

        sceneLoaded = (accelBuilder.accelerationStructure != nil)

        if sceneLoaded {
            previousJitter = camera.jitterOffset
            print("[Renderer] Scene ready!")
        }
    }

    // MARK: - Render

    func render(to drawable: CAMetalDrawable, onComplete: (() -> Void)? = nil) {
        guard sceneLoaded,
              let accelStructure = accelBuilder.accelerationStructure else {
            onComplete?()
            return
        }

        camera.aspectRatio = Float(outputWidth) / Float(outputHeight)
        let isMoving = camera.isMoving
        let startedMoving = isMoving && !previousCameraMoving
        let stoppedMoving = !isMoving && previousCameraMoving
        let jitter = camera.jitterOffset
        camera.advanceJitter()
        let now = CACurrentMediaTime()
        let exposureDeltaTime = min(max(Float(now - lastExposureUpdateTime), 1.0 / 240.0), 0.25)
        lastExposureUpdateTime = now
        updateExposure(deltaTime: exposureDeltaTime)
        recoverFromStalledOIDNJob(now: now)

        let requestedRenderMode = preferredRenderMode(isMoving: isMoving)
        let wantsMetalFX = (requestedRenderMode == .metalfx || requestedRenderMode == .metalfxSVGF) && upscaler?.supportsMetalFX == true
        let renderScale = wantsMetalFX ? metalFXRenderScale : 1.0
        let renderW = max(1, Int((Float(outputWidth) * renderScale).rounded(.toNearestOrAwayFromZero)))
        let renderH = max(1, Int((Float(outputHeight) * renderScale).rounded(.toNearestOrAwayFromZero)))

        // ── Ensure textures / temporal state ──
        let texturesChanged = pathTracer.ensureTextures(renderWidth: renderW, renderHeight: renderH,
                                 outputWidth: outputWidth, outputHeight: outputHeight)
        if texturesChanged {
            camera.syncHistory()
            previousJitter = jitter
            accumulationCount = 0
            upscaler?.reset()
            resetOIDNState()
        }

        upscaler?.configure(inputWidth: renderW,
                            inputHeight: renderH,
                            outputWidth: outputWidth,
                            outputHeight: outputHeight)

        let effectiveRenderMode = resolvedRenderMode(requestedMode: requestedRenderMode)
        activeRenderMode = effectiveRenderMode

        // ── Accumulation logic ──
        switch effectiveRenderMode {
        case .raw:
            accumulationCount = 0
        case .accumulation:
            if isMoving {
                if selectedDenoiseMethod == .oidn {
                    resetOIDNState()
                }
                accumulationCount = 0
            } else {
                accumulationCount += 1
            }
        case .svgf:
            if selectedRenderMode == .auto && stoppedMoving {
                accumulationCount = 0
            }
            accumulationCount += 1
        case .metalfx:
            if isMoving {
                accumulationCount = 0
                if startedMoving {
                    upscaler?.reset()
                }
            } else {
                if stoppedMoving {
                    accumulationCount = 0
                    upscaler?.reset()
                }
                accumulationCount += 1
            }
        case .metalfxSVGF:
            if isMoving {
                accumulationCount = 0
                if startedMoving {
                    upscaler?.reset()
                }
            } else {
                if stoppedMoving {
                    accumulationCount = 0
                    upscaler?.reset()
                }
                accumulationCount += 1
            }
        case .auto:
            accumulationCount = 0
        }

        if effectiveRenderMode == .accumulation,
           selectedDenoiseMethod == .oidn,
           !isMoving,
           shouldResetOIDNForAccumulationRestart(stoppedMoving: stoppedMoving) {
            resetOIDNState()
        }

        // ── Build uniforms ──
        let pos = camera.position
        let right = camera.rightVector
        let up = camera.upVector
        let forward = camera.forwardVector
        var uniforms = Uniforms(
            currentViewProjection: camera.viewProjectionMatrix,
            inverseViewProjection: camera.inverseViewProjectionMatrix,
            previousViewProjection: camera.previousViewProjectionMatrix,
            cameraPosition: (pos.x, pos.y, pos.z),
            frameIndex: frameIndex,
            cameraRight: (right.x, right.y, right.z),
            accumulationCount: max(accumulationCount, 1),
            cameraUp: (up.x, up.y, up.z),
            samplesPerPixel: 1,
            cameraForward: (forward.x, forward.y, forward.z),
            maxBounces: 8,
            jitterX: jitter.x,
            jitterY: jitter.y,
            previousJitterX: previousJitter.x,
            previousJitterY: previousJitter.y,
            renderWidth: UInt32(renderW),
            renderHeight: UInt32(renderH),
            outputWidth: UInt32(outputWidth),
            outputHeight: UInt32(outputHeight),
            aperture: camera.aperture,
            focusDistance: camera.focusDistance,
            lightCount: UInt32(scene.lightCount),
            debugFlags: highlightWater ? 1 : 0,
            environmentTint: (environmentSettings.tint.x,
                              environmentSettings.tint.y,
                              environmentSettings.tint.z),
            environmentIntensity: environmentSettings.intensity,
            environmentSunDirection: (environmentSettings.sunDirection.x,
                                      environmentSettings.sunDirection.y,
                                      environmentSettings.sunDirection.z),
            environmentSunIntensity: environmentSettings.sunIntensity,
            exposure: currentExposure,
            exposurePadding0: 0,
            exposurePadding1: 0,
            exposurePadding2: 0
        )

        submittedFrames += 1
        let submittedFrame = submittedFrames
        let submitTime = CACurrentMediaTime()
        if verifyConfig.enabled && submittedFrame <= 8 {
            print("[Renderer] Submit frame #\(submittedFrame) accumulation=\(accumulationCount) render=\(renderW)x\(renderH) output=\(outputWidth)x\(outputHeight) moving=\(camera.isMoving)")
        }

        let canUseOIDN = effectiveRenderMode == .accumulation
            && selectedDenoiseMethod == .oidn
            && !isMoving
            && oidnDenoiser.isAvailable
            && oidnRuntimeDisabledReason == nil

        if selectedDenoiseMethod == .oidn && !oidnDenoiser.isAvailable {
            logOIDNAvailabilityIfNeeded()
        }

        if canUseOIDN {
            if verifyConfig.enabled {
                renderAccumulationWithOIDN(drawable: drawable,
                                           uniforms: &uniforms,
                                           accelStructure: accelStructure,
                                           scene: scene,
                                           submittedFrame: submittedFrame,
                                           submitTime: submitTime,
                                           onComplete: onComplete)
            } else {
                renderAccumulationWithOIDNAsync(drawable: drawable,
                                                uniforms: &uniforms,
                                                accelStructure: accelStructure,
                                                scene: scene,
                                                submittedFrame: submittedFrame,
                                                submitTime: submitTime,
                                                onComplete: onComplete)
            }
            previousCameraMoving = isMoving
            previousJitter = jitter
            frameIndex += 1
            return
        }

        // ── Encode ──
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        pathTracer.encodePathTrace(commandBuffer: commandBuffer,
                                   uniforms: &uniforms,
                                   accelStructure: accelStructure,
                                   scene: scene)

        let hdrSource: MTLTexture?
        let exposureSource: MTLTexture?
        switch effectiveRenderMode {
        case .raw:
            hdrSource = pathTracer.colorTexture
            exposureSource = hdrSource
        case .accumulation:
            if selectedDenoiseMethod != .none {
                pathTracer.encodeClampedAccumulation(commandBuffer: commandBuffer,
                                                     uniforms: &uniforms)
            } else {
                pathTracer.encodeAccumulation(commandBuffer: commandBuffer,
                                              uniforms: &uniforms)
            }
            pathTracer.copyAccumulationToHistory(commandBuffer: commandBuffer)
            exposureSource = pathTracer.accumulatedTexture
            if selectedDenoiseMethod != .none, let accumTex = pathTracer.accumulatedTexture {
                switch selectedDenoiseMethod {
                case .svgfPlus:
                    hdrSource = pathTracer.encodeAccumulationSVGFPlus(
                        commandBuffer: commandBuffer,
                        uniforms: &uniforms,
                        sourceTexture: accumTex) ?? accumTex
                case .oidn:
                    hdrSource = accumTex
                case .eaw:
                    hdrSource = pathTracer.encodePostATrous(
                        commandBuffer: commandBuffer,
                        uniforms: &uniforms,
                        sourceTexture: accumTex) ?? accumTex
                case .bilateral:
                    hdrSource = pathTracer.encodeCrossBilateral(
                        commandBuffer: commandBuffer,
                        uniforms: &uniforms,
                        sourceTexture: accumTex) ?? accumTex
                case .nlm:
                    hdrSource = pathTracer.encodeNLM(
                        commandBuffer: commandBuffer,
                        uniforms: &uniforms,
                        sourceTexture: accumTex) ?? accumTex
                case .none:
                    hdrSource = accumTex
                }
            } else {
                hdrSource = pathTracer.accumulatedTexture
            }
        case .svgf:
            hdrSource = pathTracer.encodeSVGF(commandBuffer: commandBuffer,
                                              uniforms: &uniforms) ?? pathTracer.colorTexture
            exposureSource = hdrSource
        case .metalfx:
            if let upscaler,
               let colorTexture = pathTracer.colorTexture,
               let depthTexture = pathTracer.depthTexture,
               let motionTexture = pathTracer.motionTexture {
                let metalFXInput: MTLTexture
                if isMoving {
                    metalFXInput = colorTexture
                } else {
                    pathTracer.encodeAccumulation(commandBuffer: commandBuffer,
                                                 uniforms: &uniforms)
                    pathTracer.copyAccumulationToHistory(commandBuffer: commandBuffer)
                    metalFXInput = pathTracer.accumulatedTexture ?? colorTexture
                }

                upscaler.encode(commandBuffer: commandBuffer,
                                colorTexture: metalFXInput,
                                depthTexture: depthTexture,
                                motionTexture: motionTexture,
                                jitterX: jitter.x,
                                jitterY: jitter.y)
                hdrSource = upscaler.outputTexture ?? metalFXInput
                exposureSource = metalFXInput
            } else {
                hdrSource = pathTracer.colorTexture
                exposureSource = hdrSource
            }
        case .metalfxSVGF:
            if let upscaler,
               let colorTexture = pathTracer.colorTexture,
               let depthTexture = pathTracer.depthTexture,
               let motionTexture = pathTracer.motionTexture {
                let metalFXInput: MTLTexture
                if isMoving {
                    // Spatial pre-denoise on raw 1-spp to help MetalFX temporal
                    metalFXInput = pathTracer.encodeSpatialDenoise(
                        commandBuffer: commandBuffer,
                        uniforms: &uniforms,
                        sourceTexture: colorTexture) ?? colorTexture
                } else {
                    // Accumulate for convergence, feed clean frame to MetalFX
                    pathTracer.encodeAccumulation(commandBuffer: commandBuffer,
                                                 uniforms: &uniforms)
                    pathTracer.copyAccumulationToHistory(commandBuffer: commandBuffer)
                    metalFXInput = pathTracer.accumulatedTexture ?? colorTexture
                }
                upscaler.encode(commandBuffer: commandBuffer,
                                colorTexture: metalFXInput,
                                depthTexture: depthTexture,
                                motionTexture: motionTexture,
                                jitterX: jitter.x,
                                jitterY: jitter.y)
                hdrSource = upscaler.outputTexture ?? metalFXInput
                exposureSource = metalFXInput
            } else {
                hdrSource = pathTracer.colorTexture
                exposureSource = hdrSource
            }
        case .auto:
            hdrSource = pathTracer.colorTexture
            exposureSource = hdrSource
        }
        previousCameraMoving = isMoving
        previousJitter = jitter

        submitFrame(commandBuffer: commandBuffer,
                    drawable: drawable,
                    uniforms: &uniforms,
                    hdrSource: hdrSource,
                    exposureSource: exposureSource,
                    submittedFrame: submittedFrame,
                    submitTime: submitTime,
                    onComplete: onComplete)

        frameIndex += 1
    }

    func resize(width: Int, height: Int) {
        outputWidth = max(1, width)
        outputHeight = max(1, height)
        resetTemporalState(syncCameraHistory: true)
    }

    // MARK: - Screenshot & Verification

    private func performVerifyCapture(texture: MTLTexture, hdrTexture: MTLTexture?, capturedFrame: UInt32) {
        let isReferenceCapture = isCapturingReference()
        let url = verifyOutputURL(capturedFrame: capturedFrame)
        let saved = ScreenshotCapture.capture(texture: texture, commandQueue: commandQueue, to: url)
        let metrics = ScreenshotCapture.computeMetrics(texture: texture, commandQueue: commandQueue)
        print("[VERIFY] capture frame=\(capturedFrame) render=\(selectedRenderMode.rawValue) denoise=\(selectedDenoiseMethod.rawValue)")
        if isReferenceCapture {
            print("[VERIFY] capture_role: reference")
        }
        metrics.printReport(screenshotPath: saved ? url.path : nil)
        if isReferenceCapture {
            verifyReferencePixels = ScreenshotCapture.readRGBA8Pixels(texture: texture, commandQueue: commandQueue)
            verifyReferenceHDRPixels = hdrTexture.flatMap {
                ScreenshotCapture.readRGBA32FloatPixels(texture: $0, commandQueue: commandQueue)
            }
            verifyReferenceLabel = url.lastPathComponent
            if verifyReferencePixels == nil {
                print("[VERIFY] ERROR: Failed to read reference pixels")
                verifyReferenceReady = false
                verifyReferenceTransitionPending = false
            } else {
                verifyReferenceReady = true
                verifyReferenceTransitionPending = true
            }
        } else if let hdrTexture,
                  let referenceHDRPixels = verifyReferenceHDRPixels,
                  let referenceLabel = verifyReferenceLabel,
                  let hdrReferenceMetrics = ScreenshotCapture.computeHDRReferenceMetrics(texture: hdrTexture,
                                                                                        commandQueue: commandQueue,
                                                                                        referencePixels: referenceHDRPixels) {
            hdrReferenceMetrics.printReport(referenceLabel: referenceLabel)
        } else if let referencePixels = verifyReferencePixels,
                  let referenceLabel = verifyReferenceLabel,
                  let referenceMetrics = ScreenshotCapture.computeReferenceMetrics(texture: texture,
                                                                                  commandQueue: commandQueue,
                                                                                  referencePixels: referencePixels) {
            referenceMetrics.printReport(referenceLabel: referenceLabel)
        }
        if highlightWater {
            let magentaCoverage = ScreenshotCapture.computeMagentaCoverage(texture: texture,
                                                                           commandQueue: commandQueue)
            print(String(format: "[VERIFY] magenta_pct: %.2f", magentaCoverage))
        }

        DispatchQueue.main.async {
            let referenceReadSucceeded = !isReferenceCapture || self.verifyReferencePixels != nil
            self.verifyAllPassed = self.verifyAllPassed && saved && metrics.passed && referenceReadSucceeded
            self.advanceVerifyState()
        }
    }

    private func performInteractiveScreenshot(texture: MTLTexture) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let desktopURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/rt_screenshot_\(timestamp).png")

        let saved = ScreenshotCapture.capture(texture: texture, commandQueue: commandQueue, to: desktopURL)
        let metrics = ScreenshotCapture.computeMetrics(texture: texture, commandQueue: commandQueue)
        metrics.printReport(screenshotPath: saved ? desktopURL.path : nil)
        if highlightWater {
            let magentaCoverage = ScreenshotCapture.computeMagentaCoverage(texture: texture,
                                                                           commandQueue: commandQueue)
            print(String(format: "[VERIFY] magenta_pct: %.2f", magentaCoverage))
        }
    }

    private func debugDescription(for status: MTLCommandBufferStatus) -> String {
        switch status {
        case .notEnqueued: return "notEnqueued"
        case .enqueued: return "enqueued"
        case .committed: return "committed"
        case .scheduled: return "scheduled"
        case .completed: return "completed"
        case .error: return "error"
        @unknown default: return "unknown"
        }
    }

    private func renderAccumulationWithOIDN(drawable: CAMetalDrawable,
                                            uniforms: inout Uniforms,
                                            accelStructure: MTLAccelerationStructure,
                                            scene: SceneGeometry,
                                            submittedFrame: UInt32,
                                            submitTime: CFTimeInterval,
                                            onComplete: (() -> Void)?) {
        guard let accumulationCommandBuffer = commandQueue.makeCommandBuffer() else {
            onComplete?()
            return
        }

        pathTracer.encodePathTrace(commandBuffer: accumulationCommandBuffer,
                                   uniforms: &uniforms,
                                   accelStructure: accelStructure,
                                   scene: scene)
        pathTracer.encodeClampedAccumulation(commandBuffer: accumulationCommandBuffer,
                                             uniforms: &uniforms)
        pathTracer.copyAccumulationToHistory(commandBuffer: accumulationCommandBuffer)
        if let accumulationTexture = pathTracer.accumulatedTexture {
            pathTracer.encodeExposureMeasurement(commandBuffer: accumulationCommandBuffer,
                                                sourceTexture: accumulationTexture)
        }

        accumulationCommandBuffer.commit()
        accumulationCommandBuffer.waitUntilCompleted()

        if accumulationCommandBuffer.status == .error {
            let errorText = accumulationCommandBuffer.error.map { String(describing: $0) } ?? "unknown"
            print("[Renderer] OIDN accumulation pass failed: \(errorText)")
            onComplete?()
            return
        }

        let measuredAverageLuminance = pathTracer.readMeasuredAverageLuminance()
        DispatchQueue.main.async {
            self.measuredAverageLuminance = measuredAverageLuminance
        }

        guard let accumulationTexture = pathTracer.accumulatedTexture else {
            onComplete?()
            return
        }

        let hdrSource = prepareOIDNTexture(from: accumulationTexture) ?? accumulationTexture
        guard let presentationCommandBuffer = commandQueue.makeCommandBuffer() else {
            onComplete?()
            return
        }

        submitFrame(commandBuffer: presentationCommandBuffer,
                    drawable: drawable,
                    uniforms: &uniforms,
                    hdrSource: hdrSource,
                    exposureSource: nil,
                    submittedFrame: submittedFrame,
                    submitTime: submitTime,
                    onComplete: onComplete)
    }

    private func renderAccumulationWithOIDNAsync(drawable: CAMetalDrawable,
                                                 uniforms: inout Uniforms,
                                                 accelStructure: MTLAccelerationStructure,
                                                 scene: SceneGeometry,
                                                 submittedFrame: UInt32,
                                                 submitTime: CFTimeInterval,
                                                 onComplete: (() -> Void)?) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            onComplete?()
            return
        }

        pathTracer.encodePathTrace(commandBuffer: commandBuffer,
                                   uniforms: &uniforms,
                                   accelStructure: accelStructure,
                                   scene: scene)
        pathTracer.encodeClampedAccumulation(commandBuffer: commandBuffer,
                                             uniforms: &uniforms)
        pathTracer.copyAccumulationToHistory(commandBuffer: commandBuffer)

        guard let accumulationTexture = pathTracer.accumulatedTexture else {
            onComplete?()
            return
        }

        let shouldScheduleOIDN = shouldScheduleOIDNDenoise(for: accumulationCount)
        let includeAuxiliaryInputs = oidnUseAuxiliaryInputs
        let stagedColorTexture = shouldScheduleOIDN
            ? stageSharedTextureSnapshot(from: accumulationTexture, commandBuffer: commandBuffer)
            : nil
        let stagedAlbedoTexture = shouldScheduleOIDN && includeAuxiliaryInputs
            ? pathTracer.albedoTexture.flatMap { stageSharedTextureSnapshot(from: $0, commandBuffer: commandBuffer) }
            : nil
        let stagedNormalTexture = shouldScheduleOIDN && includeAuxiliaryInputs
            ? pathTracer.normalTexture.flatMap { stageSharedTextureSnapshot(from: $0, commandBuffer: commandBuffer) }
            : nil
        let scheduledOIDNFrame = accumulationCount

        if stagedColorTexture != nil {
            oidnJobInFlight = true
            oidnJobStartTime = CACurrentMediaTime()
            logOIDNEvent("schedule accumulationFrame=\(scheduledOIDNFrame) generation=\(oidnGeneration)")
        }

        let presentationSource = currentOIDNPresentationTexture() ?? accumulationTexture

        submitFrame(commandBuffer: commandBuffer,
                    drawable: drawable,
                    uniforms: &uniforms,
                    hdrSource: presentationSource,
                    exposureSource: accumulationTexture,
                    submittedFrame: submittedFrame,
                    submitTime: submitTime,
                    onComplete: onComplete,
                    postCompletion: { [weak self] in
                        guard let self,
                              let stagedColorTexture else {
                            return
                        }
                        self.scheduleOIDNDenoise(colorTexture: stagedColorTexture,
                                                 albedoTexture: stagedAlbedoTexture,
                                                 normalTexture: stagedNormalTexture,
                                                                                                 accumulationFrame: scheduledOIDNFrame,
                                                 generation: self.oidnGeneration,
                                                 usedAuxiliaryInputs: includeAuxiliaryInputs)
                    })
    }

    private func submitFrame(commandBuffer: MTLCommandBuffer,
                             drawable: CAMetalDrawable,
                             uniforms: inout Uniforms,
                             hdrSource: MTLTexture?,
                             exposureSource: MTLTexture?,
                             submittedFrame: UInt32,
                             submitTime: CFTimeInterval,
                             onComplete: (() -> Void)?,
                             postCompletion: (() -> Void)? = nil) {
        let hasExposureMeasurement = exposureSource != nil
        if let exposureSource {
            pathTracer.encodeExposureMeasurement(commandBuffer: commandBuffer,
                                                sourceTexture: exposureSource)
        }
        if let hdrSource {
            let tonemapSource = pathTracer.encodeBloom(commandBuffer: commandBuffer,
                                                       sourceTexture: hdrSource) ?? hdrSource
            pathTracer.encodeTonemap(commandBuffer: commandBuffer,
                                     uniforms: &uniforms,
                                     sourceTexture: tonemapSource)
        }

        let verifyTargetFrame = currentVerifyTargetFrame()
        let needsVerify = verifyConfig.enabled
            && !verifyCompleted
            && verifyTargetFrame != nil
            && accumulationCount >= verifyTargetFrame!
        let needsScreenshot = pendingScreenshot
        if needsScreenshot { pendingScreenshot = false }
        if needsVerify || needsScreenshot { captureInProgress = true }

        let verifyHDRTexture = makeVerifyHDRTexture(from: hdrSource,
                                                    commandBuffer: commandBuffer,
                                                    needsVerify: needsVerify)

        guard let tonemapped = pathTracer.tonemappedTexture else {
            onComplete?()
            return
        }

        if let blit = commandBuffer.makeBlitCommandEncoder() {
            let drawableTexture = drawable.texture
            let srcSize = MTLSize(width: min(tonemapped.width, drawableTexture.width),
                                  height: min(tonemapped.height, drawableTexture.height),
                                  depth: 1)
            blit.copy(from: tonemapped,
                      sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: srcSize,
                      to: drawableTexture,
                      destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blit.endEncoding()
        }

        commandBuffer.present(drawable)
        commandBuffer.addCompletedHandler { [weak self] buffer in
            if let self = self {
                if hasExposureMeasurement {
                    let measuredAverageLuminance = self.pathTracer.readMeasuredAverageLuminance()
                    DispatchQueue.main.async {
                        self.measuredAverageLuminance = measuredAverageLuminance
                    }
                }

                self.completedFrames += 1
                let elapsedMs = (CACurrentMediaTime() - submitTime) * 1000.0
                if self.verifyConfig.enabled && submittedFrame <= 8 {
                    let errorText = buffer.error.map { String(describing: $0) } ?? "none"
                    print(String(format: "[Renderer] Complete frame #%u status=%@ elapsed=%.1fms error=%@ accumulation=%u",
                                 submittedFrame,
                                 self.debugDescription(for: buffer.status),
                                 elapsedMs,
                                 errorText,
                                 self.accumulationCount))
                }

                if let tex = self.pathTracer.tonemappedTexture {
                    if needsVerify {
                        self.performVerifyCapture(texture: tex,
                                                  hdrTexture: verifyHDRTexture,
                                                  capturedFrame: verifyTargetFrame ?? self.accumulationCount)
                    } else if needsScreenshot {
                        self.performInteractiveScreenshot(texture: tex)
                        DispatchQueue.main.async {
                            self.captureInProgress = false
                        }
                    }
                }
            }
            postCompletion?()
            onComplete?()
        }

        commandBuffer.commit()
    }

    private func makeVerifyHDRTexture(from hdrSource: MTLTexture?,
                                      commandBuffer: MTLCommandBuffer,
                                      needsVerify: Bool) -> MTLTexture? {
        guard needsVerify,
              let hdrSource,
              hdrSource.pixelFormat == .rgba32Float else {
            return nil
        }
        return stageSharedTextureSnapshot(from: hdrSource, commandBuffer: commandBuffer)
    }

    private func prepareOIDNTexture(from sourceTexture: MTLTexture) -> MTLTexture? {
        logOIDNAvailabilityIfNeeded()
        guard let sourcePixels = ScreenshotCapture.readRGBA32FloatPixels(texture: sourceTexture,
                                                                         commandQueue: commandQueue) else {
            print("[Renderer] OIDN failed to read the accumulation buffer; falling back to raw accumulation")
            return nil
        }

        let albedoPixels = pathTracer.albedoTexture.flatMap {
            ScreenshotCapture.readRGBA16FloatPixels(texture: $0, commandQueue: commandQueue)
        }
        let normalPixels = pathTracer.normalTexture.flatMap {
            ScreenshotCapture.readRGBA16FloatPixels(texture: $0, commandQueue: commandQueue)
        }

        guard let filteredPixels = oidnDenoiser.denoiseHDR(colorRGBA: sourcePixels,
                                                           albedoRGBA: albedoPixels,
                                                           normalRGBA: normalPixels,
                                                           width: sourceTexture.width,
                                                           height: sourceTexture.height),
              let outputTexture = makeOIDNTexture(width: sourceTexture.width,
                                                  height: sourceTexture.height,
                                                  pixels: filteredPixels) else {
            print("[Renderer] OIDN failed; falling back to raw accumulation")
            return nil
        }

        return outputTexture
    }

    private func makeOIDNTexture(width: Int, height: Int, pixels: [Float]) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                            width: width,
                                                            height: height,
                                                            mipmapped: false)
        desc.storageMode = .shared
        desc.usage = [.shaderRead]
        guard let texture = device.makeTexture(descriptor: desc) else {
            return nil
        }

        let bytesPerRow = width * MemoryLayout<Float>.stride * 4
        pixels.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                texture.replace(region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                                  size: MTLSize(width: width, height: height, depth: 1)),
                                mipmapLevel: 0,
                                withBytes: baseAddress,
                                bytesPerRow: bytesPerRow)
            }
        }
        return texture
    }

    private func logOIDNAvailabilityIfNeeded() {
        guard !hasLoggedOIDNAvailability else {
            return
        }

        hasLoggedOIDNAvailability = true
        print("[Renderer] OIDN: \(oidnDenoiser.availabilityDescription)")
    }

    private func stageSharedTextureSnapshot(from sourceTexture: MTLTexture,
                                            commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: sourceTexture.pixelFormat,
                                                            width: sourceTexture.width,
                                                            height: sourceTexture.height,
                                                            mipmapped: false)
        desc.storageMode = .shared
        desc.usage = []
        guard let snapshotTexture = device.makeTexture(descriptor: desc) else {
            return nil
        }

        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: sourceTexture,
                      sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: MTLSize(width: sourceTexture.width, height: sourceTexture.height, depth: 1),
                      to: snapshotTexture,
                      destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blit.endEncoding()
        }

        return snapshotTexture
    }

    private func shouldScheduleOIDNDenoise(for accumulationFrame: UInt32) -> Bool {
        guard !oidnJobInFlight, accumulationFrame > oidnLatestFrame else {
            return false
        }

        return accumulationFrame <= 4 || accumulationFrame.isMultiple(of: 4)
    }

    private func shouldResetOIDNForAccumulationRestart(stoppedMoving: Bool) -> Bool {
        guard accumulationCount <= 1 else {
            return false
        }

        if stoppedMoving {
            return true
        }

        return oidnLatestFrame >= accumulationCount
            || oidnLatestTexture != nil
            || oidnJobInFlight
    }

    private func currentOIDNPresentationTexture() -> MTLTexture? {
        guard let oidnLatestTexture,
              let accumulationTexture = pathTracer.accumulatedTexture,
              oidnLatestTexture.width == accumulationTexture.width,
              oidnLatestTexture.height == accumulationTexture.height else {
            return nil
        }

        return oidnLatestTexture
    }

    private func scheduleOIDNDenoise(colorTexture: MTLTexture,
                                     albedoTexture: MTLTexture?,
                                     normalTexture: MTLTexture?,
                                     accumulationFrame: UInt32,
                                     generation: UInt64,
                                     usedAuxiliaryInputs: Bool) {
        oidnQueue.async { [weak self] in
            guard let self else {
                return
            }

            let colorPixels = ScreenshotCapture.readRGBA32FloatPixels(texture: colorTexture,
                                                                      commandQueue: self.commandQueue)
            let albedoPixels = albedoTexture.flatMap {
                ScreenshotCapture.readRGBA16FloatPixels(texture: $0, commandQueue: self.commandQueue)
            }
            let normalPixels = normalTexture.flatMap {
                ScreenshotCapture.readRGBA16FloatPixels(texture: $0, commandQueue: self.commandQueue)
            }

            let filteredPixels = colorPixels.flatMap {
                self.oidnDenoiser.denoiseHDR(colorRGBA: $0,
                                             albedoRGBA: albedoPixels,
                                             normalRGBA: normalPixels,
                                             width: colorTexture.width,
                                             height: colorTexture.height)
            }
            let filteredTexture = filteredPixels.flatMap {
                self.makeOIDNTexture(width: colorTexture.width,
                                     height: colorTexture.height,
                                     pixels: $0)
            }

            DispatchQueue.main.async {
                guard generation == self.oidnGeneration,
                      self.selectedDenoiseMethod == .oidn else {
                    self.logOIDNEvent("discard accumulationFrame=\(accumulationFrame) generation=\(generation) currentGeneration=\(self.oidnGeneration)")
                                        self.oidnJobStartTime = nil
                    self.oidnJobInFlight = false
                    return
                }

                if let filteredTexture {
                    self.oidnLatestTexture = filteredTexture
                    self.oidnLatestFrame = accumulationFrame
                    self.oidnRuntimeDisabledReason = nil
                    self.oidnUseAuxiliaryInputs = usedAuxiliaryInputs
                    self.logOIDNEvent("complete accumulationFrame=\(accumulationFrame) generation=\(generation)")
                } else {
                    self.logOIDNEvent("failed accumulationFrame=\(accumulationFrame) generation=\(generation)")
                }
                self.oidnJobStartTime = nil
                self.oidnJobInFlight = false
            }
        }
    }

    private func recoverFromStalledOIDNJob(now: CFTimeInterval) {
        guard oidnJobInFlight,
              let oidnJobStartTime,
              now - oidnJobStartTime >= oidnJobTimeout else {
            return
        }

        let elapsed = now - oidnJobStartTime
        if oidnUseAuxiliaryInputs {
            logOIDNEvent("watchdog fallback to color-only after \(String(format: "%.2f", elapsed))s generation=\(oidnGeneration)")
            oidnRuntimeDisabledReason = nil
            oidnUseAuxiliaryInputs = false
            resetOIDNState()
            return
        }

        logOIDNEvent("watchdog disabled OIDN after \(String(format: "%.2f", elapsed))s generation=\(oidnGeneration)")
        oidnRuntimeDisabledReason = "OIDN timed out"
        resetOIDNState()
    }

    private func resetOIDNRuntimeConfiguration() {
        oidnUseAuxiliaryInputs = true
        oidnRuntimeDisabledReason = nil
    }

    private func logOIDNEvent(_ message: String) {
        guard shouldLogOIDNEvents else {
            return
        }
        print("[OIDN] \(message)")
    }

    private func updateExposure(deltaTime: Float) {
        let targetExposure = desiredExposure(forAverageLuminance: measuredAverageLuminance)
        let adaptationRate = targetExposure < currentExposure
            ? darkToBrightAdaptationRate
            : brightToDarkAdaptationRate
        let blend = 1.0 - exp(-adaptationRate * deltaTime)
        currentExposure += (targetExposure - currentExposure) * blend
        currentExposure = min(max(currentExposure, minAutoExposure), maxAutoExposure)
    }

    private func desiredExposure(forAverageLuminance averageLuminance: Float) -> Float {
        let safeLuminance = max(averageLuminance, 1e-4)
        let targetExposure = exposureKeyValue / safeLuminance
        return min(max(targetExposure, minAutoExposure), maxAutoExposure)
    }

    private func preferredRenderMode(isMoving: Bool) -> RenderMode {
        if selectedRenderMode == .auto {
            return .svgf
        }

        return selectedRenderMode
    }

    private func resolvedRenderMode(requestedMode: RenderMode) -> RenderMode {
        if (requestedMode == .metalfx || requestedMode == .metalfxSVGF) && upscaler?.isAvailable != true {
            return .raw
        }

        return requestedMode
    }

    private func resetTemporalState(syncCameraHistory: Bool) {
        accumulationCount = 0
        frameIndex = 0
        camera.resetAccumulation()
        upscaler?.reset()
        resetOIDNState()
        previousJitter = camera.jitterOffset
        if syncCameraHistory {
            camera.syncHistory()
        }
    }

    private func resetOIDNState() {
        let hadState = oidnLatestTexture != nil || oidnLatestFrame > 0 || oidnJobInFlight
        oidnGeneration += 1
        oidnLatestTexture = nil
        oidnLatestFrame = 0
        oidnJobStartTime = nil
        oidnJobInFlight = false
        if hadState {
            logOIDNEvent("reset generation=\(oidnGeneration)")
        }
    }

    private func resetExposureState() {
        measuredAverageLuminance = exposureKeyValue
        currentExposure = desiredExposure(forAverageLuminance: measuredAverageLuminance)
        lastExposureUpdateTime = CACurrentMediaTime()
    }

    // MARK: - Helpers

    private func findAssetsDirectory() -> URL {
        // Look for assets/maps/ relative to the executable, or in the project directory
        let fileManager = FileManager.default

        // Try working directory
        let cwdAssets = URL(fileURLWithPath: fileManager.currentDirectoryPath)
            .appendingPathComponent("assets/maps")
        if fileManager.fileExists(atPath: cwdAssets.path) {
            return cwdAssets
        }

        // Try next to the executable
        let execURL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        let execAssets = execURL.appendingPathComponent("assets/maps")
        if fileManager.fileExists(atPath: execAssets.path) {
            return execAssets
        }

        // Try the app bundle
        if let resourcePath = Bundle.main.resourcePath {
            let bundleAssets = URL(fileURLWithPath: resourcePath).appendingPathComponent("assets/maps")
            if fileManager.fileExists(atPath: bundleAssets.path) {
                return bundleAssets
            }
        }

        // Try standard dev path
        let devAssets = URL(fileURLWithPath: "/Users/vaseo/dev/rt/assets/maps")
        if fileManager.fileExists(atPath: devAssets.path) {
            return devAssets
        }

        // Fallback: create assets directory in working dir
        try? fileManager.createDirectory(at: cwdAssets, withIntermediateDirectories: true)
        return cwdAssets
    }

    private func currentVerifyTargetFrame() -> UInt32? {
        guard verifyConfig.enabled else {
            return nil
        }

        if isCapturingReference(), let verifyReferenceFrame {
            return verifyReferenceFrame
        }

        guard verifyCheckpointIndex < verifyCheckpointFrames.count else {
            return nil
        }

        return verifyCheckpointFrames[verifyCheckpointIndex]
    }

    private func verifyOutputURL(capturedFrame: UInt32) -> URL {
        let hasMultipleCaptures = verifyReferenceFrame != nil || verifyCheckpointFrames.count > 1 || verifySweepMethods.count > 1
        if !hasMultipleCaptures {
            return URL(fileURLWithPath: verifyConfig.outputPath)
        }

        let fileManager = FileManager.default
        let baseDirectoryURL: URL
        if let outputDirectory = verifyConfig.outputDirectory {
            baseDirectoryURL = URL(fileURLWithPath: outputDirectory, isDirectory: true)
        } else {
            let outputURL = URL(fileURLWithPath: verifyConfig.outputPath)
            if outputURL.pathExtension.isEmpty {
                baseDirectoryURL = outputURL
            } else {
                baseDirectoryURL = outputURL.deletingPathExtension()
            }
        }

        try? fileManager.createDirectory(at: baseDirectoryURL, withIntermediateDirectories: true)

        let frameToken = String(format: "%04u", capturedFrame)
        let renderToken = selectedRenderMode.rawValue
        let denoiseToken = selectedDenoiseMethod.rawValue
        let prefix = isCapturingReference() ? "reference_" : ""
        let fileName = "\(prefix)\(renderToken)_\(denoiseToken)_f\(frameToken).png"
        return baseDirectoryURL.appendingPathComponent(fileName)
    }

    private func advanceVerifyState() {
        if verifyReferenceFrame != nil && !verifyReferenceReady {
            verifyCompleted = true
            let exitCode: Int32 = 1
            NSApplication.shared.reply(toApplicationShouldTerminate: true)
            exit(exitCode)
        }

        if verifyReferenceTransitionPending {
            verifyReferenceTransitionPending = false
            selectedDenoiseMethod = verifySweepMethods.first ?? selectedDenoiseMethod
            print("[VERIFY] Captured reference; switching to test denoiser \(selectedDenoiseMethod.displayName)")
            resetTemporalState(syncCameraHistory: true)
            resetExposureState()
            captureInProgress = false
            return
        }

        if verifyCheckpointIndex + 1 < verifyCheckpointFrames.count {
            verifyCheckpointIndex += 1
            captureInProgress = false
            return
        }

        if verifySweepIndex + 1 < verifySweepMethods.count {
            verifySweepIndex += 1
            selectedDenoiseMethod = verifySweepMethods[verifySweepIndex]
            verifyCheckpointIndex = 0
            print("[VERIFY] Switching accumulation denoiser to \(selectedDenoiseMethod.displayName)")
            resetTemporalState(syncCameraHistory: true)
            resetExposureState()
            captureInProgress = false
            return
        }

        verifyCompleted = true
        let exitCode: Int32 = verifyAllPassed ? 0 : 1
        NSApplication.shared.reply(toApplicationShouldTerminate: true)
        exit(exitCode)
    }

    private func isCapturingReference() -> Bool {
        verifyReferenceFrame != nil && !verifyReferenceReady
    }

    // MARK: - Test Scene (fallback when no BSP available)

    private func generateTestScene() -> BSPData {
        // Cornell box-ish scene for testing
        var vertices: [ParsedVertex] = []
        var indices: [UInt32] = []
        var materials: [ParsedMaterial] = []

        // Materials
        materials.append(ParsedMaterial(name: "white", albedo: SIMD3<Float>(0.73, 0.73, 0.73), emissiveStrength: 0,
                                        emissiveColor: .zero, surfaceType: MaterialSurfaceType.diffuse.rawValue,
                                        roughness: 0.82, metallic: 0.0, transmissive: 0.0, ior: 1.5,
                                        textureWidth: 0, textureHeight: 0, texturePixels: nil))
        materials.append(ParsedMaterial(name: "red", albedo: SIMD3<Float>(0.65, 0.05, 0.05), emissiveStrength: 0,
                                        emissiveColor: .zero, surfaceType: MaterialSurfaceType.diffuse.rawValue,
                                        roughness: 0.76, metallic: 0.0, transmissive: 0.0, ior: 1.5,
                                        textureWidth: 0, textureHeight: 0, texturePixels: nil))
        materials.append(ParsedMaterial(name: "green", albedo: SIMD3<Float>(0.12, 0.45, 0.15), emissiveStrength: 0,
                                        emissiveColor: .zero, surfaceType: MaterialSurfaceType.diffuse.rawValue,
                                        roughness: 0.76, metallic: 0.0, transmissive: 0.0, ior: 1.5,
                                        textureWidth: 0, textureHeight: 0, texturePixels: nil))
        materials.append(ParsedMaterial(name: "metal", albedo: SIMD3<Float>(0.72, 0.73, 0.76), emissiveStrength: 0,
                                        emissiveColor: .zero, surfaceType: MaterialSurfaceType.metal.rawValue,
                                        roughness: 0.14, metallic: 0.96, transmissive: 0.0, ior: 1.5,
                                        textureWidth: 0, textureHeight: 0, texturePixels: nil))
        materials.append(ParsedMaterial(name: "light", albedo: SIMD3<Float>(1.0, 0.9, 0.7), emissiveStrength: 15.0,
                                        emissiveColor: SIMD3<Float>(1.0, 0.9, 0.7), surfaceType: MaterialSurfaceType.emissive.rawValue,
                                        roughness: 0.25, metallic: 0.0, transmissive: 0.0, ior: 1.0,
                                        textureWidth: 0, textureHeight: 0, texturePixels: nil))

        let s: Float = 200 // half-size of room

        func addQuad(v0: SIMD3<Float>, v1: SIMD3<Float>, v2: SIMD3<Float>, v3: SIMD3<Float>,
                     normal: SIMD3<Float>, material: UInt32) {
            let base = UInt32(vertices.count)
            for v in [v0, v1, v2, v3] {
                vertices.append(ParsedVertex(position: v, normal: normal,
                                             uv: SIMD2<Float>(0, 0), materialIndex: material))
            }
            indices.append(contentsOf: [base, base+1, base+2, base, base+2, base+3])
        }

        // Floor (white)
        addQuad(v0: SIMD3(-s, 0, -s), v1: SIMD3(s, 0, -s),
                v2: SIMD3(s, 0, s), v3: SIMD3(-s, 0, s),
                normal: SIMD3(0, 1, 0), material: 0)

        // Ceiling (white)
        addQuad(v0: SIMD3(-s, s*2, s), v1: SIMD3(s, s*2, s),
                v2: SIMD3(s, s*2, -s), v3: SIMD3(-s, s*2, -s),
                normal: SIMD3(0, -1, 0), material: 0)

        // Back wall (white)
        addQuad(v0: SIMD3(-s, 0, -s), v1: SIMD3(-s, s*2, -s),
                v2: SIMD3(s, s*2, -s), v3: SIMD3(s, 0, -s),
                normal: SIMD3(0, 0, 1), material: 0)

        // Left wall (red)
        addQuad(v0: SIMD3(-s, 0, s), v1: SIMD3(-s, s*2, s),
                v2: SIMD3(-s, s*2, -s), v3: SIMD3(-s, 0, -s),
                normal: SIMD3(1, 0, 0), material: 1)

        // Right wall (green)
        addQuad(v0: SIMD3(s, 0, -s), v1: SIMD3(s, s*2, -s),
                v2: SIMD3(s, s*2, s), v3: SIMD3(s, 0, s),
                normal: SIMD3(-1, 0, 0), material: 2)

        // Ceiling light
        let ls: Float = 60
        addQuad(v0: SIMD3(-ls, s*2 - 1, -ls), v1: SIMD3(ls, s*2 - 1, -ls),
                v2: SIMD3(ls, s*2 - 1, ls), v3: SIMD3(-ls, s*2 - 1, ls),
            normal: SIMD3(0, -1, 0), material: 4)

        // Box 1 (tall)
        let bx: Float = -60, bz: Float = -40, bw: Float = 60, bh: Float = 180
        addQuad(v0: SIMD3(bx, 0, bz), v1: SIMD3(bx, bh, bz),
                v2: SIMD3(bx+bw, bh, bz), v3: SIMD3(bx+bw, 0, bz),
                normal: SIMD3(0, 0, 1), material: 0)
        addQuad(v0: SIMD3(bx+bw, 0, bz-bw), v1: SIMD3(bx+bw, bh, bz-bw),
                v2: SIMD3(bx, bh, bz-bw), v3: SIMD3(bx, 0, bz-bw),
                normal: SIMD3(0, 0, -1), material: 0)
        addQuad(v0: SIMD3(bx, 0, bz-bw), v1: SIMD3(bx, bh, bz-bw),
                v2: SIMD3(bx, bh, bz), v3: SIMD3(bx, 0, bz),
                normal: SIMD3(-1, 0, 0), material: 0)
        addQuad(v0: SIMD3(bx+bw, 0, bz), v1: SIMD3(bx+bw, bh, bz),
                v2: SIMD3(bx+bw, bh, bz-bw), v3: SIMD3(bx+bw, 0, bz-bw),
                normal: SIMD3(1, 0, 0), material: 0)
        addQuad(v0: SIMD3(bx, bh, bz), v1: SIMD3(bx, bh, bz-bw),
                v2: SIMD3(bx+bw, bh, bz-bw), v3: SIMD3(bx+bw, bh, bz),
                normal: SIMD3(0, 1, 0), material: 0)

        // Box 2 (short)
        let b2x: Float = 50, b2z: Float = 60, b2w: Float = 60, b2h: Float = 90
        addQuad(v0: SIMD3(b2x, 0, b2z), v1: SIMD3(b2x, b2h, b2z),
                v2: SIMD3(b2x+b2w, b2h, b2z), v3: SIMD3(b2x+b2w, 0, b2z),
            normal: SIMD3(0, 0, 1), material: 3)
        addQuad(v0: SIMD3(b2x+b2w, 0, b2z-b2w), v1: SIMD3(b2x+b2w, b2h, b2z-b2w),
                v2: SIMD3(b2x, b2h, b2z-b2w), v3: SIMD3(b2x, 0, b2z-b2w),
            normal: SIMD3(0, 0, -1), material: 3)
        addQuad(v0: SIMD3(b2x, 0, b2z-b2w), v1: SIMD3(b2x, b2h, b2z-b2w),
                v2: SIMD3(b2x, b2h, b2z), v3: SIMD3(b2x, 0, b2z),
            normal: SIMD3(-1, 0, 0), material: 3)
        addQuad(v0: SIMD3(b2x+b2w, 0, b2z), v1: SIMD3(b2x+b2w, b2h, b2z),
                v2: SIMD3(b2x+b2w, b2h, b2z-b2w), v3: SIMD3(b2x+b2w, 0, b2z-b2w),
            normal: SIMD3(1, 0, 0), material: 3)
        addQuad(v0: SIMD3(b2x, b2h, b2z), v1: SIMD3(b2x, b2h, b2z-b2w),
                v2: SIMD3(b2x+b2w, b2h, b2z-b2w), v3: SIMD3(b2x+b2w, b2h, b2z),
            normal: SIMD3(0, 1, 0), material: 3)

        return BSPData(
            vertices: vertices,
            indices: indices,
            materials: materials,
            entityString: "",
            spawnPosition: SIMD3<Float>(0, 80, 180),
            spawnAngle: 0,
            preferredLiquidSurface: nil,
            environment: .default
        )
    }
}
