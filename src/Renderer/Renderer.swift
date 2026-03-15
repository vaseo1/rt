import AppKit
import Metal
import MetalFX
import QuartzCore
import simd

// ─── Renderer ────────────────────────────────────────────────────────────────
//
// Orchestrates the full render pipeline:
//   1. Path trace at full resolution, 8 bounces
//   2. Accumulate over frames (reset on camera move)
//   3. Tonemap + present to drawable

class Renderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let camera = Camera()

    private var pathTracer: PathTracer
    private var accelBuilder: AccelStructureBuilder
    private var scene: SceneGeometry
    private var upscaler: MetalFXUpscaler?

    // Resolution
    private var outputWidth: Int = 2560
    private var outputHeight: Int = 1440

    // Accumulation
    private(set) var accumulationCount: UInt32 = 0
    private var frameIndex: UInt32 = 0

    // State
    private var sceneLoaded = false

    // Verification
    var verifyConfig = VerifyConfig()
    var pendingScreenshot = false
    private var verifyCompleted = false
    private(set) var captureInProgress = false
    private var submittedFrames: UInt32 = 0
    private var completedFrames: UInt32 = 0

    init(device: MTLDevice, view: GameView) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        self.pathTracer = PathTracer(device: device, commandQueue: commandQueue)
        self.accelBuilder = AccelStructureBuilder(device: device, commandQueue: commandQueue)
        self.scene = SceneGeometry(device: device)

        let scale = view.window?.backingScaleFactor ?? 2.0
        outputWidth = Int(view.bounds.width * scale)
        outputHeight = Int(view.bounds.height * scale)

        loadScene()
    }

    // MARK: - Scene Loading

    private func loadScene() {
        // Try to find Quake assets
        let assetsDir = findAssetsDirectory()
        var bspData: BSPData?

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
        let bspPath = assetsDir.appendingPathComponent("e1m1.bsp")
        if FileManager.default.fileExists(atPath: bspPath.path) {
            print("[Renderer] Loading BSP from: \(bspPath.path)")
            do {
                bspData = try BSPLoader().load(from: bspPath, palette: palette)
            } catch {
                print("[Renderer] Failed to load BSP: \(error)")
            }
        }

        // 2. Try extracting from pak0.pak
        if bspData == nil {
            if FileManager.default.fileExists(atPath: pakPath.path) {
                print("[Renderer] Loading from PAK: \(pakPath.path)")
                do {
                    let pakData = try PAKLoader().extractBSP(from: pakPath, mapName: "maps/e1m1.bsp")
                    bspData = try BSPLoader().load(data: pakData, palette: palette)
                } catch {
                    print("[Renderer] Failed to load PAK: \(error)")
                }
            }
        }

        // 3. Fallback: generate a test scene
        if bspData == nil {
            print("[Renderer] No Quake assets found. Generating test scene.")
            print("[Renderer] Place pak0.pak or e1m1.bsp in: \(assetsDir.path)")
            bspData = generateTestScene()
        }

        guard let data = bspData else { return }

        // Upload to GPU
        scene.loadFromBSP(data)

        // Set camera to spawn position
        camera.position = data.spawnPosition
        camera.yaw = data.spawnAngle

        // Build acceleration structure
        accelBuilder.build(scene: scene)

        sceneLoaded = (accelBuilder.accelerationStructure != nil)

        if sceneLoaded {
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

        // ── Render resolution ──
        let renderW = outputWidth
        let renderH = outputHeight

        camera.aspectRatio = Float(outputWidth) / Float(outputHeight)

        // ── Accumulation logic ──
        if camera.isMoving {
            accumulationCount = 1
            camera.resetAccumulation()
        } else {
            accumulationCount += 1
        }
        camera.advanceJitter()

        // ── Build uniforms ──
        let jitter = camera.jitterOffset
        let pos = camera.position
        var uniforms = Uniforms(
            inverseViewProjection: camera.inverseViewProjectionMatrix,
            previousViewProjection: camera.previousViewProjectionMatrix,
            cameraPosition: (pos.x, pos.y, pos.z),
            frameIndex: frameIndex,
            accumulationCount: accumulationCount,
            samplesPerPixel: 1,
            maxBounces: 8,
            jitterX: jitter.x,
            jitterY: jitter.y,
            renderWidth: UInt32(renderW),
            renderHeight: UInt32(renderH),
            outputWidth: UInt32(outputWidth),
            outputHeight: UInt32(outputHeight),
            aperture: camera.aperture,
            focusDistance: camera.focusDistance,
            lightCount: UInt32(scene.lightCount)
        )

        // ── Ensure textures ──
        let texturesChanged = pathTracer.ensureTextures(renderWidth: renderW, renderHeight: renderH,
                                 outputWidth: outputWidth, outputHeight: outputHeight)
        if texturesChanged {
            accumulationCount = 1
            camera.resetAccumulation()
            uniforms.accumulationCount = accumulationCount
        }

        // ── Encode ──
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        submittedFrames += 1
        let submittedFrame = submittedFrames
        let submitTime = CACurrentMediaTime()
        if verifyConfig.enabled && submittedFrame <= 8 {
            print("[Renderer] Submit frame #\(submittedFrame) accumulation=\(accumulationCount) render=\(renderW)x\(renderH) output=\(outputWidth)x\(outputHeight) moving=\(camera.isMoving)")
        }

        pathTracer.encode(commandBuffer: commandBuffer,
                          uniforms: &uniforms,
                          accelStructure: accelStructure,
                          scene: scene)

        // ── Present ──
        guard let tonemapped = pathTracer.tonemappedTexture else { return }

        // Blit tonemapped result to drawable
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

        // Capture screenshot/verify state before the completed handler
        let needsVerify = verifyConfig.enabled && !verifyCompleted && accumulationCount >= verifyConfig.targetFrames
        let needsScreenshot = pendingScreenshot
        if needsVerify { verifyCompleted = true }
        if needsScreenshot { pendingScreenshot = false }
        if needsVerify || needsScreenshot { captureInProgress = true }

        commandBuffer.addCompletedHandler { [weak self] buffer in
            // Screenshot / verify capture (GPU work is done, off main thread)
            if let self = self {
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
                    self.performVerifyCapture(texture: tex)
                        defer {
                            if !needsVerify {
                                DispatchQueue.main.async {
                                    self.captureInProgress = false
                                }
                            }
                        }
                } else if needsScreenshot {
                    self.performInteractiveScreenshot(texture: tex)
                }
            }
            }
            onComplete?()
        }

        commandBuffer.commit()

        frameIndex += 1
    }

    func resize(width: Int, height: Int) {
        outputWidth = max(1, width)
        outputHeight = max(1, height)
        accumulationCount = 0
    }

    // MARK: - Screenshot & Verification

    private func performVerifyCapture(texture: MTLTexture) {
        let url = URL(fileURLWithPath: verifyConfig.outputPath)
        let saved = ScreenshotCapture.capture(texture: texture, commandQueue: commandQueue, to: url)
        let metrics = ScreenshotCapture.computeMetrics(texture: texture, commandQueue: commandQueue)
        metrics.printReport(screenshotPath: saved ? url.path : nil)

        // Exit with appropriate code
        let exitCode: Int32 = metrics.passed ? 0 : 1
        DispatchQueue.main.async {
            NSApplication.shared.reply(toApplicationShouldTerminate: true)
            exit(exitCode)
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

    // MARK: - Test Scene (fallback when no BSP available)

    private func generateTestScene() -> BSPData {
        // Cornell box-ish scene for testing
        var vertices: [ParsedVertex] = []
        var indices: [UInt32] = []
        var materials: [ParsedMaterial] = []

        // Materials
        materials.append(ParsedMaterial(name: "white", albedo: SIMD3<Float>(0.73, 0.73, 0.73), emissiveStrength: 0,
                                        emissiveColor: .zero, textureWidth: 0, textureHeight: 0, texturePixels: nil))
        materials.append(ParsedMaterial(name: "red", albedo: SIMD3<Float>(0.65, 0.05, 0.05), emissiveStrength: 0,
                                        emissiveColor: .zero, textureWidth: 0, textureHeight: 0, texturePixels: nil))
        materials.append(ParsedMaterial(name: "green", albedo: SIMD3<Float>(0.12, 0.45, 0.15), emissiveStrength: 0,
                                        emissiveColor: .zero, textureWidth: 0, textureHeight: 0, texturePixels: nil))
        materials.append(ParsedMaterial(name: "light", albedo: SIMD3<Float>(1.0, 0.9, 0.7), emissiveStrength: 15.0,
                                        emissiveColor: SIMD3<Float>(1.0, 0.9, 0.7), textureWidth: 0, textureHeight: 0, texturePixels: nil))

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
                normal: SIMD3(0, -1, 0), material: 3)

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
                normal: SIMD3(0, 0, 1), material: 0)
        addQuad(v0: SIMD3(b2x+b2w, 0, b2z-b2w), v1: SIMD3(b2x+b2w, b2h, b2z-b2w),
                v2: SIMD3(b2x, b2h, b2z-b2w), v3: SIMD3(b2x, 0, b2z-b2w),
                normal: SIMD3(0, 0, -1), material: 0)
        addQuad(v0: SIMD3(b2x, 0, b2z-b2w), v1: SIMD3(b2x, b2h, b2z-b2w),
                v2: SIMD3(b2x, b2h, b2z), v3: SIMD3(b2x, 0, b2z),
                normal: SIMD3(-1, 0, 0), material: 0)
        addQuad(v0: SIMD3(b2x+b2w, 0, b2z), v1: SIMD3(b2x+b2w, b2h, b2z),
                v2: SIMD3(b2x+b2w, b2h, b2z-b2w), v3: SIMD3(b2x+b2w, 0, b2z-b2w),
                normal: SIMD3(1, 0, 0), material: 0)
        addQuad(v0: SIMD3(b2x, b2h, b2z), v1: SIMD3(b2x, b2h, b2z-b2w),
                v2: SIMD3(b2x+b2w, b2h, b2z-b2w), v3: SIMD3(b2x+b2w, b2h, b2z),
                normal: SIMD3(0, 1, 0), material: 0)

        return BSPData(
            vertices: vertices,
            indices: indices,
            materials: materials,
            entityString: "",
            spawnPosition: SIMD3<Float>(0, 80, 180),
            spawnAngle: 0
        )
    }
}
