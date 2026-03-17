import AppKit
import Metal
import QuartzCore

class GameView: NSView, CALayerDelegate {
    private struct InputSnapshot {
        var forward = false
        var back = false
        var left = false
        var right = false
        var up = false
        var sprint = false
        var mouseDeltaX: Float = 0
        var mouseDeltaY: Float = 0
        var cycleRenderMode = false
        var cycleDenoiseMethod = false
        var requestScreenshot = false
    }

    private struct ScriptedOIDNRepro {
        private let baselineCaptureFrame: UInt32 = 180
        private let movementStartFrame: UInt32 = 181
        private let movementEndFrame: UInt32 = 204
        private let rawPostMoveCaptureFrame: UInt32 = 210
        private let oidnSwitchFrame: UInt32 = 211
        private let oidnCaptureFrame: UInt32 = 450
        private let shutdownFrame: UInt32 = 486

        private(set) var frame: UInt32 = 0

        mutating func advance() -> (forward: Bool, captureScreenshot: Bool, switchToOIDN: Bool, terminate: Bool) {
            frame += 1

            let isMovingForward = frame >= movementStartFrame && frame < movementEndFrame
            let shouldCaptureScreenshot = frame == baselineCaptureFrame
                || frame == rawPostMoveCaptureFrame
                || frame == oidnCaptureFrame
            let shouldSwitchToOIDN = frame == oidnSwitchFrame
            let shouldTerminate = frame >= shutdownFrame
            return (isMovingForward, shouldCaptureScreenshot, shouldSwitchToOIDN, shouldTerminate)
        }

        var phaseDescription: String {
            switch frame {
            case ..<180:
                return "warmup"
            case 180:
                return "baseline-capture"
            case 181..<204:
                return "moving"
            case 204..<210:
                return "raw-settle"
            case 210:
                return "raw-post-move-capture"
            case 211:
                return "switch-to-oidn"
            case 212..<450:
                return "oidn-settle"
            case 450:
                return "oidn-post-switch-capture"
            default:
                return "shutdown"
            }
        }
    }

    let device: MTLDevice
    let metalLayer: CAMetalLayer
    var renderer: Renderer?
    private var displayLink: CVDisplayLink?
    private let renderQueue = DispatchQueue(label: "dev.rt.render", qos: .userInteractive)
    private let inputLock = NSLock()
    private var renderFrameQueued = false
    private var keysPressed: Set<UInt16> = []
    private var mouseCaptured = false
    private var mouseDeltaX: Float = 0
    private var mouseDeltaY: Float = 0
    private var shiftPressed = false
    private var pendingRenderModeCycle = false
    private var pendingDenoiseMethodCycle = false
    private var pendingScreenshotRequest = false

    // Match CAMetalLayer triple buffering so the GPU can stay busier on fast Apple Silicon.
    private let frameSemaphore = DispatchSemaphore(value: 3)
    private var debugFrameNumber: UInt32 = 0
    private var loggedSemaphoreSaturation = false

    // Key codes
    private let keyW: UInt16 = 13
    private let keyA: UInt16 = 0
    private let keyS: UInt16 = 1
    private let keyD: UInt16 = 2
    private let keyEscape: UInt16 = 53
    private let keySpace: UInt16 = 49
    private let keyLeftShift: UInt16 = 56
    private let keyRightShift: UInt16 = 60
    private let keyM: UInt16 = 46
    private let keyN: UInt16 = 45
    private let keyF12: UInt16 = 111
    private var scriptedOIDNRepro: ScriptedOIDNRepro?

    init(frame: NSRect, device: MTLDevice, launchConfig: LaunchConfig) {
        self.device = device
        self.metalLayer = CAMetalLayer()
        self.scriptedOIDNRepro = launchConfig.scriptedOIDNRepro ? ScriptedOIDNRepro() : nil

        super.init(frame: frame)

        wantsLayer = true

        metalLayer.device = device
        metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.framebufferOnly = false
        metalLayer.maximumDrawableCount = 3
        metalLayer.drawableSize = CGSize(width: frame.width * 2, height: frame.height * 2) // Retina
        metalLayer.displaySyncEnabled = true

        layer = metalLayer

        setupDisplayLink()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not implemented")
    }

    override var acceptsFirstResponder: Bool { true }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas {
            removeTrackingArea(area)
        }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.activeAlways, .mouseMoved, .mouseEnteredAndExited],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(area)
    }

    func captureMouse() {
        mouseCaptured = true
        CGAssociateMouseAndMouseCursorPosition(0)
        NSCursor.hide()
    }

    func releaseMouse() {
        mouseCaptured = false
        CGAssociateMouseAndMouseCursorPosition(1)
        NSCursor.unhide()
    }

    // MARK: - Layout

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        let scale = window?.backingScaleFactor ?? 2.0
        metalLayer.drawableSize = CGSize(
            width: newSize.width * scale,
            height: newSize.height * scale
        )
        let drawableWidth = Int(newSize.width * scale)
        let drawableHeight = Int(newSize.height * scale)
        renderQueue.async { [weak self] in
            self?.renderer?.resize(width: drawableWidth,
                                   height: drawableHeight)
        }
    }

    // MARK: - Display Link

    private func setupDisplayLink() {
        CVDisplayLinkCreateWithActiveCGDisplays(&displayLink)
        guard let displayLink = displayLink else { return }

        let callback: CVDisplayLinkOutputCallback = { _, _, _, _, _, userInfo -> CVReturn in
            let view = Unmanaged<GameView>.fromOpaque(userInfo!).takeUnretainedValue()
            view.scheduleRenderFrame()
            return kCVReturnSuccess
        }

        CVDisplayLinkSetOutputCallback(displayLink, callback,
                                       Unmanaged.passUnretained(self).toOpaque())
        CVDisplayLinkStart(displayLink)
    }

    private func scheduleRenderFrame() {
        inputLock.lock()
        guard !renderFrameQueued else {
            inputLock.unlock()
            return
        }
        renderFrameQueued = true
        inputLock.unlock()

        renderQueue.async { [weak self] in
            guard let self else { return }
            self.renderFrame()
            self.inputLock.lock()
            self.renderFrameQueued = false
            self.inputLock.unlock()
        }
    }

    private func consumeInputSnapshot() -> InputSnapshot {
        inputLock.lock()
        defer { inputLock.unlock() }

        let snapshot = InputSnapshot(
            forward: keysPressed.contains(keyW),
            back: keysPressed.contains(keyS),
            left: keysPressed.contains(keyA),
            right: keysPressed.contains(keyD),
            up: keysPressed.contains(keySpace),
            sprint: shiftPressed,
            mouseDeltaX: mouseDeltaX,
            mouseDeltaY: mouseDeltaY,
            cycleRenderMode: pendingRenderModeCycle,
            cycleDenoiseMethod: pendingDenoiseMethodCycle,
            requestScreenshot: pendingScreenshotRequest
        )

        mouseDeltaX = 0
        mouseDeltaY = 0
        pendingRenderModeCycle = false
        pendingDenoiseMethodCycle = false
        pendingScreenshotRequest = false

        return snapshot
    }

    private func renderFrame() {
        guard let renderer = renderer else { return }
        if renderer.captureInProgress { return }

        // Skip frame if GPU is saturated — keeps main thread free for input events
        guard frameSemaphore.wait(timeout: .now()) == .success else {
            if renderer.verifyConfig.enabled && !loggedSemaphoreSaturation {
                print("[GameView] Frame skipped: waiting for in-flight GPU work")
                loggedSemaphoreSaturation = true
            }
            return
        }
        loggedSemaphoreSaturation = false

        debugFrameNumber += 1
        if renderer.verifyConfig.enabled && debugFrameNumber <= 8 {
            print("[GameView] renderFrame #\(debugFrameNumber)")
        }

        let dt: Float = 1.0 / 60.0
        let input = consumeInputSnapshot()

        if input.cycleRenderMode {
            renderer.cycleRenderMode()
        }

        if input.cycleDenoiseMethod {
            renderer.cycleDenoiseMethod()
        }

        if input.requestScreenshot {
            renderer.pendingScreenshot = true
        }

        // Update camera from input
        var forward = input.forward
        let back = input.back
        let left = input.left
        let right = input.right
        let up = input.up
        let sprint = input.sprint

        if var scriptedOIDNRepro {
            let step = scriptedOIDNRepro.advance()
            self.scriptedOIDNRepro = scriptedOIDNRepro
            forward = step.forward

            if step.switchToOIDN {
                print("[OIDN_REPRO] switching denoiser frame=\(scriptedOIDNRepro.frame) phase=\(scriptedOIDNRepro.phaseDescription)")
                renderer.cycleDenoiseMethod()
            }

            if step.captureScreenshot {
                print("[OIDN_REPRO] capture requested frame=\(scriptedOIDNRepro.frame) phase=\(scriptedOIDNRepro.phaseDescription)")
                renderer.pendingScreenshot = true
            }

            if step.terminate {
                print("[OIDN_REPRO] terminating frame=\(scriptedOIDNRepro.frame) phase=\(scriptedOIDNRepro.phaseDescription)")
                DispatchQueue.main.async {
                    NSApp.terminate(nil)
                }
            }
        }

        renderer.camera.update(
            forward: forward, back: back,
            left: left, right: right,
            up: up, sprint: sprint,
            mouseDeltaX: input.mouseDeltaX,
            mouseDeltaY: input.mouseDeltaY,
            dt: dt
        )

        // Render
        guard let drawable = metalLayer.nextDrawable() else {
            if renderer.verifyConfig.enabled {
                print("[GameView] nextDrawable() returned nil")
            }
            frameSemaphore.signal()
            return
        }
        renderer.render(to: drawable) { [weak self] in
            self?.frameSemaphore.signal()
        }

        // Update title with compact scene, camera, and render state.
        let windowTitle = renderer.windowTitle
        DispatchQueue.main.async { [weak self] in
            self?.window?.title = windowTitle
        }
    }

    deinit {
        if let displayLink = displayLink {
            CVDisplayLinkStop(displayLink)
        }
    }

    // MARK: - Keyboard Input

    override func keyDown(with event: NSEvent) {
        if event.keyCode == keyEscape {
            if mouseCaptured {
                releaseMouse()
            } else {
                NSApp.terminate(nil)
            }
        } else if event.keyCode == keyM {
            if !event.isARepeat {
                inputLock.lock()
                pendingRenderModeCycle = true
                inputLock.unlock()
            }
            return
        } else if event.keyCode == keyN {
            if !event.isARepeat {
                inputLock.lock()
                pendingDenoiseMethodCycle = true
                inputLock.unlock()
            }
            return
        } else if event.keyCode == keyF12 {
            inputLock.lock()
            pendingScreenshotRequest = true
            inputLock.unlock()
        }

        inputLock.lock()
        keysPressed.insert(event.keyCode)
        inputLock.unlock()
    }

    override func keyUp(with event: NSEvent) {
        inputLock.lock()
        keysPressed.remove(event.keyCode)
        inputLock.unlock()
    }

    override func flagsChanged(with event: NSEvent) {
        if event.keyCode == keyLeftShift || event.keyCode == keyRightShift {
            inputLock.lock()
            shiftPressed = event.modifierFlags.contains(.shift)
            inputLock.unlock()
        }
    }

    // MARK: - Mouse Input

    override func mouseMoved(with event: NSEvent) {
        guard mouseCaptured else { return }
        inputLock.lock()
        mouseDeltaX += Float(event.deltaX)
        mouseDeltaY += Float(event.deltaY)
        inputLock.unlock()
    }

    override func mouseDragged(with event: NSEvent) {
        mouseMoved(with: event)
    }

    override func mouseDown(with event: NSEvent) {
        if !mouseCaptured {
            captureMouse()
        }
    }
}
