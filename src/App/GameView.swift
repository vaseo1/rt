import AppKit
import Metal
import QuartzCore

class GameView: NSView, CALayerDelegate {
    let device: MTLDevice
    let metalLayer: CAMetalLayer
    var renderer: Renderer?
    private var displayLink: CVDisplayLink?
    private var keysPressed: Set<UInt16> = []
    private var mouseCaptured = false
    private var mouseDeltaX: Float = 0
    private var mouseDeltaY: Float = 0
    private var shiftPressed = false

    // Limit in-flight frames so nextDrawable() never blocks the main thread
    private let frameSemaphore = DispatchSemaphore(value: 2)
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

    init(frame: NSRect, device: MTLDevice) {
        self.device = device
        self.metalLayer = CAMetalLayer()

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
        renderer?.resize(width: Int(newSize.width * scale),
                         height: Int(newSize.height * scale))
    }

    // MARK: - Display Link

    private func setupDisplayLink() {
        CVDisplayLinkCreateWithActiveCGDisplays(&displayLink)
        guard let displayLink = displayLink else { return }

        let callback: CVDisplayLinkOutputCallback = { _, _, _, _, _, userInfo -> CVReturn in
            let view = Unmanaged<GameView>.fromOpaque(userInfo!).takeUnretainedValue()
            DispatchQueue.main.async {
                view.renderFrame()
            }
            return kCVReturnSuccess
        }

        CVDisplayLinkSetOutputCallback(displayLink, callback,
                                       Unmanaged.passUnretained(self).toOpaque())
        CVDisplayLinkStart(displayLink)
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

        // Update camera from input
        let forward = keysPressed.contains(keyW)
        let back    = keysPressed.contains(keyS)
        let left    = keysPressed.contains(keyA)
        let right   = keysPressed.contains(keyD)
        let up      = keysPressed.contains(keySpace)
        let sprint  = shiftPressed

        renderer.camera.update(
            forward: forward, back: back,
            left: left, right: right,
            up: up, sprint: sprint,
            mouseDeltaX: mouseDeltaX,
            mouseDeltaY: mouseDeltaY,
            dt: dt
        )

        mouseDeltaX = 0
        mouseDeltaY = 0

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
        window?.title = renderer.windowTitle
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
                renderer?.cycleRenderMode()
            }
            return
        } else if event.keyCode == keyN {
            if !event.isARepeat {
                renderer?.cycleDenoiseMethod()
            }
            return
        } else if event.keyCode == keyF12 {
            renderer?.pendingScreenshot = true
        }

        keysPressed.insert(event.keyCode)
    }

    override func keyUp(with event: NSEvent) {
        keysPressed.remove(event.keyCode)
    }

    override func flagsChanged(with event: NSEvent) {
        if event.keyCode == keyLeftShift || event.keyCode == keyRightShift {
            shiftPressed = event.modifierFlags.contains(.shift)
        }
    }

    // MARK: - Mouse Input

    override func mouseMoved(with event: NSEvent) {
        guard mouseCaptured else { return }
        mouseDeltaX += Float(event.deltaX)
        mouseDeltaY += Float(event.deltaY)
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
