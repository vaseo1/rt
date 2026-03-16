import AppKit
import Metal

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var renderer: Renderer!
    var launchConfig = LaunchConfig()

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }

        let windowRect = NSRect(x: 0, y: 0, width: 1280, height: 720)
        window = NSWindow(
            contentRect: windowRect,
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = "RT Path Tracer"
        window.center()

        let gameView = GameView(frame: windowRect,
                    device: device,
                    launchConfig: launchConfig)
        window.contentView = gameView

        renderer = Renderer(device: device, view: gameView, launchConfig: launchConfig)
        gameView.renderer = renderer

        window.makeKeyAndOrderFront(nil)

        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)

        if !launchConfig.verifyConfig.enabled && !launchConfig.scriptedOIDNRepro {
            // Capture mouse for FPS-style control (skip in verify mode)
            gameView.captureMouse()
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}
