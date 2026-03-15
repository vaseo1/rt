import AppKit
import Foundation

// Parse --verify CLI flags
struct VerifyConfig {
    var enabled = false
    var targetFrames: UInt32 = 64
    var outputPath = "verify_screenshot.png"
}

func parseVerifyArgs() -> VerifyConfig {
    var config = VerifyConfig()
    let args = CommandLine.arguments
    for i in 0..<args.count {
        switch args[i] {
        case "--verify":
            config.enabled = true
        case "--verify-frames":
            if i + 1 < args.count, let n = UInt32(args[i + 1]) {
                config.targetFrames = n
            }
        case "--verify-output":
            if i + 1 < args.count {
                config.outputPath = args[i + 1]
            }
        default:
            break
        }
    }
    return config
}

let verifyConfig = parseVerifyArgs()

let app = NSApplication.shared
let delegate = AppDelegate()
delegate.verifyConfig = verifyConfig
app.delegate = delegate
app.run()
