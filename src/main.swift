import AppKit
import Foundation
import simd

enum RenderMode: String, CaseIterable {
    case auto
    case raw
    case accumulation
    case svgf
    case metalfx
    case metalfxSVGF = "metalfx-svgf"

    var displayName: String {
        switch self {
        case .auto:
            return "Auto"
        case .raw:
            return "Raw"
        case .accumulation:
            return "Accumulation"
        case .svgf:
            return "SVGF"
        case .metalfx:
            return "MetalFX"
        case .metalfxSVGF:
            return "MetalFX+SVGF"
        }
    }

    static var argumentList: String {
        allCases.map(\.rawValue).joined(separator: ", ")
    }
}

// Parse startup CLI flags
struct VerifyConfig {
    var enabled = false
    var targetFrames: UInt32 = 64
    var outputPath = "verify_screenshot.png"
}

struct LaunchConfig {
    var verifyConfig = VerifyConfig()
    var startPosition: SIMD3<Float>?
    var lookAtPosition: SIMD3<Float>?
    var lookAtWater = false
    var highlightWater = false
    var renderMode: RenderMode = .accumulation
}

private func failArgumentParsing(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(2)
}

private func parseVector3(_ rawValue: String) -> SIMD3<Float>? {
    let parts = rawValue
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespaces) }
    guard parts.count == 3,
          let x = Float(parts[0]),
          let y = Float(parts[1]),
          let z = Float(parts[2]) else {
        return nil
    }

    return SIMD3<Float>(x, y, z)
}

func parseLaunchArgs() -> LaunchConfig {
    var config = LaunchConfig()
    let args = Array(CommandLine.arguments.dropFirst())
    var index = 0

    while index < args.count {
        let arg = args[index]

        switch arg {
        case "--verify":
            config.verifyConfig.enabled = true
        case "--verify-frames":
            guard index + 1 < args.count, let frameCount = UInt32(args[index + 1]) else {
                failArgumentParsing("--verify-frames requires a positive integer value")
            }
            config.verifyConfig.targetFrames = frameCount
            index += 1
        case "--verify-output":
            guard index + 1 < args.count else {
                failArgumentParsing("--verify-output requires a path")
            }
            config.verifyConfig.outputPath = args[index + 1]
            index += 1
        case "--start-pos", "--start-position":
            guard index + 3 < args.count,
                  let x = Float(args[index + 1]),
                  let y = Float(args[index + 2]),
                  let z = Float(args[index + 3]) else {
                failArgumentParsing("\(arg) requires three Float values: x y z")
            }
            config.startPosition = SIMD3<Float>(x, y, z)
            index += 3
        case "--look-at":
            guard index + 3 < args.count,
                  let x = Float(args[index + 1]),
                  let y = Float(args[index + 2]),
                  let z = Float(args[index + 3]) else {
                failArgumentParsing("--look-at requires three Float values: x y z")
            }
            config.lookAtPosition = SIMD3<Float>(x, y, z)
            index += 3
        case "--look-at-water":
            config.lookAtWater = true
        case "--highlight-water":
            config.highlightWater = true
        case "--render-mode":
            guard index + 1 < args.count,
                  let renderMode = RenderMode(rawValue: args[index + 1].lowercased()) else {
                failArgumentParsing("--render-mode requires one of: \(RenderMode.argumentList)")
            }
            config.renderMode = renderMode
            index += 1
        default:
            if arg.hasPrefix("--start-pos=") || arg.hasPrefix("--start-position=") {
                let components = arg.split(separator: "=", maxSplits: 1)
                guard components.count == 2,
                      let position = parseVector3(String(components[1])) else {
                    failArgumentParsing("\(arg) must be in the form --start-pos=x,y,z")
                }
                config.startPosition = position
            } else if arg.hasPrefix("--look-at=") {
                let components = arg.split(separator: "=", maxSplits: 1)
                guard components.count == 2,
                      let lookAtPosition = parseVector3(String(components[1])) else {
                    failArgumentParsing("\(arg) must be in the form --look-at=x,y,z")
                }
                config.lookAtPosition = lookAtPosition
            } else if arg.hasPrefix("--render-mode=") {
                let components = arg.split(separator: "=", maxSplits: 1)
                guard components.count == 2,
                      let renderMode = RenderMode(rawValue: String(components[1]).lowercased()) else {
                    failArgumentParsing("\(arg) must be one of: \(RenderMode.argumentList)")
                }
                config.renderMode = renderMode
            }
        }

        index += 1
    }

    return config
}

let launchConfig = parseLaunchArgs()

let app = NSApplication.shared
let delegate = AppDelegate()
delegate.launchConfig = launchConfig
app.delegate = delegate
app.run()
