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
    var outputDirectory: String?
    var checkpointFrames: [UInt32] = []
    var sweepDenoiseMethods = false
    var referenceFrames: UInt32?
    var referenceDenoiseMethod: DenoiseMethod = .none
}

enum DenoiseMethod: String, CaseIterable {
    case none
    case oidn
    case svgfPlus = "svgfplus"
    case eaw
    case bilateral
    case nlm

    var displayName: String {
        switch self {
        case .none: return "None"
        case .oidn: return "OIDN"
        case .svgfPlus: return "SVGF+"
        case .eaw: return "EAW"
        case .bilateral: return "Bilateral"
        case .nlm: return "NLM"
        }
    }

    static var argumentList: String {
        allCases.map(\.rawValue).joined(separator: ", ")
    }
}

struct LaunchConfig {
    var verifyConfig = VerifyConfig()
    var startPosition: SIMD3<Float>?
    var lookAtPosition: SIMD3<Float>?
    var lookAtWater = false
    var highlightWater = false
    var renderMode: RenderMode = .accumulation
    var denoiseMethod: DenoiseMethod = .none
    var scriptedOIDNRepro = false
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

private func parseFrameList(_ rawValue: String) -> [UInt32]? {
    let parts = rawValue
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespaces) }

    guard !parts.isEmpty else {
        return nil
    }

    var frames: [UInt32] = []
    frames.reserveCapacity(parts.count)

    for part in parts {
        guard let frame = UInt32(part), frame > 0 else {
            return nil
        }
        frames.append(frame)
    }

    return Array(Set(frames)).sorted()
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
        case "--verify-output-dir":
            guard index + 1 < args.count else {
                failArgumentParsing("--verify-output-dir requires a directory path")
            }
            config.verifyConfig.outputDirectory = args[index + 1]
            index += 1
        case "--verify-checkpoints":
            guard index + 1 < args.count,
                  let checkpoints = parseFrameList(args[index + 1]) else {
                failArgumentParsing("--verify-checkpoints requires a comma-separated frame list, e.g. 1,2,4,8,16")
            }
            config.verifyConfig.checkpointFrames = checkpoints
            if let lastCheckpoint = checkpoints.last {
                config.verifyConfig.targetFrames = max(config.verifyConfig.targetFrames, lastCheckpoint)
            }
            index += 1
        case "--verify-sweep-denoise":
            config.verifyConfig.sweepDenoiseMethods = true
        case "--verify-reference-frames":
            guard index + 1 < args.count, let frameCount = UInt32(args[index + 1]), frameCount > 0 else {
                failArgumentParsing("--verify-reference-frames requires a positive integer value")
            }
            config.verifyConfig.referenceFrames = frameCount
            index += 1
        case "--verify-reference-denoise-method":
            guard index + 1 < args.count,
                  let method = DenoiseMethod(rawValue: args[index + 1].lowercased()) else {
                failArgumentParsing("--verify-reference-denoise-method requires one of: \(DenoiseMethod.argumentList)")
            }
            config.verifyConfig.referenceDenoiseMethod = method
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
        case "--scripted-oidn-repro":
            config.scriptedOIDNRepro = true
        case "--render-mode":
            guard index + 1 < args.count,
                  let renderMode = RenderMode(rawValue: args[index + 1].lowercased()) else {
                failArgumentParsing("--render-mode requires one of: \(RenderMode.argumentList)")
            }
            config.renderMode = renderMode
            index += 1
        case "--denoise-method":
            guard index + 1 < args.count,
                  let method = DenoiseMethod(rawValue: args[index + 1].lowercased()) else {
                failArgumentParsing("--denoise-method requires one of: \(DenoiseMethod.argumentList)")
            }
            config.denoiseMethod = method
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
            } else if arg.hasPrefix("--denoise-method=") {
                let components = arg.split(separator: "=", maxSplits: 1)
                guard components.count == 2,
                      let method = DenoiseMethod(rawValue: String(components[1]).lowercased()) else {
                    failArgumentParsing("\(arg) must be one of: \(DenoiseMethod.argumentList)")
                }
                config.denoiseMethod = method
            } else if arg.hasPrefix("--verify-output-dir=") {
                let components = arg.split(separator: "=", maxSplits: 1)
                guard components.count == 2 else {
                    failArgumentParsing("\(arg) must be in the form --verify-output-dir=path")
                }
                config.verifyConfig.outputDirectory = String(components[1])
            } else if arg.hasPrefix("--verify-checkpoints=") {
                let components = arg.split(separator: "=", maxSplits: 1)
                guard components.count == 2,
                      let checkpoints = parseFrameList(String(components[1])) else {
                    failArgumentParsing("\(arg) must be in the form --verify-checkpoints=1,2,4,8")
                }
                config.verifyConfig.checkpointFrames = checkpoints
                if let lastCheckpoint = checkpoints.last {
                    config.verifyConfig.targetFrames = max(config.verifyConfig.targetFrames, lastCheckpoint)
                }
            } else if arg.hasPrefix("--verify-reference-frames=") {
                let components = arg.split(separator: "=", maxSplits: 1)
                guard components.count == 2,
                      let frameCount = UInt32(components[1]), frameCount > 0 else {
                    failArgumentParsing("\(arg) must be in the form --verify-reference-frames=256")
                }
                config.verifyConfig.referenceFrames = frameCount
            } else if arg.hasPrefix("--verify-reference-denoise-method=") {
                let components = arg.split(separator: "=", maxSplits: 1)
                guard components.count == 2,
                      let method = DenoiseMethod(rawValue: String(components[1]).lowercased()) else {
                    failArgumentParsing("\(arg) must be one of: \(DenoiseMethod.argumentList)")
                }
                config.verifyConfig.referenceDenoiseMethod = method
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
