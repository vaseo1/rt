import Darwin
import Foundation

final class OIDNDenoiser {
    private typealias DeviceHandle = OpaquePointer
    private typealias FilterHandle = OpaquePointer

    private enum DeviceType: Int32 {
        case cpu = 1
    }

    private enum Format: Int32 {
        case float3 = 3
    }

    private enum Quality: Int32 {
        case high = 6
    }

    private enum ErrorCode: Int32 {
        case none = 0
    }

    private typealias NewDeviceFn = @convention(c) (Int32) -> DeviceHandle?
    private typealias ReleaseDeviceFn = @convention(c) (DeviceHandle?) -> Void
    private typealias CommitDeviceFn = @convention(c) (DeviceHandle?) -> Void
    private typealias GetDeviceErrorFn = @convention(c) (DeviceHandle?, UnsafeMutablePointer<UnsafePointer<CChar>?>?) -> Int32
    private typealias NewFilterFn = @convention(c) (DeviceHandle?, UnsafePointer<CChar>) -> FilterHandle?
    private typealias ReleaseFilterFn = @convention(c) (FilterHandle?) -> Void
    private typealias SetSharedFilterImageFn = @convention(c) (FilterHandle?, UnsafePointer<CChar>, UnsafeMutableRawPointer?, Int32, Int, Int, Int, Int, Int) -> Void
    private typealias SetFilterBoolFn = @convention(c) (FilterHandle?, UnsafePointer<CChar>, Bool) -> Void
    private typealias SetFilterIntFn = @convention(c) (FilterHandle?, UnsafePointer<CChar>, Int32) -> Void
    private typealias SetFilterFloatFn = @convention(c) (FilterHandle?, UnsafePointer<CChar>, Float) -> Void
    private typealias CommitFilterFn = @convention(c) (FilterHandle?) -> Void
    private typealias ExecuteFilterFn = @convention(c) (FilterHandle?) -> Void

    private struct API {
        let newDevice: NewDeviceFn
        let releaseDevice: ReleaseDeviceFn
        let commitDevice: CommitDeviceFn
        let getDeviceError: GetDeviceErrorFn
        let newFilter: NewFilterFn
        let releaseFilter: ReleaseFilterFn
        let setSharedFilterImage: SetSharedFilterImageFn
        let setFilterBool: SetFilterBoolFn
        let setFilterInt: SetFilterIntFn
        let setFilterFloat: SetFilterFloatFn
        let commitFilter: CommitFilterFn
        let executeFilter: ExecuteFilterFn
    }

    private let libraryHandle: UnsafeMutableRawPointer?
    private let api: API?
    private let device: DeviceHandle?

    let availabilityDescription: String
    var isAvailable: Bool {
        api != nil && device != nil
    }

    init() {
        let candidatePaths = [
            ProcessInfo.processInfo.environment["OIDN_LIBRARY_PATH"],
            "/opt/homebrew/opt/open-image-denoise/lib/libOpenImageDenoise.dylib",
            "/usr/local/opt/open-image-denoise/lib/libOpenImageDenoise.dylib"
        ].compactMap { $0 }

        var loadedHandle: UnsafeMutableRawPointer?
        var loadedPath: String?
        var lastError = "Open Image Denoise not found. Install it with `brew install open-image-denoise`."

        for path in candidatePaths {
            guard FileManager.default.fileExists(atPath: path) else {
                continue
            }

            if let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) {
                loadedHandle = handle
                loadedPath = path
                break
            }

            if let errorCString = dlerror() {
                lastError = String(cString: errorCString)
            }
        }

        guard let libraryHandle = loadedHandle else {
            self.libraryHandle = nil
            self.api = nil
            self.device = nil
            self.availabilityDescription = lastError
            return
        }

        func loadSymbol<T>(_ name: String, as type: T.Type) -> T? {
            guard let symbol = dlsym(libraryHandle, name) else {
                return nil
            }
            return unsafeBitCast(symbol, to: type)
        }

        guard let newDevice = loadSymbol("oidnNewDevice", as: NewDeviceFn.self),
              let releaseDevice = loadSymbol("oidnReleaseDevice", as: ReleaseDeviceFn.self),
              let commitDevice = loadSymbol("oidnCommitDevice", as: CommitDeviceFn.self),
              let getDeviceError = loadSymbol("oidnGetDeviceError", as: GetDeviceErrorFn.self),
              let newFilter = loadSymbol("oidnNewFilter", as: NewFilterFn.self),
              let releaseFilter = loadSymbol("oidnReleaseFilter", as: ReleaseFilterFn.self),
              let setSharedFilterImage = loadSymbol("oidnSetSharedFilterImage", as: SetSharedFilterImageFn.self),
              let setFilterBool = loadSymbol("oidnSetFilterBool", as: SetFilterBoolFn.self),
              let setFilterInt = loadSymbol("oidnSetFilterInt", as: SetFilterIntFn.self),
              let setFilterFloat = loadSymbol("oidnSetFilterFloat", as: SetFilterFloatFn.self),
              let commitFilter = loadSymbol("oidnCommitFilter", as: CommitFilterFn.self),
              let executeFilter = loadSymbol("oidnExecuteFilter", as: ExecuteFilterFn.self) else {
            dlclose(libraryHandle)
            self.libraryHandle = nil
            self.api = nil
            self.device = nil
            self.availabilityDescription = "Failed to load OIDN entry points from \(loadedPath ?? "library")."
            return
        }

        let api = API(
            newDevice: newDevice,
            releaseDevice: releaseDevice,
            commitDevice: commitDevice,
            getDeviceError: getDeviceError,
            newFilter: newFilter,
            releaseFilter: releaseFilter,
            setSharedFilterImage: setSharedFilterImage,
            setFilterBool: setFilterBool,
            setFilterInt: setFilterInt,
            setFilterFloat: setFilterFloat,
            commitFilter: commitFilter,
            executeFilter: executeFilter
        )

        guard let device = api.newDevice(DeviceType.cpu.rawValue) else {
            dlclose(libraryHandle)
            self.libraryHandle = nil
            self.api = nil
            self.device = nil
            self.availabilityDescription = "OIDN failed to create a CPU device."
            return
        }

        api.commitDevice(device)
        if let error = Self.readDeviceError(api: api, device: device) {
            api.releaseDevice(device)
            dlclose(libraryHandle)
            self.libraryHandle = nil
            self.api = nil
            self.device = nil
            self.availabilityDescription = error
            return
        }

        self.libraryHandle = libraryHandle
        self.api = api
        self.device = device
        self.availabilityDescription = "Loaded OIDN from \(loadedPath ?? "library")."
    }

    deinit {
        if let api, let device {
            api.releaseDevice(device)
        }
        if let libraryHandle {
            dlclose(libraryHandle)
        }
    }

    func denoiseHDR(colorRGBA: [Float],
                    albedoRGBA: [Float]?,
                    normalRGBA: [Float]?,
                    width: Int,
                    height: Int) -> [Float]? {
        guard let api, let device, width > 0, height > 0 else {
            return nil
        }

        let pixelCount = width * height
        guard colorRGBA.count == pixelCount * 4 else {
            return nil
        }

        var colorRGB = Self.makeSanitizedColorRGB(from: colorRGBA, pixelCount: pixelCount)
        var outputRGB = [Float](repeating: 0, count: pixelCount * 3)
        var albedoRGB = albedoRGBA.flatMap { Self.makeSanitizedAlbedoRGB(from: $0, pixelCount: pixelCount) }
        var normalRGB = normalRGBA.flatMap { Self.makeSanitizedNormalRGB(from: $0, pixelCount: pixelCount) }
        let inputScale = Self.computeInputScale(colorRGBPixels: colorRGB)

        guard let filter = "RT".withCString({ api.newFilter(device, $0) }) else {
            return nil
        }
        defer { api.releaseFilter(filter) }

        let rowStride = width * MemoryLayout<Float>.stride * 3

        func withOptionalBytes<Result>(_ pixels: inout [Float]?,
                                       _ body: (UnsafeMutableRawPointer?) -> Result) -> Result {
            guard var unwrappedPixels = pixels else {
                return body(nil)
            }

            let result = unwrappedPixels.withUnsafeMutableBytes { bytes -> Result in
                body(bytes.baseAddress)
            }
            pixels = unwrappedPixels
            return result
        }

        let executionSucceeded = colorRGB.withUnsafeMutableBytes { colorBytes in
            outputRGB.withUnsafeMutableBytes { outputBytes in
                withOptionalBytes(&albedoRGB) { albedoBaseAddress in
                    withOptionalBytes(&normalRGB) { normalBaseAddress in
                        guard let colorBaseAddress = colorBytes.baseAddress,
                              let outputBaseAddress = outputBytes.baseAddress else {
                            return false
                        }

                        "color".withCString { colorName in
                            api.setSharedFilterImage(filter,
                                                     colorName,
                                                     colorBaseAddress,
                                                     Format.float3.rawValue,
                                                     width,
                                                     height,
                                                     0,
                                                     0,
                                                     rowStride)
                        }
                        if let albedoBaseAddress {
                            "albedo".withCString { albedoName in
                                api.setSharedFilterImage(filter,
                                                         albedoName,
                                                         albedoBaseAddress,
                                                         Format.float3.rawValue,
                                                         width,
                                                         height,
                                                         0,
                                                         0,
                                                         rowStride)
                            }
                        }
                        if let normalBaseAddress {
                            "normal".withCString { normalName in
                                api.setSharedFilterImage(filter,
                                                         normalName,
                                                         normalBaseAddress,
                                                         Format.float3.rawValue,
                                                         width,
                                                         height,
                                                         0,
                                                         0,
                                                         rowStride)
                            }
                        }
                        "output".withCString { outputName in
                            api.setSharedFilterImage(filter,
                                                     outputName,
                                                     outputBaseAddress,
                                                     Format.float3.rawValue,
                                                     width,
                                                     height,
                                                     0,
                                                     0,
                                                     rowStride)
                        }
                        "hdr".withCString { hdrName in
                            api.setFilterBool(filter, hdrName, true)
                        }
                        "quality".withCString { qualityName in
                            api.setFilterInt(filter, qualityName, Quality.high.rawValue)
                        }
                        "inputScale".withCString { inputScaleName in
                            api.setFilterFloat(filter, inputScaleName, inputScale)
                        }
                        if albedoBaseAddress != nil || normalBaseAddress != nil {
                            "cleanAux".withCString { cleanAuxName in
                                api.setFilterBool(filter, cleanAuxName, true)
                            }
                        }

                        api.commitFilter(filter)
                        api.executeFilter(filter)
                        return true
                    }
                }
            }
        }

        guard executionSucceeded, Self.readDeviceError(api: api, device: device) == nil else {
            return nil
        }

        var outputRGBA = colorRGBA
        for pixelIndex in 0..<pixelCount {
            let rgbaIndex = pixelIndex * 4
            let rgbIndex = pixelIndex * 3
            outputRGBA[rgbaIndex] = Self.sanitizeHDRComponent(outputRGB[rgbIndex])
            outputRGBA[rgbaIndex + 1] = Self.sanitizeHDRComponent(outputRGB[rgbIndex + 1])
            outputRGBA[rgbaIndex + 2] = Self.sanitizeHDRComponent(outputRGB[rgbIndex + 2])
            outputRGBA[rgbaIndex + 3] = colorRGBA[rgbaIndex + 3]
        }

        return outputRGBA
    }

    private static func readDeviceError(api: API, device: DeviceHandle?) -> String? {
        var messagePointer: UnsafePointer<CChar>?
        let errorCode = ErrorCode(rawValue: api.getDeviceError(device, &messagePointer)) ?? .none
        guard errorCode != .none else {
            return nil
        }

        if let messagePointer {
            return String(cString: messagePointer)
        }
        return "OIDN reported error code \(errorCode.rawValue)."
    }

    private static func makeSanitizedColorRGB(from rgbaPixels: [Float], pixelCount: Int) -> [Float] {
        var rgbPixels = [Float](repeating: 0, count: pixelCount * 3)
        for pixelIndex in 0..<pixelCount {
            let rgbaIndex = pixelIndex * 4
            let rgbIndex = pixelIndex * 3
            rgbPixels[rgbIndex] = sanitizeHDRComponent(rgbaPixels[rgbaIndex])
            rgbPixels[rgbIndex + 1] = sanitizeHDRComponent(rgbaPixels[rgbaIndex + 1])
            rgbPixels[rgbIndex + 2] = sanitizeHDRComponent(rgbaPixels[rgbaIndex + 2])
        }
        return rgbPixels
    }

    private static func makeSanitizedAlbedoRGB(from rgbaPixels: [Float], pixelCount: Int) -> [Float]? {
        guard rgbaPixels.count == pixelCount * 4 else {
            return nil
        }

        var rgbPixels = [Float](repeating: 0, count: pixelCount * 3)
        for pixelIndex in 0..<pixelCount {
            let rgbaIndex = pixelIndex * 4
            let rgbIndex = pixelIndex * 3
            rgbPixels[rgbIndex] = sanitizeUnitIntervalComponent(rgbaPixels[rgbaIndex])
            rgbPixels[rgbIndex + 1] = sanitizeUnitIntervalComponent(rgbaPixels[rgbaIndex + 1])
            rgbPixels[rgbIndex + 2] = sanitizeUnitIntervalComponent(rgbaPixels[rgbaIndex + 2])
        }
        return rgbPixels
    }

    private static func makeSanitizedNormalRGB(from rgbaPixels: [Float], pixelCount: Int) -> [Float]? {
        guard rgbaPixels.count == pixelCount * 4 else {
            return nil
        }

        var rgbPixels = [Float](repeating: 0, count: pixelCount * 3)
        for pixelIndex in 0..<pixelCount {
            let rgbaIndex = pixelIndex * 4
            let rgbIndex = pixelIndex * 3
            let x = sanitizeSignedUnitComponent(rgbaPixels[rgbaIndex])
            let y = sanitizeSignedUnitComponent(rgbaPixels[rgbaIndex + 1])
            let z = sanitizeSignedUnitComponent(rgbaPixels[rgbaIndex + 2])
            let lengthSquared = x * x + y * y + z * z
            if lengthSquared > 1e-8 {
                let inverseLength = 1 / sqrt(lengthSquared)
                rgbPixels[rgbIndex] = x * inverseLength
                rgbPixels[rgbIndex + 1] = y * inverseLength
                rgbPixels[rgbIndex + 2] = z * inverseLength
            } else {
                rgbPixels[rgbIndex] = 0
                rgbPixels[rgbIndex + 1] = 1
                rgbPixels[rgbIndex + 2] = 0
            }
        }
        return rgbPixels
    }

    private static func computeInputScale(colorRGBPixels: [Float]) -> Float {
        var luminanceSum: Double = 0
        var validCount = 0

        for pixelIndex in stride(from: 0, to: colorRGBPixels.count, by: 3) {
            let r = colorRGBPixels[pixelIndex]
            let g = colorRGBPixels[pixelIndex + 1]
            let b = colorRGBPixels[pixelIndex + 2]
            let luminance = max(0.2126 * r + 0.7152 * g + 0.0722 * b, 0)
            if luminance.isFinite && luminance > 0 {
                luminanceSum += Double(luminance)
                validCount += 1
            }
        }

        let averageLuminance = validCount > 0 ? Float(luminanceSum / Double(validCount)) : 1
        return 1 / max(averageLuminance, 1e-3)
    }

    private static func sanitizeHDRComponent(_ value: Float) -> Float {
        guard value.isFinite else {
            return 0
        }
        return min(max(value, 0), 1e4)
    }

    private static func sanitizeUnitIntervalComponent(_ value: Float) -> Float {
        guard value.isFinite else {
            return 0
        }
        return min(max(value, 0), 1)
    }

    private static func sanitizeSignedUnitComponent(_ value: Float) -> Float {
        guard value.isFinite else {
            return 0
        }
        return min(max(value, -1), 1)
    }
}