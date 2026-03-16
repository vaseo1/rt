import Metal
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

struct FrameMetrics {
    let avgBrightness: Float
    let nonBlackPercent: Float
    let highFrequencyLumaRMS: Float
    let minValue: Float
    let maxValue: Float
    let nanCount: Int
    let width: Int
    let height: Int

    var passed: Bool {
        nonBlackPercent > 5.0 && nanCount == 0 && avgBrightness > 0.01
    }

    func printReport(screenshotPath: String? = nil) {
        if let path = screenshotPath {
            print("[VERIFY] screenshot: \(path)")
        }
        print(String(format: "[VERIFY] avg_brightness: %.4f", avgBrightness))
        print(String(format: "[VERIFY] non_black_pct: %.1f", nonBlackPercent))
        print(String(format: "[VERIFY] hf_luma_rms: %.5f", highFrequencyLumaRMS))
        print(String(format: "[VERIFY] min_value: %.4f", minValue))
        print(String(format: "[VERIFY] max_value: %.4f", maxValue))
        print("[VERIFY] nan_count: \(nanCount)")
        print("[VERIFY] resolution: \(width)x\(height)")
        print("[VERIFY] status: \(passed ? "PASS" : "FAIL")")
    }
}

struct ReferenceFrameMetrics {
    let meanAbsoluteError: Float
    let rmse: Float
    let psnr: Float
    let lumaRMSE: Float
    let maxAbsoluteError: Float

    func printReport(referenceLabel: String) {
        print("[VERIFY] reference: \(referenceLabel)")
        print(String(format: "[VERIFY] ref_mae: %.5f", meanAbsoluteError))
        print(String(format: "[VERIFY] ref_rmse: %.5f", rmse))
        if psnr.isFinite {
            print(String(format: "[VERIFY] ref_psnr: %.2f", psnr))
        } else {
            print("[VERIFY] ref_psnr: inf")
        }
        print(String(format: "[VERIFY] ref_luma_rmse: %.5f", lumaRMSE))
        print(String(format: "[VERIFY] ref_max_abs_err: %.5f", maxAbsoluteError))
    }
}

struct HDRReferenceFrameMetrics {
    let meanAbsoluteError: Float
    let rmse: Float
    let peakPSNR: Float
    let lumaRMSE: Float
    let relativeLumaRMSE: Float
    let maxAbsoluteError: Float

    func printReport(referenceLabel: String) {
        print("[VERIFY] hdr_reference: \(referenceLabel)")
        print(String(format: "[VERIFY] hdr_ref_mae: %.5f", meanAbsoluteError))
        print(String(format: "[VERIFY] hdr_ref_rmse: %.5f", rmse))
        if peakPSNR.isFinite {
            print(String(format: "[VERIFY] hdr_ref_peak_psnr: %.2f", peakPSNR))
        } else {
            print("[VERIFY] hdr_ref_peak_psnr: inf")
        }
        print(String(format: "[VERIFY] hdr_ref_luma_rmse: %.5f", lumaRMSE))
        print(String(format: "[VERIFY] hdr_ref_rel_luma_rmse: %.5f", relativeLumaRMSE))
        print(String(format: "[VERIFY] hdr_ref_max_abs_err: %.5f", maxAbsoluteError))
    }
}

enum ScreenshotCapture {

    /// Copy a GPU-private texture to a CPU-readable staging texture, then read pixels.
    private static func readPixels(from texture: MTLTexture, commandQueue: MTLCommandQueue) -> [UInt8]? {
        let w = texture.width
        let h = texture.height
        let bytesPerRow = w * 4

        if texture.storageMode != .private {
            var pixels = [UInt8](repeating: 0, count: bytesPerRow * h)
            texture.getBytes(&pixels,
                             bytesPerRow: bytesPerRow,
                             from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                             size: MTLSize(width: w, height: h, depth: 1)),
                             mipmapLevel: 0)
            return pixels
        }

        // Create a shared-mode staging texture for CPU readback
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: texture.pixelFormat, width: w, height: h, mipmapped: false)
        desc.storageMode = .shared
        desc.usage = []
        guard let staging = texture.device.makeTexture(descriptor: desc) else { return nil }

        // Blit from private texture to shared staging texture
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let blit = cmdBuf.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: texture,
                  sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: staging,
                  destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var pixels = [UInt8](repeating: 0, count: bytesPerRow * h)
        staging.getBytes(&pixels,
                         bytesPerRow: bytesPerRow,
                         from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                         size: MTLSize(width: w, height: h, depth: 1)),
                         mipmapLevel: 0)
        return pixels
    }

    static func readRGBA8Pixels(texture: MTLTexture, commandQueue: MTLCommandQueue) -> [UInt8]? {
        guard var pixels = readPixels(from: texture, commandQueue: commandQueue) else {
            return nil
        }

        for i in stride(from: 0, to: pixels.count, by: 4) {
            let b = pixels[i]
            pixels[i] = pixels[i + 2]
            pixels[i + 2] = b
        }

        return pixels
    }

    static func readRGBA32FloatPixels(texture: MTLTexture, commandQueue: MTLCommandQueue) -> [Float]? {
        guard texture.pixelFormat == .rgba32Float else {
            return nil
        }

        let w = texture.width
        let h = texture.height
        let bytesPerRow = w * MemoryLayout<Float>.stride * 4
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: w, height: h, depth: 1))

        if texture.storageMode != .private {
            var pixels = [Float](repeating: 0, count: w * h * 4)
            texture.getBytes(&pixels,
                             bytesPerRow: bytesPerRow,
                             from: region,
                             mipmapLevel: 0)
            return pixels
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: texture.pixelFormat, width: w, height: h, mipmapped: false)
        desc.storageMode = .shared
        desc.usage = []
        guard let staging = texture.device.makeTexture(descriptor: desc),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let blit = cmdBuf.makeBlitCommandEncoder() else {
            return nil
        }

        blit.copy(from: texture,
                  sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: staging,
                  destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        var pixels = [Float](repeating: 0, count: w * h * 4)
        staging.getBytes(&pixels,
                         bytesPerRow: bytesPerRow,
                         from: region,
                         mipmapLevel: 0)
        return pixels
    }

    static func readRGBA16FloatPixels(texture: MTLTexture, commandQueue: MTLCommandQueue) -> [Float]? {
        guard texture.pixelFormat == .rgba16Float else {
            return nil
        }

        let w = texture.width
        let h = texture.height
        let componentCount = w * h * 4
        let bytesPerRow = w * MemoryLayout<UInt16>.stride * 4
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: w, height: h, depth: 1))

        func decodeHalfPixels(from texture: MTLTexture) -> [Float] {
            var halfPixels = [UInt16](repeating: 0, count: componentCount)
            texture.getBytes(&halfPixels,
                             bytesPerRow: bytesPerRow,
                             from: region,
                             mipmapLevel: 0)
            return halfPixels.map { Float(Float16(bitPattern: $0)) }
        }

        if texture.storageMode != .private {
            return decodeHalfPixels(from: texture)
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: texture.pixelFormat, width: w, height: h, mipmapped: false)
        desc.storageMode = .shared
        desc.usage = []
        guard let staging = texture.device.makeTexture(descriptor: desc),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let blit = cmdBuf.makeBlitCommandEncoder() else {
            return nil
        }

        blit.copy(from: texture,
                  sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: staging,
                  destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        return decodeHalfPixels(from: staging)
    }

    /// Save a bgra8Unorm texture to a PNG file.
    static func capture(texture: MTLTexture, commandQueue: MTLCommandQueue, to url: URL) -> Bool {
        let w = texture.width
        let h = texture.height
        guard var pixels = readRGBA8Pixels(texture: texture, commandQueue: commandQueue) else {
            print("[VERIFY] ERROR: Failed to read texture pixels")
            return false
        }

        let bitsPerComponent = 8
        let bytesPerRow = w * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let context = CGContext(data: &pixels,
                                      width: w,
                                      height: h,
                                      bitsPerComponent: bitsPerComponent,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo.rawValue),
              let image = context.makeImage() else {
            print("[VERIFY] ERROR: Failed to create CGImage")
            return false
        }

        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            print("[VERIFY] ERROR: Failed to create image destination at \(url.path)")
            return false
        }

        CGImageDestinationAddImage(dest, image, nil)
        let ok = CGImageDestinationFinalize(dest)
        if !ok {
            print("[VERIFY] ERROR: Failed to write PNG to \(url.path)")
        }
        return ok
    }

    /// Compute rendering metrics from a bgra8Unorm texture.
    static func computeMetrics(texture: MTLTexture, commandQueue: MTLCommandQueue) -> FrameMetrics {
        let w = texture.width
        let h = texture.height
        guard let pixels = readRGBA8Pixels(texture: texture, commandQueue: commandQueue) else {
            return FrameMetrics(avgBrightness: 0, nonBlackPercent: 0,
                                highFrequencyLumaRMS: 0,
                                minValue: 0, maxValue: 0, nanCount: 0,
                                width: w, height: h)
        }

        let totalPixels = w * h
        var brightnessSum: Double = 0
        var nonBlackCount = 0
        var nanCount = 0
        var minVal: Float = 1.0
        var maxVal: Float = 0.0
        var luminanceValues = [Float](repeating: 0, count: totalPixels)

        for i in stride(from: 0, to: pixels.count, by: 4) {
            let pixelIndex = i / 4
            let r = Float(pixels[i])     / 255.0
            let g = Float(pixels[i + 1]) / 255.0
            let b = Float(pixels[i + 2]) / 255.0

            if r.isNaN || g.isNaN || b.isNaN {
                nanCount += 1
                luminanceValues[pixelIndex] = 0
                continue
            }

            // Rec. 709 luminance
            let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            luminanceValues[pixelIndex] = lum
            brightnessSum += Double(lum)

            minVal = min(minVal, min(r, min(g, b)))
            maxVal = max(maxVal, max(r, max(g, b)))

            if lum > 0.01 {
                nonBlackCount += 1
            }
        }

        let avgBrightness = totalPixels > 0 ? Float(brightnessSum / Double(totalPixels)) : 0
        let nonBlackPct = totalPixels > 0 ? Float(nonBlackCount) / Float(totalPixels) * 100.0 : 0
        var highFrequencyEnergy: Double = 0

        if w > 2 && h > 2 {
            for y in 1..<(h - 1) {
                for x in 1..<(w - 1) {
                    let centerIndex = y * w + x
                    let neighborhoodMean = (
                        luminanceValues[(y - 1) * w + (x - 1)] + luminanceValues[(y - 1) * w + x] + luminanceValues[(y - 1) * w + (x + 1)] +
                        luminanceValues[y * w + (x - 1)] + luminanceValues[y * w + x] + luminanceValues[y * w + (x + 1)] +
                        luminanceValues[(y + 1) * w + (x - 1)] + luminanceValues[(y + 1) * w + x] + luminanceValues[(y + 1) * w + (x + 1)]
                    ) / 9.0
                    let delta = Double(luminanceValues[centerIndex] - neighborhoodMean)
                    highFrequencyEnergy += delta * delta
                }
            }
        }

        let highFrequencySampleCount = max((w - 2) * (h - 2), 1)
        let highFrequencyLumaRMS = Float(sqrt(highFrequencyEnergy / Double(highFrequencySampleCount)))

        return FrameMetrics(
            avgBrightness: avgBrightness,
            nonBlackPercent: nonBlackPct,
            highFrequencyLumaRMS: highFrequencyLumaRMS,
            minValue: minVal,
            maxValue: maxVal,
            nanCount: nanCount,
            width: w,
            height: h
        )
    }

    static func computeReferenceMetrics(texture: MTLTexture,
                                        commandQueue: MTLCommandQueue,
                                        referencePixels: [UInt8]) -> ReferenceFrameMetrics? {
        guard let pixels = readRGBA8Pixels(texture: texture, commandQueue: commandQueue),
              pixels.count == referencePixels.count,
              !pixels.isEmpty else {
            return nil
        }

        var absoluteErrorSum: Double = 0
        var squaredErrorSum: Double = 0
        var lumaSquaredErrorSum: Double = 0
        var maxAbsoluteError: Float = 0
        let sampleCount = pixels.count / 4

        for i in stride(from: 0, to: pixels.count, by: 4) {
            let r = Float(pixels[i]) / 255.0
            let g = Float(pixels[i + 1]) / 255.0
            let b = Float(pixels[i + 2]) / 255.0
            let rr = Float(referencePixels[i]) / 255.0
            let rg = Float(referencePixels[i + 1]) / 255.0
            let rb = Float(referencePixels[i + 2]) / 255.0

            let dr = r - rr
            let dg = g - rg
            let db = b - rb

            let perPixelMAE = (abs(dr) + abs(dg) + abs(db)) / 3.0
            let perPixelMSE = (dr * dr + dg * dg + db * db) / 3.0
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            let refLuma = 0.2126 * rr + 0.7152 * rg + 0.0722 * rb
            let lumaDelta = luma - refLuma

            absoluteErrorSum += Double(perPixelMAE)
            squaredErrorSum += Double(perPixelMSE)
            lumaSquaredErrorSum += Double(lumaDelta * lumaDelta)
            maxAbsoluteError = max(maxAbsoluteError, max(abs(dr), max(abs(dg), abs(db))))
        }

        let safeSampleCount = max(sampleCount, 1)
        let meanAbsoluteError = Float(absoluteErrorSum / Double(safeSampleCount))
        let mse = Float(squaredErrorSum / Double(safeSampleCount))
        let rmse = sqrt(mse)
        let lumaRMSE = Float(sqrt(lumaSquaredErrorSum / Double(safeSampleCount)))
        let psnr: Float = mse > 1e-12 ? (10.0 * log10(1.0 / mse)) : .infinity

        return ReferenceFrameMetrics(meanAbsoluteError: meanAbsoluteError,
                                     rmse: rmse,
                                     psnr: psnr,
                                     lumaRMSE: lumaRMSE,
                                     maxAbsoluteError: maxAbsoluteError)
    }

    static func computeHDRReferenceMetrics(texture: MTLTexture,
                                           commandQueue: MTLCommandQueue,
                                           referencePixels: [Float]) -> HDRReferenceFrameMetrics? {
        guard let pixels = readRGBA32FloatPixels(texture: texture, commandQueue: commandQueue),
              pixels.count == referencePixels.count,
              !pixels.isEmpty else {
            return nil
        }

        var absoluteErrorSum: Double = 0
        var squaredErrorSum: Double = 0
        var lumaSquaredErrorSum: Double = 0
        var relativeLumaSquaredErrorSum: Double = 0
        var maxAbsoluteError: Float = 0
        var peakReferenceValue: Float = 0
        let sampleCount = pixels.count / 4

        for i in stride(from: 0, to: pixels.count, by: 4) {
            let r = pixels[i]
            let g = pixels[i + 1]
            let b = pixels[i + 2]
            let rr = referencePixels[i]
            let rg = referencePixels[i + 1]
            let rb = referencePixels[i + 2]

            let dr = r - rr
            let dg = g - rg
            let db = b - rb

            let perPixelMAE = (abs(dr) + abs(dg) + abs(db)) / 3.0
            let perPixelMSE = (dr * dr + dg * dg + db * db) / 3.0
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            let refLuma = 0.2126 * rr + 0.7152 * rg + 0.0722 * rb
            let lumaDelta = luma - refLuma
            let relativeLumaDelta = lumaDelta / max(refLuma, 1e-3)

            absoluteErrorSum += Double(perPixelMAE)
            squaredErrorSum += Double(perPixelMSE)
            lumaSquaredErrorSum += Double(lumaDelta * lumaDelta)
            relativeLumaSquaredErrorSum += Double(relativeLumaDelta * relativeLumaDelta)
            maxAbsoluteError = max(maxAbsoluteError, max(abs(dr), max(abs(dg), abs(db))))
            peakReferenceValue = max(peakReferenceValue, max(rr, max(rg, rb)))
        }

        let safeSampleCount = max(sampleCount, 1)
        let meanAbsoluteError = Float(absoluteErrorSum / Double(safeSampleCount))
        let mse = Float(squaredErrorSum / Double(safeSampleCount))
        let rmse = sqrt(mse)
        let lumaRMSE = Float(sqrt(lumaSquaredErrorSum / Double(safeSampleCount)))
        let relativeLumaRMSE = Float(sqrt(relativeLumaSquaredErrorSum / Double(safeSampleCount)))
        let safePeak = max(peakReferenceValue, 1e-4)
        let peakPSNR: Float = mse > 1e-12 ? (10.0 * log10((safePeak * safePeak) / mse)) : .infinity

        return HDRReferenceFrameMetrics(meanAbsoluteError: meanAbsoluteError,
                                        rmse: rmse,
                                        peakPSNR: peakPSNR,
                                        lumaRMSE: lumaRMSE,
                                        relativeLumaRMSE: relativeLumaRMSE,
                                        maxAbsoluteError: maxAbsoluteError)
    }

    static func computeMagentaCoverage(texture: MTLTexture, commandQueue: MTLCommandQueue) -> Float {
        guard let pixels = readRGBA8Pixels(texture: texture, commandQueue: commandQueue) else {
            return 0
        }

        let totalPixels = max(texture.width * texture.height, 1)
        var magentaPixels = 0

        for i in stride(from: 0, to: pixels.count, by: 4) {
            let red = Float(pixels[i]) / 255.0
            let green = Float(pixels[i + 1]) / 255.0
            let blue = Float(pixels[i + 2]) / 255.0

            if red > 0.45 && blue > 0.45 && green < min(red, blue) * 0.55 {
                magentaPixels += 1
            }
        }

        return Float(magentaPixels) / Float(totalPixels) * 100.0
    }
}
