import Metal
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

struct FrameMetrics {
    let avgBrightness: Float
    let nonBlackPercent: Float
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
        print(String(format: "[VERIFY] min_value: %.4f", minValue))
        print(String(format: "[VERIFY] max_value: %.4f", maxValue))
        print("[VERIFY] nan_count: \(nanCount)")
        print("[VERIFY] resolution: \(width)x\(height)")
        print("[VERIFY] status: \(passed ? "PASS" : "FAIL")")
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

    /// Save a bgra8Unorm texture to a PNG file.
    static func capture(texture: MTLTexture, commandQueue: MTLCommandQueue, to url: URL) -> Bool {
        let w = texture.width
        let h = texture.height
        guard var pixels = readPixels(from: texture, commandQueue: commandQueue) else {
            print("[VERIFY] ERROR: Failed to read texture pixels")
            return false
        }

        // BGRA → RGBA swap
        for i in stride(from: 0, to: pixels.count, by: 4) {
            let b = pixels[i]
            pixels[i] = pixels[i + 2]
            pixels[i + 2] = b
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
        guard let pixels = readPixels(from: texture, commandQueue: commandQueue) else {
            return FrameMetrics(avgBrightness: 0, nonBlackPercent: 0,
                                minValue: 0, maxValue: 0, nanCount: 0,
                                width: w, height: h)
        }

        let totalPixels = w * h
        var brightnessSum: Double = 0
        var nonBlackCount = 0
        var nanCount = 0
        var minVal: Float = 1.0
        var maxVal: Float = 0.0

        for i in stride(from: 0, to: pixels.count, by: 4) {
            let b = Float(pixels[i])     / 255.0
            let g = Float(pixels[i + 1]) / 255.0
            let r = Float(pixels[i + 2]) / 255.0

            if r.isNaN || g.isNaN || b.isNaN {
                nanCount += 1
                continue
            }

            // Rec. 709 luminance
            let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            brightnessSum += Double(lum)

            minVal = min(minVal, min(r, min(g, b)))
            maxVal = max(maxVal, max(r, max(g, b)))

            if lum > 0.01 {
                nonBlackCount += 1
            }
        }

        let avgBrightness = totalPixels > 0 ? Float(brightnessSum / Double(totalPixels)) : 0
        let nonBlackPct = totalPixels > 0 ? Float(nonBlackCount) / Float(totalPixels) * 100.0 : 0

        return FrameMetrics(
            avgBrightness: avgBrightness,
            nonBlackPercent: nonBlackPct,
            minValue: minVal,
            maxValue: maxVal,
            nanCount: nanCount,
            width: w,
            height: h
        )
    }

    static func computeMagentaCoverage(texture: MTLTexture, commandQueue: MTLCommandQueue) -> Float {
        guard let pixels = readPixels(from: texture, commandQueue: commandQueue) else {
            return 0
        }

        let totalPixels = max(texture.width * texture.height, 1)
        var magentaPixels = 0

        for i in stride(from: 0, to: pixels.count, by: 4) {
            let blue = Float(pixels[i]) / 255.0
            let green = Float(pixels[i + 1]) / 255.0
            let red = Float(pixels[i + 2]) / 255.0

            if red > 0.45 && blue > 0.45 && green < min(red, blue) * 0.55 {
                magentaPixels += 1
            }
        }

        return Float(magentaPixels) / Float(totalPixels) * 100.0
    }
}
