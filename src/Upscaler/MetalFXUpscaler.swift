import Metal
import MetalFX

// ─── MetalFX Temporal Upscaler ───────────────────────────────────────────────
//
// Wraps MTLFXTemporalScaler for upscaling low-res path-traced frames
// during camera movement. Requires:
//   - Color input (low-res path trace output)
//   - Depth buffer (primary ray hit distance)
//   - Motion vectors (screen-space pixel displacement)
// Outputs a full-resolution temporally stable frame.

class MetalFXUpscaler {
    let device: MTLDevice
    private var temporalScaler: MTLFXTemporalScaler?

    private var inputWidth: Int = 0
    private var inputHeight: Int = 0
    private var outputWidth: Int = 0
    private var outputHeight: Int = 0

    private(set) var outputTexture: MTLTexture?

    init(device: MTLDevice) {
        self.device = device
    }

    func configure(inputWidth: Int, inputHeight: Int,
                   outputWidth: Int, outputHeight: Int,
                   colorFormat: MTLPixelFormat = .rgba32Float,
                   depthFormat: MTLPixelFormat = .r32Float,
                   motionFormat: MTLPixelFormat = .rg16Float,
                   outputFormat: MTLPixelFormat = .rgba32Float) {
        guard inputWidth != self.inputWidth || inputHeight != self.inputHeight
                || outputWidth != self.outputWidth || outputHeight != self.outputHeight else {
            return
        }

        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight

        let desc = MTLFXTemporalScalerDescriptor()
        desc.inputWidth = inputWidth
        desc.inputHeight = inputHeight
        desc.outputWidth = outputWidth
        desc.outputHeight = outputHeight
        desc.colorTextureFormat = colorFormat
        desc.depthTextureFormat = depthFormat
        desc.motionTextureFormat = motionFormat
        desc.outputTextureFormat = outputFormat

        temporalScaler = desc.makeTemporalScaler(device: device)

        if temporalScaler == nil {
            print("[MetalFX] WARNING: Failed to create temporal scaler. MetalFX may not be supported.")
            return
        }

        // Create output texture
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: outputFormat,
            width: outputWidth,
            height: outputHeight,
            mipmapped: false
        )
        texDesc.usage = [.shaderRead, .shaderWrite, .renderTarget]
        texDesc.storageMode = .private
        outputTexture = device.makeTexture(descriptor: texDesc)

        print("[MetalFX] Configured: \(inputWidth)x\(inputHeight) → \(outputWidth)x\(outputHeight)")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                colorTexture: MTLTexture,
                depthTexture: MTLTexture,
                motionTexture: MTLTexture,
                jitterX: Float, jitterY: Float) {
        guard let scaler = temporalScaler,
              let output = outputTexture else { return }

        scaler.colorTexture = colorTexture
        scaler.depthTexture = depthTexture
        scaler.motionTexture = motionTexture
        scaler.outputTexture = output
        scaler.jitterOffsetX = jitterX
        scaler.jitterOffsetY = jitterY
        scaler.reset = false

        scaler.encode(commandBuffer: commandBuffer)
    }

    func reset() {
        temporalScaler?.reset = true
    }

    var isAvailable: Bool {
        temporalScaler != nil
    }
}
