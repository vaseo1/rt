import Foundation
import simd

class Camera {
    var position: SIMD3<Float> = SIMD3<Float>(0, 50, 0)
    var yaw: Float = 0        // radians, around Y axis
    var pitch: Float = 0      // radians, around X axis

    var fovY: Float = 60.0    // degrees
    var nearZ: Float = 0.1
    var farZ: Float = 10000.0
    var aspectRatio: Float = 16.0 / 9.0
    var aperture: Float = 0.0           // 0 = pinhole (no DOF), try 2.0 for subtle effect
    var focusDistance: Float = 500.0    // distance to focus plane

    var moveSpeed: Float = 200.0
    var sprintMultiplier: Float = 3.0
    var mouseSensitivity: Float = 0.002

    private(set) var isMoving: Bool = false

    // Halton jitter for sub-pixel sampling
    private var haltonIndex: Int = 0

    // Previous frame matrix for motion vectors
    private(set) var previousViewProjectionMatrix: float4x4 = .identity

    func update(forward: Bool, back: Bool, left: Bool, right: Bool,
                up: Bool, sprint: Bool,
                mouseDeltaX: Float, mouseDeltaY: Float, dt: Float) {

        // Store previous VP before updating
        previousViewProjectionMatrix = viewProjectionMatrix

        // Mouse look
        let hasMouseInput = abs(mouseDeltaX) > 0.5 || abs(mouseDeltaY) > 0.5
        yaw += mouseDeltaX * mouseSensitivity
        pitch -= mouseDeltaY * mouseSensitivity
        pitch = max(-.pi / 2.0 + 0.01, min(.pi / 2.0 - 0.01, pitch))

        // Movement
        let hasHorizontalInput = forward || back || left || right
        let hasKeyInput = hasHorizontalInput || up
        isMoving = hasKeyInput || hasMouseInput

        if hasKeyInput {
            let fwd = forwardVector
            let rgt = rightVector
            var move = SIMD3<Float>(0, 0, 0)

            if forward { move += fwd }
            if back    { move -= fwd }
            if right   { move += rgt }
            if left    { move -= rgt }
            if up      { move.y += 1 }

            let len = length(move)
            if len > 0.001 {
                let speed = moveSpeed * (sprint && hasHorizontalInput ? sprintMultiplier : 1.0)
                move = normalize(move) * speed * dt
                position += move
            }
        }
    }

    var forwardVector: SIMD3<Float> {
        SIMD3<Float>(
            sin(yaw) * cos(pitch),
            sin(pitch),
            -cos(yaw) * cos(pitch)
        )
    }

    var rightVector: SIMD3<Float> {
        SIMD3<Float>(cos(yaw), 0, sin(yaw))
    }

    var upVector: SIMD3<Float> {
        cross(rightVector, forwardVector)
    }

    var viewMatrix: float4x4 {
        let f = forwardVector
        let r = rightVector
        let u = cross(r, f)

        return float4x4(rows: [
            SIMD4<Float>(r.x,  r.y,  r.z, -dot(r, position)),
            SIMD4<Float>(u.x,  u.y,  u.z, -dot(u, position)),
            SIMD4<Float>(-f.x, -f.y, -f.z, dot(f, position)),
            SIMD4<Float>(0,    0,    0,    1)
        ])
    }

    var projectionMatrix: float4x4 {
        let fov = fovY * .pi / 180.0
        let tanHalfFov = tan(fov / 2.0)

        var m = float4x4(0)
        m[0][0] = 1.0 / (aspectRatio * tanHalfFov)
        m[1][1] = 1.0 / tanHalfFov
        m[2][2] = -(farZ + nearZ) / (farZ - nearZ)
        m[2][3] = -1.0
        m[3][2] = -(2.0 * farZ * nearZ) / (farZ - nearZ)
        return m
    }

    var viewProjectionMatrix: float4x4 {
        projectionMatrix * viewMatrix
    }

    var inverseViewProjectionMatrix: float4x4 {
        viewProjectionMatrix.inverse
    }

    // Halton sequence for sub-pixel jitter (base 2, base 3)
    var jitterOffset: SIMD2<Float> {
        let idx = haltonIndex
        return SIMD2<Float>(halton(index: idx, base: 2),
                            halton(index: idx, base: 3))
    }

    func advanceJitter() {
        haltonIndex += 1
        if haltonIndex > 255 { haltonIndex = 0 }
    }

    func syncHistory() {
        previousViewProjectionMatrix = viewProjectionMatrix
    }

    func lookAt(target: SIMD3<Float>) {
        let toTarget = target - position
        let distanceToTarget = length(toTarget)
        guard distanceToTarget > 0.001 else { return }

        let direction = toTarget / distanceToTarget
        yaw = atan2(direction.x, -direction.z)
        pitch = asin(max(-1.0, min(1.0, direction.y)))
        pitch = max(-.pi / 2.0 + 0.01, min(.pi / 2.0 - 0.01, pitch))
    }

    func resetAccumulation() {
        haltonIndex = 0
    }

    private func halton(index: Int, base: Int) -> Float {
        var result: Float = 0
        var f: Float = 1.0
        var i = index
        while i > 0 {
            f /= Float(base)
            result += f * Float(i % base)
            i /= base
        }
        return result
    }
}

// MARK: - float4x4 helpers

extension float4x4 {
    static var identity: float4x4 {
        float4x4(diagonal: SIMD4<Float>(1, 1, 1, 1))
    }

    init(_ value: Float) {
        self = float4x4(diagonal: SIMD4<Float>(repeating: value))
    }
}
