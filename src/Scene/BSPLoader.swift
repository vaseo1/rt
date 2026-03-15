import Foundation
import simd

// ─── Quake 1 BSP v29 Loader ─────────────────────────────────────────────────
//
// Parses the binary BSP format used by Quake 1.
// Reference: https://www.gamers.org/dEngine/quake/spec/quake-spec34/qkspec_4.htm

// MARK: - BSP Constants

private let bspVersion: Int32 = 29

private let lumpEntities    = 0
private let lumpPlanes      = 1
private let lumpTextures    = 2
private let lumpVertices    = 3
private let lumpVisibility  = 4
private let lumpNodes       = 5
private let lumpTexInfo     = 6
private let lumpFaces       = 7
private let lumpLighting    = 8
private let lumpClipNodes   = 9
private let lumpLeaves      = 10
private let lumpMarkSurfaces = 11
private let lumpEdges       = 12
private let lumpSurfEdges   = 13
private let lumpModels      = 14
private let lumpCount       = 15

// MARK: - BSP Structs

private struct BSPLump {
    let offset: Int32
    let length: Int32
}

private struct BSPVertex {
    let x: Float
    let y: Float
    let z: Float

    var simd: SIMD3<Float> {
        // Quake uses Z-up; convert to Y-up
        SIMD3<Float>(x, z, -y)
    }
}

private struct BSPEdge {
    let v0: UInt16
    let v1: UInt16
}

private struct BSPFace {
    let planeId: Int16
    let side: Int16
    let firstEdge: Int32
    let numEdges: Int16
    let texInfoId: Int16
    let lightStyles: (UInt8, UInt8, UInt8, UInt8)
    let lightmapOffset: Int32
}

private struct BSPTexInfo {
    let uAxis: SIMD3<Float>
    let uOffset: Float
    let vAxis: SIMD3<Float>
    let vOffset: Float
    let mipTexIndex: UInt32
    let flags: UInt32
}

private struct BSPPlane {
    let normal: SIMD3<Float>
    let dist: Float
    let type: Int32
}

private struct BSPMipTex {
    let name: String        // 16 chars
    let width: UInt32
    let height: UInt32
    let offsets: (UInt32, UInt32, UInt32, UInt32) // mip offsets
    let pixelData: Data?    // palette indices for mip level 0
}

// MARK: - Parsed Output

struct ParsedVertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var uv: SIMD2<Float>
    var materialIndex: UInt32
}

enum MaterialSurfaceType: UInt32 {
    case diffuse = 0
    case metal = 1
    case liquid = 2
    case emissive = 3
}

struct ParsedMaterial {
    var name: String
    var albedo: SIMD3<Float>
    var emissiveStrength: Float
    var emissiveColor: SIMD3<Float>
    var surfaceType: UInt32
    var roughness: Float
    var metallic: Float
    var transmissive: Float
    var ior: Float
    var textureWidth: Int
    var textureHeight: Int
    var texturePixels: [UInt8]?
}

private struct MaterialClassification {
    let surfaceType: UInt32
    let roughness: Float
    let metallic: Float
    let transmissive: Float
    let ior: Float
}

struct BSPData {
    var vertices: [ParsedVertex]
    var indices: [UInt32]
    var materials: [ParsedMaterial]
    var entityString: String
    var spawnPosition: SIMD3<Float>
    var spawnAngle: Float
}

// MARK: - BSP Loader

class BSPLoader {
    private var data: Data = Data()
    private var lumps: [BSPLump] = []
    private var palette: [UInt8]?

    func load(from url: URL, palette: [UInt8]? = nil) throws -> BSPData {
        data = try Data(contentsOf: url)
        self.palette = palette
        return try parse()
    }

    func load(data: Data, palette: [UInt8]? = nil) throws -> BSPData {
        self.data = data
        self.palette = palette
        return try parse()
    }

    private func parse() throws -> BSPData {
        guard data.count > 4 + lumpCount * 8 else {
            throw BSPError.invalidFile("File too small")
        }

        let version: Int32 = readValue(at: 0)
        guard version == bspVersion else {
            throw BSPError.invalidFile("Invalid BSP version: \(version), expected \(bspVersion)")
        }

        // Read lump directory
        lumps = (0..<lumpCount).map { i in
            let base = 4 + i * 8
            return BSPLump(
                offset: readValue(at: base),
                length: readValue(at: base + 4)
            )
        }

        // Parse core lumps
        let vertices = readVertices()
        let edges = readEdges()
        let surfEdges = readSurfEdges()
        let faces = readFaces()
        let texInfos = readTexInfos()
        let planes = readPlanes()
        let mipTextures = readMipTextures()

        // Build materials from textures
        var materials: [ParsedMaterial] = []
        for tex in mipTextures {
            let name = tex.name.lowercased()
            let isEmissive = name.hasPrefix("light") ||
                             name.hasPrefix("*lava") ||
                             name.hasPrefix("*teleport") ||
                             name.hasPrefix("flame") ||
                             name.contains("light")
            let classification = classifyMaterial(named: name, isEmissive: isEmissive)

            // Convert paletted texture to RGBA using Quake palette
            var texturePixels: [UInt8]? = nil
            if let pixelData = tex.pixelData, let pal = palette, pal.count >= 768 {
                let count = Int(tex.width) * Int(tex.height)
                var rgba = [UInt8](repeating: 0, count: count * 4)
                for i in 0..<count {
                    let idx = Int(pixelData[pixelData.startIndex + i])
                    rgba[i * 4 + 0] = pal[idx * 3 + 0]
                    rgba[i * 4 + 1] = pal[idx * 3 + 1]
                    rgba[i * 4 + 2] = pal[idx * 3 + 2]
                    rgba[i * 4 + 3] = (idx == 255) ? 0 : 255
                }
                texturePixels = rgba
            }

            materials.append(ParsedMaterial(
                name: tex.name,
                albedo: colorForTexture(name: name),
                emissiveStrength: isEmissive ? 25.0 : 0.0,
                emissiveColor: isEmissive ? emissiveColorForTexture(name: name) : .zero,
                surfaceType: classification.surfaceType,
                roughness: classification.roughness,
                metallic: classification.metallic,
                transmissive: classification.transmissive,
                ior: classification.ior,
                textureWidth: Int(tex.width),
                textureHeight: Int(tex.height),
                texturePixels: texturePixels
            ))
        }

        if materials.isEmpty {
            materials.append(ParsedMaterial(name: "default", albedo: SIMD3<Float>(0.7, 0.7, 0.7), emissiveStrength: 0,
                                            emissiveColor: .zero, surfaceType: MaterialSurfaceType.diffuse.rawValue,
                                            roughness: 0.85, metallic: 0.0, transmissive: 0.0, ior: 1.5,
                                            textureWidth: 0, textureHeight: 0, texturePixels: nil))
        }

        // Triangulate faces
        var parsedVertices: [ParsedVertex] = []
        var indices: [UInt32] = []

        for face in faces {
            let numEdges = Int(face.numEdges)
            guard numEdges >= 3 else { continue }

            let plane = planes[Int(face.planeId)]
            var faceNormal = SIMD3<Float>(plane.normal.x, plane.normal.z, -plane.normal.y) // convert to Y-up
            if face.side != 0 {
                faceNormal = -faceNormal
            }

            let texInfo = texInfos[Int(face.texInfoId)]
            let matIndex = min(UInt32(texInfo.mipTexIndex), UInt32(materials.count - 1))

            // Gather face vertices from edge list
            var faceVerts: [SIMD3<Float>] = []
            for e in 0..<numEdges {
                let surfEdge: Int32 = surfEdges[Int(face.firstEdge) + e]
                let vertIndex: Int
                if surfEdge >= 0 {
                    vertIndex = Int(edges[Int(surfEdge)].v0)
                } else {
                    vertIndex = Int(edges[Int(-surfEdge)].v1)
                }
                faceVerts.append(vertices[vertIndex].simd)
            }

            // Compute UVs
            let uAxis = SIMD3<Float>(texInfo.uAxis.x, texInfo.uAxis.z, -texInfo.uAxis.y)
            let vAxis = SIMD3<Float>(texInfo.vAxis.x, texInfo.vAxis.z, -texInfo.vAxis.y)

            // Fan triangulation: vertex 0 is hub
            let baseIndex = UInt32(parsedVertices.count)

            let texIdx = min(Int(matIndex), mipTextures.count - 1)
            let texWidth = Float(max(1, mipTextures[texIdx].width))
            let texHeight = Float(max(1, mipTextures[texIdx].height))

            for v in faceVerts {
                let u = dot(v, uAxis) + texInfo.uOffset
                let vCoord = dot(v, vAxis) + texInfo.vOffset
                parsedVertices.append(ParsedVertex(
                    position: v,
                    normal: faceNormal,
                    uv: SIMD2<Float>(u / texWidth, vCoord / texHeight),
                    materialIndex: matIndex
                ))
            }

            for i in 1..<(numEdges - 1) {
                indices.append(baseIndex)
                indices.append(baseIndex + UInt32(i))
                indices.append(baseIndex + UInt32(i + 1))
            }
        }

        // Smooth normals at shared vertices
        smoothNormals(vertices: &parsedVertices, indices: indices)

        // Parse entities for spawn point
        let entityStr = readEntityString()
        let (spawnPos, spawnAngle) = parseSpawnPoint(from: entityStr)

        print("[BSP] Loaded: \(parsedVertices.count) vertices, \(indices.count / 3) triangles, \(materials.count) materials")

        return BSPData(
            vertices: parsedVertices,
            indices: indices,
            materials: materials,
            entityString: entityStr,
            spawnPosition: spawnPos,
            spawnAngle: spawnAngle
        )
    }

    // MARK: - Lump Readers

    private func readVertices() -> [BSPVertex] {
        let lump = lumps[lumpVertices]
        let count = Int(lump.length) / 12 // 3 floats × 4 bytes
        return (0..<count).map { i in
            let base = Int(lump.offset) + i * 12
            return BSPVertex(
                x: readValue(at: base),
                y: readValue(at: base + 4),
                z: readValue(at: base + 8)
            )
        }
    }

    private func readEdges() -> [BSPEdge] {
        let lump = lumps[lumpEdges]
        let count = Int(lump.length) / 4
        return (0..<count).map { i in
            let base = Int(lump.offset) + i * 4
            return BSPEdge(
                v0: readValue(at: base),
                v1: readValue(at: base + 2)
            )
        }
    }

    private func readSurfEdges() -> [Int32] {
        let lump = lumps[lumpSurfEdges]
        let count = Int(lump.length) / 4
        return (0..<count).map { i in
            readValue(at: Int(lump.offset) + i * 4)
        }
    }

    private func readFaces() -> [BSPFace] {
        let lump = lumps[lumpFaces]
        let faceSize = 20
        let count = Int(lump.length) / faceSize
        return (0..<count).map { i in
            let base = Int(lump.offset) + i * faceSize
            return BSPFace(
                planeId: readValue(at: base),
                side: readValue(at: base + 2),
                firstEdge: readValue(at: base + 4),
                numEdges: readValue(at: base + 8),
                texInfoId: readValue(at: base + 10),
                lightStyles: (
                    readValue(at: base + 12),
                    readValue(at: base + 13),
                    readValue(at: base + 14),
                    readValue(at: base + 15)
                ),
                lightmapOffset: readValue(at: base + 16)
            )
        }
    }

    private func readTexInfos() -> [BSPTexInfo] {
        let lump = lumps[lumpTexInfo]
        let tiSize = 40
        let count = Int(lump.length) / tiSize
        return (0..<count).map { i in
            let base = Int(lump.offset) + i * tiSize
            return BSPTexInfo(
                uAxis: SIMD3<Float>(
                    readValue(at: base),
                    readValue(at: base + 4),
                    readValue(at: base + 8)
                ),
                uOffset: readValue(at: base + 12),
                vAxis: SIMD3<Float>(
                    readValue(at: base + 16),
                    readValue(at: base + 20),
                    readValue(at: base + 24)
                ),
                vOffset: readValue(at: base + 28),
                mipTexIndex: readValue(at: base + 32),
                flags: readValue(at: base + 36)
            )
        }
    }

    private func readPlanes() -> [BSPPlane] {
        let lump = lumps[lumpPlanes]
        let planeSize = 20
        let count = Int(lump.length) / planeSize
        return (0..<count).map { i in
            let base = Int(lump.offset) + i * planeSize
            return BSPPlane(
                normal: SIMD3<Float>(
                    readValue(at: base),
                    readValue(at: base + 4),
                    readValue(at: base + 8)
                ),
                dist: readValue(at: base + 12),
                type: readValue(at: base + 16)
            )
        }
    }

    private func readMipTextures() -> [BSPMipTex] {
        let lump = lumps[lumpTextures]
        let base = Int(lump.offset)

        guard lump.length > 4 else { return [] }
        let numTextures: Int32 = readValue(at: base)

        var textures: [BSPMipTex] = []
        for i in 0..<Int(numTextures) {
            let offsetToTex: Int32 = readValue(at: base + 4 + i * 4)
            guard offsetToTex >= 0 else { continue }
            let texBase = base + Int(offsetToTex)

            // Read 16-byte name
            let nameData = data[texBase..<(texBase + 16)]
            let name = String(data: nameData, encoding: .ascii)?
                .trimmingCharacters(in: .controlCharacters)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\0")) ?? "unnamed"

            let width: UInt32 = readValue(at: texBase + 16)
            let height: UInt32 = readValue(at: texBase + 20)
            let offset0: UInt32 = readValue(at: texBase + 24)

            // Extract mip level 0 pixel data if embedded
            var pixelData: Data? = nil
            if offset0 > 0 && width > 0 && height > 0 {
                let pixelStart = texBase + Int(offset0)
                let pixelCount = Int(width) * Int(height)
                if pixelStart >= 0 && pixelStart + pixelCount <= data.count {
                    pixelData = data[pixelStart..<(pixelStart + pixelCount)]
                }
            }

            textures.append(BSPMipTex(
                name: name,
                width: width,
                height: height,
                offsets: (
                    offset0,
                    readValue(at: texBase + 28),
                    readValue(at: texBase + 32),
                    readValue(at: texBase + 36)
                ),
                pixelData: pixelData
            ))
        }
        return textures
    }

    private func readEntityString() -> String {
        let lump = lumps[lumpEntities]
        let entityData = data[Int(lump.offset)..<(Int(lump.offset) + Int(lump.length))]
        return String(data: entityData, encoding: .ascii) ?? ""
    }

    // MARK: - Helpers

    private func readValue<T>(at offset: Int) -> T {
        data.withUnsafeBytes { ptr in
            let value = UnsafeMutablePointer<T>.allocate(capacity: 1)
            defer { value.deallocate() }
            memcpy(value, ptr.baseAddress! + offset, MemoryLayout<T>.size)
            return value.pointee
        }
    }

    private func colorForTexture(name: String) -> SIMD3<Float> {
        // Approximate colors for common Quake textures by name prefix
        let n = name.lowercased()
        if n.hasPrefix("sky")       { return SIMD3<Float>(0.5, 0.6, 0.9) }
        if n.hasPrefix("*lava")     { return SIMD3<Float>(1.0, 0.3, 0.0) }
        if n.hasPrefix("*water")    { return SIMD3<Float>(0.1, 0.3, 0.5) }
        if n.hasPrefix("*slime")    { return SIMD3<Float>(0.1, 0.5, 0.0) }
        if n.hasPrefix("*teleport") { return SIMD3<Float>(0.5, 0.0, 1.0) }
        if n.hasPrefix("light")     { return SIMD3<Float>(1.0, 0.9, 0.7) }
        if n.contains("metal")      { return SIMD3<Float>(0.5, 0.5, 0.55) }
        if n.contains("wood")       { return SIMD3<Float>(0.5, 0.35, 0.2) }
        if n.contains("brick")      { return SIMD3<Float>(0.55, 0.3, 0.2) }
        if n.contains("wiz")        { return SIMD3<Float>(0.35, 0.25, 0.4) }
        if n.contains("door")       { return SIMD3<Float>(0.4, 0.3, 0.2) }
        // Default stone/concrete
        return SIMD3<Float>(0.45, 0.42, 0.38)
    }

    private func emissiveColorForTexture(name: String) -> SIMD3<Float> {
        let n = name.lowercased()
        if n.hasPrefix("*lava")     { return SIMD3<Float>(1.0, 0.4, 0.05) }
        if n.hasPrefix("*teleport") { return SIMD3<Float>(0.6, 0.1, 1.0) }
        if n.hasPrefix("flame")     { return SIMD3<Float>(1.0, 0.6, 0.1) }
        // Default warm white for light textures
        return SIMD3<Float>(1.0, 0.9, 0.7)
    }

    private func classifyMaterial(named name: String, isEmissive: Bool) -> MaterialClassification {
        let n = name.lowercased()

        if isEmissive {
            return MaterialClassification(
                surfaceType: MaterialSurfaceType.emissive.rawValue,
                roughness: n.hasPrefix("*lava") ? 0.45 : 0.3,
                metallic: 0.0,
                transmissive: 0.0,
                ior: 1.0
            )
        }

        if n.hasPrefix("*water") {
            return MaterialClassification(
                surfaceType: MaterialSurfaceType.liquid.rawValue,
                roughness: 0.02,
                metallic: 0.0,
                transmissive: 0.97,
                ior: 1.333
            )
        }

        if n.hasPrefix("*slime") {
            return MaterialClassification(
                surfaceType: MaterialSurfaceType.liquid.rawValue,
                roughness: 0.05,
                metallic: 0.0,
                transmissive: 0.82,
                ior: 1.38
            )
        }

        if n.contains("metal") || n.contains("chrome") || n.contains("steel") ||
            n.contains("iron") || n.contains("grate") || n.contains("chain") ||
            n.contains("gear") || n.contains("plate") || n.contains("bolt") ||
            n.contains("silver") || n.contains("copper") || n.contains("brass") ||
            (n.contains("door") && !n.contains("wood")) {
            let roughness: Float
            if n.contains("chrome") {
                roughness = 0.08
            } else if n.contains("grate") || n.contains("chain") {
                roughness = 0.32
            } else if n.contains("door") {
                roughness = 0.28
            } else {
                roughness = 0.18
            }

            return MaterialClassification(
                surfaceType: MaterialSurfaceType.metal.rawValue,
                roughness: roughness,
                metallic: 0.92,
                transmissive: 0.0,
                ior: 1.5
            )
        }

        if n.contains("wood") || n.contains("crate") || n.contains("beam") {
            return MaterialClassification(
                surfaceType: MaterialSurfaceType.diffuse.rawValue,
                roughness: 0.72,
                metallic: 0.0,
                transmissive: 0.0,
                ior: 1.5
            )
        }

        if n.contains("brick") || n.contains("stone") || n.contains("rock") ||
            n.contains("wall") || n.contains("mortar") {
            return MaterialClassification(
                surfaceType: MaterialSurfaceType.diffuse.rawValue,
                roughness: 0.88,
                metallic: 0.0,
                transmissive: 0.0,
                ior: 1.5
            )
        }

        if n.contains("floor") || n.contains("tile") || n.contains("trim") {
            return MaterialClassification(
                surfaceType: MaterialSurfaceType.diffuse.rawValue,
                roughness: 0.62,
                metallic: 0.0,
                transmissive: 0.0,
                ior: 1.5
            )
        }

        if n.hasPrefix("sky") {
            return MaterialClassification(
                surfaceType: MaterialSurfaceType.diffuse.rawValue,
                roughness: 1.0,
                metallic: 0.0,
                transmissive: 0.0,
                ior: 1.0
            )
        }

        return MaterialClassification(
            surfaceType: MaterialSurfaceType.diffuse.rawValue,
            roughness: 0.82,
            metallic: 0.0,
            transmissive: 0.0,
            ior: 1.5
        )
    }

    private func smoothNormals(vertices: inout [ParsedVertex], indices: [UInt32]) {
        let cosThreshold: Float = cos(60.0 * .pi / 180.0)
        let scale: Float = 100.0

        var cornerWeights = [Float](repeating: 0, count: vertices.count)
        for tri in stride(from: 0, to: indices.count, by: 3) {
            let i0 = Int(indices[tri])
            let i1 = Int(indices[tri + 1])
            let i2 = Int(indices[tri + 2])

            let p0 = vertices[i0].position
            let p1 = vertices[i1].position
            let p2 = vertices[i2].position

            cornerWeights[i0] += cornerAngle(at: p0, prev: p2, next: p1)
            cornerWeights[i1] += cornerAngle(at: p1, prev: p0, next: p2)
            cornerWeights[i2] += cornerAngle(at: p2, prev: p1, next: p0)
        }

        var posMap: [SIMD3<Int32>: [Int]] = [:]
        for i in 0..<vertices.count {
            let p = vertices[i].position
            let key = SIMD3<Int32>(Int32(round(p.x * scale)),
                                   Int32(round(p.y * scale)),
                                   Int32(round(p.z * scale)))
            posMap[key, default: []].append(i)
        }

        var newNormals = vertices.map { $0.normal }
        for (_, group) in posMap {
            guard group.count > 1 else { continue }
            for idx in group {
                let origNormal = vertices[idx].normal
                var smooth = SIMD3<Float>(0, 0, 0)
                for otherIdx in group {
                    if dot(origNormal, vertices[otherIdx].normal) >= cosThreshold {
                        let weight = max(cornerWeights[otherIdx], 0.0)
                        smooth += vertices[otherIdx].normal * max(weight, 0.0001)
                    }
                }
                let len = length(smooth)
                if len > 0.001 {
                    newNormals[idx] = smooth / len
                }
            }
        }
        for i in 0..<vertices.count {
            vertices[i].normal = newNormals[i]
        }
    }

    private func cornerAngle(at current: SIMD3<Float>, prev: SIMD3<Float>, next: SIMD3<Float>) -> Float {
        let edge0 = prev - current
        let edge1 = next - current

        let len0 = length(edge0)
        let len1 = length(edge1)
        guard len0 > 0.0001, len1 > 0.0001 else { return 0 }

        let dir0 = edge0 / len0
        let dir1 = edge1 / len1
        let cosine = max(-1.0, min(1.0, dot(dir0, dir1)))
        return acos(cosine)
    }

    private func parseSpawnPoint(from entityStr: String) -> (SIMD3<Float>, Float) {
        // Find info_player_start entity
        var inPlayerStart = false
        var origin = SIMD3<Float>(0, 100, 0)
        var angle: Float = 0

        let lines = entityStr.components(separatedBy: .newlines)
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.contains("info_player_start") {
                inPlayerStart = true
            }
            if inPlayerStart && trimmed.contains("\"origin\"") {
                if let value = extractQuotedValue(from: trimmed, key: "origin") {
                    let parts = value.split(separator: " ").compactMap { Float($0) }
                    if parts.count == 3 {
                        // Convert Quake coords (Z-up) to our coords (Y-up)
                        origin = SIMD3<Float>(parts[0], parts[2], -parts[1])
                    }
                }
            }
            if inPlayerStart && trimmed.contains("\"angle\"") {
                if let value = extractQuotedValue(from: trimmed, key: "angle") {
                    angle = (Float(value) ?? 0) * .pi / 180.0
                }
            }
            if inPlayerStart && trimmed == "}" {
                break
            }
        }

        return (origin, angle)
    }

    private func extractQuotedValue(from line: String, key: String) -> String? {
        // Parse: "key" "value"
        let parts = line.split(separator: "\"").map(String.init)
        for i in 0..<parts.count {
            if parts[i] == key, i + 2 < parts.count {
                return parts[i + 2]
            }
        }
        return nil
    }
}

// MARK: - PAK Archive Loader

private func readUnaligned<T>(from data: Data, at offset: Int) -> T {
    data.withUnsafeBytes { ptr in
        let value = UnsafeMutablePointer<T>.allocate(capacity: 1)
        defer { value.deallocate() }
        memcpy(value, ptr.baseAddress! + offset, MemoryLayout<T>.size)
        return value.pointee
    }
}

class PAKLoader {
    struct PAKEntry {
        let name: String
        let offset: Int
        let size: Int
    }

    func extractBSP(from pakURL: URL, mapName: String = "maps/e1m1.bsp") throws -> Data {
        let pakData = try Data(contentsOf: pakURL)

        guard pakData.count > 12 else {
            throw BSPError.invalidFile("PAK file too small")
        }

        // PAK header: "PACK" magic + dir offset + dir length
        let magic = String(data: pakData[0..<4], encoding: .ascii)
        guard magic == "PACK" else {
            throw BSPError.invalidFile("Not a PAK file (magic: \(magic ?? "nil"))")
        }

        let dirOffset: Int32 = readUnaligned(from: pakData, at: 4)
        let dirLength: Int32 = readUnaligned(from: pakData, at: 8)
        let entryCount = Int(dirLength) / 64

        // Read directory
        for i in 0..<entryCount {
            let entryBase = Int(dirOffset) + i * 64
            let nameData = pakData[entryBase..<(entryBase + 56)]
            let name = String(data: nameData, encoding: .ascii)?
                .trimmingCharacters(in: CharacterSet(charactersIn: "\0")) ?? ""
            let fileOffset: Int32 = readUnaligned(from: pakData, at: entryBase + 56)
            let fileSize: Int32 = readUnaligned(from: pakData, at: entryBase + 60)

            if name.lowercased() == mapName.lowercased() {
                let start = Int(fileOffset)
                let end = start + Int(fileSize)
                guard end <= pakData.count else {
                    throw BSPError.invalidFile("BSP data extends beyond PAK file")
                }
                print("[PAK] Found \(name): \(fileSize) bytes at offset \(fileOffset)")
                return Data(pakData[start..<end])
            }
        }

        throw BSPError.mapNotFound("Map '\(mapName)' not found in PAK file")
    }

    func extractFile(from pakURL: URL, name: String) throws -> Data {
        let pakData = try Data(contentsOf: pakURL)
        guard pakData.count > 12 else {
            throw BSPError.invalidFile("PAK file too small")
        }
        let magic = String(data: pakData[0..<4], encoding: .ascii)
        guard magic == "PACK" else {
            throw BSPError.invalidFile("Not a PAK file")
        }
        let dirOffset: Int32 = readUnaligned(from: pakData, at: 4)
        let dirLength: Int32 = readUnaligned(from: pakData, at: 8)
        let entryCount = Int(dirLength) / 64

        for i in 0..<entryCount {
            let entryBase = Int(dirOffset) + i * 64
            let nameData = pakData[entryBase..<(entryBase + 56)]
            let entryName = String(data: nameData, encoding: .ascii)?
                .trimmingCharacters(in: CharacterSet(charactersIn: "\0")) ?? ""
            let fileOffset: Int32 = readUnaligned(from: pakData, at: entryBase + 56)
            let fileSize: Int32 = readUnaligned(from: pakData, at: entryBase + 60)

            if entryName.lowercased() == name.lowercased() {
                let start = Int(fileOffset)
                let end = start + Int(fileSize)
                guard end <= pakData.count else {
                    throw BSPError.invalidFile("File extends beyond PAK")
                }
                return Data(pakData[start..<end])
            }
        }
        throw BSPError.mapNotFound("'\(name)' not found in PAK")
    }

    func listEntries(from pakURL: URL) throws -> [PAKEntry] {
        let pakData = try Data(contentsOf: pakURL)
        guard pakData.count > 12 else { return [] }

        let magic = String(data: pakData[0..<4], encoding: .ascii)
        guard magic == "PACK" else { return [] }

        let dirOffset: Int32 = readUnaligned(from: pakData, at: 4)
        let dirLength: Int32 = readUnaligned(from: pakData, at: 8)
        let entryCount = Int(dirLength) / 64

        var entries: [PAKEntry] = []
        for i in 0..<entryCount {
            let entryBase = Int(dirOffset) + i * 64
            let nameData = pakData[entryBase..<(entryBase + 56)]
            let name = String(data: nameData, encoding: .ascii)?
                .trimmingCharacters(in: CharacterSet(charactersIn: "\0")) ?? ""
            let fileOffset: Int32 = readUnaligned(from: pakData, at: entryBase + 56)
            let fileSize: Int32 = readUnaligned(from: pakData, at: entryBase + 60)
            entries.append(PAKEntry(name: name, offset: Int(fileOffset), size: Int(fileSize)))
        }
        return entries
    }
}

// MARK: - Errors

enum BSPError: Error, LocalizedError {
    case invalidFile(String)
    case mapNotFound(String)

    var errorDescription: String? {
        switch self {
        case .invalidFile(let msg): return "Invalid BSP/PAK: \(msg)"
        case .mapNotFound(let msg): return msg
        }
    }
}
