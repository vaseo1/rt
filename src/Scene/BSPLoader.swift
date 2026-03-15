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
}

// MARK: - Parsed Output

struct ParsedVertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var uv: SIMD2<Float>
    var materialIndex: UInt32
}

struct ParsedMaterial {
    var name: String
    var albedo: SIMD3<Float>
    var emissiveStrength: Float
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

    func load(from url: URL) throws -> BSPData {
        data = try Data(contentsOf: url)
        return try parse()
    }

    func load(data: Data) throws -> BSPData {
        self.data = data
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
            materials.append(ParsedMaterial(
                name: tex.name,
                albedo: colorForTexture(name: name),
                emissiveStrength: isEmissive ? 25.0 : 0.0
            ))
        }

        if materials.isEmpty {
            materials.append(ParsedMaterial(name: "default", albedo: SIMD3<Float>(0.7, 0.7, 0.7), emissiveStrength: 0))
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

            for v in faceVerts {
                let u = dot(v, uAxis) + texInfo.uOffset
                let vCoord = dot(v, vAxis) + texInfo.vOffset
                parsedVertices.append(ParsedVertex(
                    position: v,
                    normal: faceNormal,
                    uv: SIMD2<Float>(u / 64.0, vCoord / 64.0), // normalize roughly
                    materialIndex: matIndex
                ))
            }

            for i in 1..<(numEdges - 1) {
                indices.append(baseIndex)
                indices.append(baseIndex + UInt32(i))
                indices.append(baseIndex + UInt32(i + 1))
            }
        }

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

            textures.append(BSPMipTex(
                name: name,
                width: width,
                height: height,
                offsets: (
                    readValue(at: texBase + 24),
                    readValue(at: texBase + 28),
                    readValue(at: texBase + 32),
                    readValue(at: texBase + 36)
                )
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
            var value = UnsafeMutablePointer<T>.allocate(capacity: 1)
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
        var value = UnsafeMutablePointer<T>.allocate(capacity: 1)
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
