import Foundation

/// Stable, local FlowSwift IPC protocol. Version 1 is JSON Lines over a Unix
/// domain socket; every request and response occupies exactly one UTF-8 line.
public enum FlowIPCProtocol {
    public static let version = 1
    public static let bundleIdentifier = "com.shaun.flowswift"
    public static let maximumLineBytes = 1_048_576

    public static var applicationSupportDirectory: URL {
        if let path = ProcessInfo.processInfo.environment["FLOW_IPC_DIR"] {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support", isDirectory: true)
        return base.appendingPathComponent("FlowSwift", isDirectory: true)
    }

    public static var socketURL: URL {
        applicationSupportDirectory.appendingPathComponent("flowswift-v1.sock")
    }
}

public enum FlowIPCMethod: String, Codable, CaseIterable, Sendable {
    case start
    case stop
    case cancel
    case status
    case health
}

/// JSON-compatible value used for request parameters and response payloads.
public enum FlowIPCValue: Codable, Equatable, Sendable {
    case string(String)
    case bool(Bool)
    case int(Int)
    case double(Double)
    case object([String: FlowIPCValue])
    case array([FlowIPCValue])
    case null

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if value.decodeNil() { self = .null }
        else if let v = try? value.decode(Bool.self) { self = .bool(v) }
        else if let v = try? value.decode(Int.self) { self = .int(v) }
        else if let v = try? value.decode(Double.self) { self = .double(v) }
        else if let v = try? value.decode(String.self) { self = .string(v) }
        else if let v = try? value.decode([String: FlowIPCValue].self) { self = .object(v) }
        else if let v = try? value.decode([FlowIPCValue].self) { self = .array(v) }
        else {
            throw DecodingError.dataCorruptedError(
                in: value,
                debugDescription: "Unsupported JSON value"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .string(let v): try value.encode(v)
        case .bool(let v): try value.encode(v)
        case .int(let v): try value.encode(v)
        case .double(let v): try value.encode(v)
        case .object(let v): try value.encode(v)
        case .array(let v): try value.encode(v)
        case .null: try value.encodeNil()
        }
    }

    public var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    public var boolValue: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }
}

public struct FlowIPCRequest: Codable, Equatable, Sendable {
    public let version: Int
    public let id: String
    public let method: FlowIPCMethod
    public let params: [String: FlowIPCValue]

    public init(
        version: Int = FlowIPCProtocol.version,
        id: String,
        method: FlowIPCMethod,
        params: [String: FlowIPCValue] = [:]
    ) {
        self.version = version
        self.id = id
        self.method = method
        self.params = params
    }

    private enum CodingKeys: String, CodingKey { case version, id, method, params }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        version = try values.decode(Int.self, forKey: .version)
        id = try values.decode(String.self, forKey: .id)
        method = try values.decode(FlowIPCMethod.self, forKey: .method)
        params = try values.decodeIfPresent([String: FlowIPCValue].self, forKey: .params) ?? [:]
    }
}

public struct FlowIPCResponse: Codable, Equatable, Sendable {
    public let version: Int
    public let id: String
    public let result: [String: FlowIPCValue]?
    public let error: String?

    public init(
        version: Int = FlowIPCProtocol.version,
        id: String,
        result: [String: FlowIPCValue]? = nil,
        error: String? = nil
    ) {
        self.version = version
        self.id = id
        self.result = result
        self.error = error
    }

    public static func success(
        id: String,
        _ result: [String: FlowIPCValue] = [:]
    ) -> FlowIPCResponse {
        FlowIPCResponse(id: id, result: result)
    }

    public static func failure(id: String, _ error: String) -> FlowIPCResponse {
        FlowIPCResponse(id: id, error: error)
    }
}

public enum FlowIPCEventType: String, Codable, Sendable {
    case partial
    case final
    case cleanup
    case error
    case status
}

public struct FlowIPCEvent: Codable, Equatable, Sendable {
    public let type: FlowIPCEventType
    public let correlationId: String
    public let text: String?
    public let message: String?
    public let state: String?

    public init(
        type: FlowIPCEventType,
        correlationId: String,
        text: String? = nil,
        message: String? = nil,
        state: String? = nil
    ) {
        self.type = type
        self.correlationId = correlationId
        self.text = text
        self.message = message
        self.state = state
    }
}

public struct FlowIPCEventEnvelope: Codable, Equatable, Sendable {
    public let version: Int
    public let event: FlowIPCEvent

    public init(version: Int = FlowIPCProtocol.version, event: FlowIPCEvent) {
        self.version = version
        self.event = event
    }
}

public enum FlowIPCCodec {
    private static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        var data = try makeEncoder().encode(value)
        data.append(0x0A)
        return data
    }

    public static func decodeRequest(_ line: Data) throws -> FlowIPCRequest {
        let payload = line.last == 0x0A ? line.dropLast() : line[...]
        return try JSONDecoder().decode(FlowIPCRequest.self, from: Data(payload))
    }
}
