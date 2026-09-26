import Foundation
import Darwin
import Dispatch

public enum FlowIPCServerError: LocalizedError, Equatable {
    case alreadyRunning
    case socketPathTooLong
    case systemCall(String, Int32)

    public var errorDescription: String? {
        switch self {
        case .alreadyRunning:
            return "FlowSwift IPC is already running"
        case .socketPathTooLong:
            return "FlowSwift IPC socket path is too long"
        case .systemCall(let name, let code):
            return "FlowSwift IPC \(name) failed: \(String(cString: strerror(code)))"
        }
    }
}

/// Local JSONL server used by Darkadyan. The server owns the socket path only
/// after a successful bind and removes it during orderly shutdown.
public final class FlowIPCServer {
    public typealias Handler = (FlowIPCRequest) -> FlowIPCResponse

    public let socketURL: URL
    private let handler: Handler
    private let queue = DispatchQueue(label: "com.shaun.flowswift.ipc")
    private let queueKey = DispatchSpecificKey<UInt8>()

    private final class Connection {
        let fd: Int32
        var buffer = Data()
        var source: DispatchSourceRead?

        init(fd: Int32) { self.fd = fd }
    }

    private var listenFD: Int32 = -1
    private var acceptSource: DispatchSourceRead?
    private var connections: [Int32: Connection] = [:]
    private var correlationOwners: [String: Int32] = [:]
    private var ownsSocketPath = false

    public init(socketURL: URL = FlowIPCProtocol.socketURL, handler: @escaping Handler) {
        self.socketURL = socketURL
        self.handler = handler
        queue.setSpecific(key: queueKey, value: 1)
    }

    deinit { stop() }

    public var isRunning: Bool {
        performSync { listenFD >= 0 }
    }

    public func start() throws {
        if DispatchQueue.getSpecific(key: queueKey) != nil {
            try startLocked()
            return
        }
        var caught: Error?
        queue.sync {
            do { try startLocked() }
            catch { caught = error }
        }
        if let caught { throw caught }
    }

    public func stop() {
        if DispatchQueue.getSpecific(key: queueKey) != nil { stopLocked(); return }
        queue.sync { stopLocked() }
    }

    public func emit(_ event: FlowIPCEvent) {
        let work: () -> Void = { [weak self] in self?.emitLocked(event) }
        if DispatchQueue.getSpecific(key: queueKey) != nil { work() }
        else { queue.async(execute: work) }
    }

    private func startLocked() throws {
        guard listenFD < 0 else { return }
        let directory = socketURL.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            _ = chmod(directory.path, 0o700)
        } catch {
            throw FlowIPCServerError.systemCall("mkdir", errno)
        }

        if FileManager.default.fileExists(atPath: socketURL.path) {
            if probeLiveServer(at: socketURL.path) {
                throw FlowIPCServerError.alreadyRunning
            }
            guard unlink(socketURL.path) == 0 || errno == ENOENT else {
                throw FlowIPCServerError.systemCall("unlink", errno)
            }
        }

        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw FlowIPCServerError.systemCall("socket", errno) }
        do {
            try withSocketAddress(path: socketURL.path) { address, length in
                guard Darwin.bind(fd, address, length) == 0 else {
                    throw FlowIPCServerError.systemCall("bind", errno)
                }
            }
            ownsSocketPath = true
            guard chmod(socketURL.path, 0o600) == 0 else {
                throw FlowIPCServerError.systemCall("chmod", errno)
            }
            guard Darwin.listen(fd, 8) == 0 else {
                throw FlowIPCServerError.systemCall("listen", errno)
            }
            let flags = fcntl(fd, F_GETFL, 0)
            guard flags >= 0, fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0 else {
                throw FlowIPCServerError.systemCall("fcntl", errno)
            }
        } catch {
            Darwin.close(fd)
            if ownsSocketPath { _ = unlink(socketURL.path); ownsSocketPath = false }
            throw error
        }

        listenFD = fd
        let source = DispatchSource.makeReadSource(fileDescriptor: fd, queue: queue)
        source.setEventHandler { [weak self] in self?.acceptReadyLocked() }
        acceptSource = source
        source.resume()
    }

    private func stopLocked() {
        guard listenFD >= 0 || ownsSocketPath else { return }
        acceptSource?.cancel()
        acceptSource = nil
        if listenFD >= 0 { Darwin.close(listenFD); listenFD = -1 }
        for fd in Array(connections.keys) { closeConnectionLocked(fd) }
        correlationOwners.removeAll()
        if ownsSocketPath { _ = unlink(socketURL.path); ownsSocketPath = false }
    }

    private func acceptReadyLocked() {
        while listenFD >= 0 {
            let fd = Darwin.accept(listenFD, nil, nil)
            if fd < 0 {
                if errno == EINTR { continue }
                if errno == EAGAIN || errno == EWOULDBLOCK { return }
                return
            }
            var noSigPipe: Int32 = 1
            _ = withUnsafePointer(to: &noSigPipe) {
                setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, $0, socklen_t(MemoryLayout<Int32>.size))
            }
            let connection = Connection(fd: fd)
            let source = DispatchSource.makeReadSource(fileDescriptor: fd, queue: queue)
            source.setEventHandler { [weak self] in self?.readReadyLocked(fd) }
            connection.source = source
            connections[fd] = connection
            source.resume()
        }
    }

    private func readReadyLocked(_ fd: Int32) {
        guard let connection = connections[fd] else { return }
        var bytes = [UInt8](repeating: 0, count: 16_384)
        let count = Darwin.read(fd, &bytes, bytes.count)
        if count == 0 { closeConnectionLocked(fd); return }
        if count < 0 {
            if errno != EINTR && errno != EAGAIN { closeConnectionLocked(fd) }
            return
        }
        connection.buffer.append(bytes, count: count)
        if connection.buffer.count > FlowIPCProtocol.maximumLineBytes {
            writeLocked(FlowIPCResponse.failure(id: "", "request_too_large"), to: fd)
            closeConnectionLocked(fd)
            return
        }
        drainLinesLocked(connection)
    }

    private func drainLinesLocked(_ connection: Connection) {
        while let newline = connection.buffer.firstIndex(of: 0x0A) {
            let line = connection.buffer.prefix(upTo: newline)
            connection.buffer.removeSubrange(...newline)
            guard !line.isEmpty else { continue }
            handleLineLocked(Data(line), from: connection.fd)
        }
    }

    private func handleLineLocked(_ line: Data, from fd: Int32) {
        let request: FlowIPCRequest
        do {
            request = try FlowIPCCodec.decodeRequest(line)
        } catch {
            let id = ((try? JSONSerialization.jsonObject(with: line)) as? [String: Any])?["id"] as? String ?? ""
            writeLocked(FlowIPCResponse.failure(id: id, "invalid_request"), to: fd)
            return
        }
        guard request.version == FlowIPCProtocol.version else {
            writeLocked(FlowIPCResponse.failure(id: request.id, "unsupported_protocol_version"), to: fd)
            return
        }
        guard !request.id.isEmpty else {
            writeLocked(FlowIPCResponse.failure(id: "", "missing_correlation_id"), to: fd)
            return
        }
        correlationOwners[request.id] = fd
        if request.method == .start,
           let explicit = request.params["correlationId"]?.stringValue,
           !explicit.isEmpty {
            correlationOwners[explicit] = fd
        }
        let response = handler(request)
        writeLocked(response, to: fd)
    }

    private func emitLocked(_ event: FlowIPCEvent) {
        let targets: [Int32]
        if let owner = correlationOwners[event.correlationId], connections[owner] != nil {
            targets = [owner]
        } else {
            targets = Array(connections.keys)
        }
        let envelope = FlowIPCEventEnvelope(event: event)
        for fd in targets { writeLocked(envelope, to: fd) }
    }

    private func writeLocked<T: Encodable>(_ value: T, to fd: Int32) {
        guard connections[fd] != nil, let data = try? FlowIPCCodec.encode(value) else { return }
        let ok = data.withUnsafeBytes { raw -> Bool in
            guard let base = raw.baseAddress else { return false }
            var offset = 0
            while offset < raw.count {
                let sent = Darwin.send(fd, base.advanced(by: offset), raw.count - offset, 0)
                if sent > 0 { offset += sent; continue }
                if sent < 0 && errno == EINTR { continue }
                return false
            }
            return true
        }
        if !ok { closeConnectionLocked(fd) }
    }

    private func closeConnectionLocked(_ fd: Int32) {
        guard let connection = connections.removeValue(forKey: fd) else { return }
        connection.source?.cancel()
        connection.source = nil
        Darwin.close(fd)
        correlationOwners = correlationOwners.filter { $0.value != fd }
    }

    private func probeLiveServer(at path: String) -> Bool {
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { Darwin.close(fd) }
        return (try? withSocketAddress(path: path) { address, length in
            Darwin.connect(fd, address, length) == 0
        }) ?? false
    }

    private func withSocketAddress<T>(
        path: String,
        _ body: (UnsafePointer<sockaddr>, socklen_t) throws -> T
    ) throws -> T {
        var address = sockaddr_un()
        let pathBytes = path.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: address.sun_path) else {
            throw FlowIPCServerError.socketPathTooLong
        }
        address.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutableBytes(of: &address.sun_path) { raw in
            raw.initializeMemory(as: UInt8.self, repeating: 0)
            for index in pathBytes.indices {
                raw[index] = UInt8(bitPattern: pathBytes[index])
            }
        }
        let length = MemoryLayout.offset(of: \sockaddr_un.sun_path)! + pathBytes.count
        address.sun_len = UInt8(length)
        return try withUnsafePointer(to: &address) { pointer in
            try pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                try body($0, socklen_t(length))
            }
        }
    }

    private func performSync<T>(_ body: () -> T) -> T {
        if DispatchQueue.getSpecific(key: queueKey) != nil { return body() }
        return queue.sync(execute: body)
    }
}
