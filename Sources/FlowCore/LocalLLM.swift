import Foundation
import Darwin

/// Offline punctuation classifier. No generated words and no automatic downloads.
public final class LocalCleanupModel: CleanupLLM {
    public let modelLabel = "FullStop · punteggiatura"
    private let lock = NSLock()
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var buffered = Data()
    public var timeoutSeconds: Double = 12
    public init() {}

    public func warmUp() {
        lock.lock(); defer { lock.unlock() }
        do { try start(until: Date().addingTimeInterval(timeoutSeconds)) }
        catch { stop() }
    }

    private func response(until deadline: Date) throws -> [String: Any] {
        guard let output else { throw Failure.unavailable }
        while Date() < deadline {
            if let end = buffered.firstIndex(of: 10) {
                let line = buffered.prefix(upTo: end)
                buffered.removeSubrange(...end)
                guard let object = try JSONSerialization.jsonObject(with: line) as? [String: Any] else { throw Failure.protocolError }
                return object
            }
            var descriptor = pollfd(fd: output.fileDescriptor, events: Int16(POLLIN), revents: 0)
            let status = Darwin.poll(&descriptor, 1, 100)
            if status < 0 { if errno == EINTR { continue }; throw Failure.unavailable }
            if status > 0 {
                let chunk = output.availableData
                guard !chunk.isEmpty else { throw Failure.unavailable }
                buffered.append(chunk)
                if buffered.count > 128_000 { throw Failure.protocolError }
            }
        }
        throw Failure.timeout
    }
    private enum Failure: Error { case unavailable, protocolError, timeout }
    private func stop() {
        if let process, process.isRunning {
            process.terminate()
            let deadline = Date().addingTimeInterval(0.2)
            while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.01) }
            if process.isRunning { Darwin.kill(process.processIdentifier, SIGKILL) }
        }
        try? input?.close(); try? output?.close()
        process = nil; input = nil; output = nil; buffered.removeAll()
    }
    private func start(until deadline: Date) throws {
        if process?.isRunning == true { return }
        stop()
        let env = ProcessInfo.processInfo.environment
        let lab = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop/LLM-Lab")
        let python = env["FLOW_PUNCTUATION_PYTHON"] ?? lab.appendingPathComponent("engines/hybrid-s1/bin/python").path
        let script = env["FLOW_PUNCTUATION_WORKER"] ?? Bundle.main.url(forResource: "punctuation_worker", withExtension: "py")?.path
        let model = env["FLOW_PUNCTUATION_MODEL"] ?? lab.appendingPathComponent("models/flow-fullstop-base").path
        guard let script, FileManager.default.isExecutableFile(atPath: python), FileManager.default.fileExists(atPath: script) else { throw Failure.unavailable }
        let child = Process(), incoming = Pipe(), outgoing = Pipe()
        child.executableURL = URL(fileURLWithPath: python)
        child.arguments = ["-u", script, model]
        child.standardInput = incoming; child.standardOutput = outgoing
        child.standardError = FileHandle.nullDevice
        try child.run()
        process = child; input = incoming.fileHandleForWriting; output = outgoing.fileHandleForReading
        guard try response(until: deadline)["ready"] as? Bool == true else { throw Failure.protocolError }
    }
    public func generate(messages: [(role: String, content: String)], maxTokens: Int) -> String? {
        guard let text = messages.last(where: { $0.role == "user" })?.content, text.utf8.count <= 16_000 else { return nil }
        lock.lock(); defer { lock.unlock() }
        do {
            let deadline = Date().addingTimeInterval(timeoutSeconds)
            try start(until: deadline)
            var data = try JSONSerialization.data(withJSONObject: ["text": text]); data.append(10)
            try input?.write(contentsOf: data)
            guard let result = try response(until: deadline)["text"] as? String else { throw Failure.protocolError }
            return result
        } catch {
            stop()
            flowLog("[punctuation] unavailable or timeout; preserving transcript")
            return nil
        }
    }
    public func unload() { lock.lock(); defer { lock.unlock() }; stop() }
    deinit { stop() }
}
