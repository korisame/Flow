import AppKit
import FlowCore
import Darwin

// A second process must never steal the running application's IPC socket.
try FileManager.default.createDirectory(at: FlowConfig.flowDir, withIntermediateDirectories: true)
let lockPath = FlowConfig.flowDir.appendingPathComponent("instance.lock").path
let instanceFD = open(lockPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
guard instanceFD >= 0, flock(instanceFD, LOCK_EX | LOCK_NB) == 0 else { exit(1) }

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
