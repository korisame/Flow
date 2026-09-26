import Foundation
import FlowCore

let arguments = CommandLine.arguments
if arguments.count >= 3 && arguments[1] == "cleanup" {
    let model = LocalCleanupModel()
    model.timeoutSeconds = 30
    let start = Date()
    guard let result = model.generate(messages: [("user", arguments[2])], maxTokens: 0) else { exit(2) }
    let data = try JSONSerialization.data(withJSONObject: ["text": result, "seconds": -start.timeIntervalSinceNow], options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
    model.unload()
} else if arguments.count >= 3 && arguments[1] == "transcribe" {
    let transcriber = Transcriber()
    let result = try await transcriber.transcribe(WavIO.loadAsFlowSamples(URL(fileURLWithPath: arguments[2])))
    let data = try JSONSerialization.data(withJSONObject: ["text": result.text, "load_s": result.loadS, "asr_s": result.transcribeS], options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
    await transcriber.unload()
} else if arguments.count >= 3 && arguments[1] == "process" {
    print(processText(arguments[2], verbalCommands: true, removeFillers: true))
} else {
    print("Flow: process <text> | cleanup <text> | transcribe <audio-file>")
}
