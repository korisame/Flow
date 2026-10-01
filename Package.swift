// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FlowSwift",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.17.3"),
    ],
    targets: [
        .target(
            name: "FlowCore",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "FlowCLI",
            dependencies: ["FlowCore"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "FlowApp",
            dependencies: ["FlowCore"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "FlowCoreTests",
            dependencies: ["FlowCore"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
