// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "MLXKit", targets: ["MLXKit"])
    ],
    targets: [
        .target(
            name: "MLXKit",
            path: "Sources/MLXKit"
        )
    ]
)
