// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "QwenASRKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "QwenASRKit", targets: ["QwenASRKit"]),
    ],
    targets: [
        .target(
            name: "QwenASRCLib",
            path: "Sources/QwenASRCLib",
            publicHeadersPath: "include",
            cSettings: [
                .define("USE_BLAS"),
                .define("ACCELERATE_NEW_LAPACK"),
                .headerSearchPath("ort"),
                .unsafeFlags(["-O3", "-ffast-math"]),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .target(
            name: "QwenASRKit",
            dependencies: ["QwenASRCLib"],
            path: "Sources/QwenASRKit",
            exclude: ["QwenOnnxASR.swift"]
        ),
    ]
)
