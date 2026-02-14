#if os(iOS)
import SwiftUI
import ReplayKit

/// The bundle identifier of the Broadcast Upload Extension (derived from main app bundle ID).
enum BroadcastConstants {
    static let extensionBundleID: String = {
        let mainBundleID = Bundle.main.bundleIdentifier ?? "com.voiceping.transcribe"
        return "\(mainBundleID).broadcast"
    }()
}

/// UIViewRepresentable that wraps RPSystemBroadcastPickerView.
/// The picker is rendered at minimal size and near-zero opacity.
/// A coordinator holds a reference so it can be triggered programmatically.
struct BroadcastPickerView: UIViewRepresentable {
    /// Binding so the SwiftUI button can trigger the picker programmatically.
    @Binding var triggerBroadcast: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> UIView {
        let picker = RPSystemBroadcastPickerView(frame: CGRect(x: 0, y: 0, width: 44, height: 44))
        picker.preferredExtension = BroadcastConstants.extensionBundleID
        picker.showsMicrophoneButton = false
        picker.alpha = 0.015

        context.coordinator.picker = picker

        NSLog("[BroadcastPickerView] Created picker with preferredExtension=%@, subviews=%@",
              BroadcastConstants.extensionBundleID,
              picker.subviews.map { String(describing: type(of: $0)) }.joined(separator: ", "))

        return picker
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        if triggerBroadcast {
            DispatchQueue.main.async {
                triggerBroadcast = false
                context.coordinator.triggerBroadcastPicker()
            }
        }
    }

    class Coordinator {
        weak var picker: RPSystemBroadcastPickerView?

        func triggerBroadcastPicker() {
            guard let picker else {
                NSLog("[BroadcastPickerView] Coordinator: picker is nil!")
                return
            }
            NSLog("[BroadcastPickerView] Triggering picker — %d subviews", picker.subviews.count)

            for subview in picker.subviews {
                NSLog("[BroadcastPickerView] Subview type: %@", String(describing: type(of: subview)))
                if let button = subview as? UIButton {
                    NSLog("[BroadcastPickerView] Found UIButton — sending touchUpInside")
                    button.sendActions(for: .touchUpInside)
                    return
                }
            }
            NSLog("[BroadcastPickerView] WARNING: No UIButton found in picker subviews!")
        }
    }
}

/// A styled SwiftUI button that triggers the system broadcast picker.
struct BroadcastStartButton: View {
    @State private var triggerBroadcast = false

    var body: some View {
        ZStack {
            // Hidden picker — must be in view hierarchy for the system sheet to work
            BroadcastPickerView(triggerBroadcast: $triggerBroadcast)
                .frame(width: 1, height: 1)
                .opacity(0.01)
                .allowsHitTesting(false)

            Button {
                NSLog("[BroadcastStartButton] Tapped — requesting broadcast trigger")
                triggerBroadcast = true
            } label: {
                Label("Start System Broadcast", systemImage: "record.circle")
                    .font(.headline)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
                    .background(.red, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .accessibilityIdentifier("broadcast_picker")
    }
}
#endif
