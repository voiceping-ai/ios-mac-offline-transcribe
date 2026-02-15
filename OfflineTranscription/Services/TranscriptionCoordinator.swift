import Foundation

/// Coordinates real-time transcription: inference loop, VAD, chunking, and text assembly.
///
/// Extracted from WhisperService to isolate inference loop state and logic.
/// Owns buffer tracking, silence detection, and chunk management internally.
/// Delegates observable state updates back to WhisperService via internal mutation methods.
///
/// Supports both offline (batch) and streaming transcription loops.
@MainActor
final class TranscriptionCoordinator {

    // MARK: - Internal State

    private(set) var transcriptionTask: Task<Void, Never>?
    private var lingeringTranscriptionTask: Task<Void, Never>?
    private var lastBufferSize: Int = 0
    private var lastConfirmedSegmentEndSeconds: Float = 0
    private var prevUnconfirmedSegments: [ASRSegment] = []
    private var consecutiveSilenceCount: Int = 0
    private var hasCompletedFirstInference: Bool = false
    private var movingAverageInferenceSeconds: Double = 0.0
    private(set) var completedChunksText: String = ""
    private var lastUIMeterUpdateTimestamp: CFAbsoluteTime = 0

    // MARK: - Constants

    private static let sampleRate: Float = AudioConstants.sampleRateFloat
    private static let displayEnergyFrameLimit = 160
    private static let uiMeterUpdateInterval: CFTimeInterval = 0.12
    private static let inlineWhitespaceRegex: NSRegularExpression = {
        return try! NSRegularExpression(pattern: "[^\\S\\n]+")
    }()

    /// Maximum audio chunk duration (seconds). Each chunk is transcribed independently;
    /// when the buffer exceeds this, the current hypothesis is confirmed and a new chunk begins.
    /// WhisperKit: 15s (multi-segment, eager mode confirms progressively).
    /// sherpa-onnx offline: 3.5s (single-segment, matches Android chunk cadence for
    /// faster updates -- each inference processes a small slice, keeping latency low).
    private static let defaultMaxChunkSeconds: Float = 15.0
    private static let sherpaOfflineMaxChunkSeconds: Float = 3.5
    private static let omnilingualOfflineMaxChunkSeconds: Float = 4.0

    // MARK: - Adaptive Delay (CPU-aware, matches Android)
    /// Initial inference gate: show first words quickly (matches Android's 0.35s).
    private static let initialMinNewAudioSeconds: Float = 0.35
    /// Omnilingual is substantially heavier than SenseVoice/Moonshine; use slower initial gate.
    private static let omnilingualInitialMinNewAudioSeconds: Float = 3.0
    /// Base delay between inferences for sherpa-onnx offline after first decode.
    private static let sherpaBaseDelaySeconds: Float = 0.7
    /// Heavier omnilingual base delay to avoid UI starvation.
    private static let omnilingualBaseDelaySeconds: Float = 3.0
    /// Target inference duty cycle -- inference should use at most this fraction of wall time.
    private static let targetInferenceDutyCycle: Float = 0.24
    /// Maximum CPU-protection delay cap.
    private static let maxCpuProtectDelaySeconds: Float = 1.6
    /// EMA smoothing factor for inference time tracking.
    private static let inferenceEmaAlpha: Double = 0.20
    /// Minimum RMS energy to submit audio for inference.
    private static let minInferenceRMS: Float = 0.012
    /// Bypass VAD for the first N seconds so initial speech is never dropped.
    private static let initialVADBypassSeconds: Float = 1.0
    /// Keep a pre-roll of audio when VAD says silence, so utterance onsets
    /// that straddle VAD boundaries are not lost.
    private static let vadPrerollSeconds: Float = 0.6

    // MARK: - Service Reference

    private unowned let service: WhisperService

    init(service: WhisperService) {
        self.service = service
    }

    // MARK: - Model-Dependent Configuration

    private var maxChunkSeconds: Float {
        guard service.selectedModel.engineType == .sherpaOnnxOffline else {
            return Self.defaultMaxChunkSeconds
        }
        return isOmnilingualModel
            ? Self.omnilingualOfflineMaxChunkSeconds
            : Self.sherpaOfflineMaxChunkSeconds
    }

    private var isOmnilingualModel: Bool {
        if service.selectedModel.sherpaModelConfig?.modelType == .omnilingualCtc {
            return true
        }
        return service.selectedModel.id.lowercased().contains("omnilingual")
    }

    // MARK: - Task Management

    func cancelAndTrackTranscriptionTask() {
        guard let task = transcriptionTask else { return }
        task.cancel()
        lingeringTranscriptionTask = task
        transcriptionTask = nil
    }

    func drainLingeringTranscriptionTask() async {
        if let activeTask = transcriptionTask {
            activeTask.cancel()
            lingeringTranscriptionTask = activeTask
            transcriptionTask = nil
        }
        if let lingering = lingeringTranscriptionTask {
            _ = await lingering.result
            lingeringTranscriptionTask = nil
        }
    }

    // MARK: - Real-time Loop

    func startLoop() {
        cancelAndTrackTranscriptionTask()
        guard let engine = service.activeEngine else { return }

        if engine.isStreaming {
            transcriptionTask = Task {
                await streamingLoop(engine: engine)
            }
        } else {
            transcriptionTask = Task {
                await offlineLoop(engine: engine)
            }
        }
    }

    private func offlineLoop(engine: ASREngine) async {
        while service.isRecording && service.isTranscribing && !Task.isCancelled {
            do {
                try await transcribeCurrentBuffer(engine: engine)
            } catch {
                if !Task.isCancelled {
                    service.updateLastError(.transcriptionFailed(underlying: error))
                }
                break
            }
        }

        if !Task.isCancelled {
            service.endTranscriptionLoop()
        }
    }

    private func streamingLoop(engine: ASREngine) async {
        while service.isRecording && service.isTranscribing && !Task.isCancelled {
            try? await Task.sleep(for: .milliseconds(100))

            refreshRealtimeMeters(engine: engine)

            // Poll streaming result
            if let result = engine.getStreamingResult() {
                let nextHypothesis = normalizedJoinedText(from: result.segments)
                if service.unconfirmedSegments != result.segments {
                    service.updateUnconfirmedSegments(result.segments)
                }
                if service.hypothesisText != nextHypothesis {
                    service.updateTranscriptionText(
                        confirmed: service.confirmedText,
                        hypothesis: nextHypothesis
                    )
                    service.scheduleTranslationUpdate()
                }

                // Endpoint detection -> finalize utterance as a new chunk
                if engine.isEndpointDetected() {
                    finalizeCurrentChunk()
                    engine.resetStreamingState()
                }
            }
        }

        if !Task.isCancelled {
            // Capture final result before stopping
            if let result = engine.getStreamingResult(),
               !normalizedJoinedText(from: result.segments).isEmpty {
                service.updateUnconfirmedSegments(result.segments)
                finalizeCurrentChunk()
            }

            service.endTranscriptionLoop()
        }
    }

    private func transcribeCurrentBuffer(engine: ASREngine) async throws {
        let logger = InferenceLogger.shared
        let currentBuffer = service.effectiveAudioSamples
        let nextBufferSize = currentBuffer.count - lastBufferSize
        let nextBufferSeconds = Float(nextBufferSize) / Self.sampleRate
        refreshRealtimeMeters(engine: engine)

        let effectiveDelay = adaptiveDelay()
        guard nextBufferSeconds > Float(effectiveDelay) else {
            try await Task.sleep(for: .milliseconds(100))
            return
        }

        if service.useVAD && service.audioCaptureMode == .microphone {
            // Bypass VAD for the first second so initial speech is never dropped
            // Skip VAD entirely for device audio mode -- speaker output has continuous energy
            let vadBypassSamples = Int(Self.sampleRate * Self.initialVADBypassSeconds)
            let bypassVadDuringStartup = !hasCompletedFirstInference && currentBuffer.count <= vadBypassSamples
            if !bypassVadDuringStartup {
                let voiceDetected = isVoiceDetected(
                    in: service.effectiveRelativeEnergy,
                    nextBufferInSeconds: nextBufferSeconds
                )
                if !voiceDetected {
                    consecutiveSilenceCount += 1
                    // Keep a pre-roll so utterance onsets straddling VAD are preserved
                    let prerollSamples = Int(Self.sampleRate * Self.vadPrerollSeconds)
                    lastBufferSize = max(currentBuffer.count - prerollSamples, 0)
                    if consecutiveSilenceCount == 1 || consecutiveSilenceCount % 10 == 0 {
                        logger.log("VAD silence #\(consecutiveSilenceCount) totalBuffer=\(currentBuffer.count) (\(String(format: "%.1f", Float(currentBuffer.count) / Self.sampleRate))s)")
                    }
                    return
                }
                consecutiveSilenceCount = 0
            }
        }

        // Chunk-based windowing: process audio in fixed-size chunks to prevent
        // models from receiving unbounded audio. When the buffer grows past the
        // current chunk boundary, finalize the hypothesis and start a new chunk.
        let bufferEndSeconds = Float(currentBuffer.count) / Self.sampleRate
        var chunkEndSeconds = lastConfirmedSegmentEndSeconds + maxChunkSeconds

        if bufferEndSeconds > chunkEndSeconds {
            finalizeCurrentChunk()
            lastConfirmedSegmentEndSeconds = chunkEndSeconds
            // Recompute for the new chunk so we don't produce an empty slice
            chunkEndSeconds = lastConfirmedSegmentEndSeconds + maxChunkSeconds
        }

        // Slice audio for the current chunk window
        let sliceStartSeconds = lastConfirmedSegmentEndSeconds
        let sliceStartSample = min(Int(sliceStartSeconds * Self.sampleRate), currentBuffer.count)
        let sliceEndSample = min(Int(chunkEndSeconds * Self.sampleRate), currentBuffer.count)
        let audioSamples = Array(currentBuffer[sliceStartSample..<sliceEndSample])
        guard !audioSamples.isEmpty else { return }

        // RMS energy gate: skip inference on near-silence audio to avoid
        // SenseVoice hallucinations ("I.", "Yeah.", "The.") and save CPU.
        // NOTE: lastBufferSize is NOT updated on skip -- this ensures that when
        // speech resumes after silence, nextBufferSeconds is already large enough
        // to pass the delay guard immediately, giving near-instant response.
        let sliceRMS = sqrt(audioSamples.reduce(Float(0)) { $0 + $1 * $1 } / Float(audioSamples.count))
        if sliceRMS < Self.minInferenceRMS {
            logger.log("SKIP low-energy slice rms=\(String(format: "%.4f", sliceRMS)) < \(Self.minInferenceRMS)")
            try await Task.sleep(for: .milliseconds(500))
            return
        }

        lastBufferSize = currentBuffer.count

        let options = ASRTranscriptionOptions(
            withTimestamps: service.enableTimestamps,
            temperature: 0.0
        )

        let sliceDurationSeconds = Float(audioSamples.count) / Self.sampleRate
        logger.log("BUFFER SUBMIT model=\(service.selectedModel.id) sliceStart=\(String(format: "%.2f", sliceStartSeconds))s sliceEnd=\(String(format: "%.2f", Float(sliceEndSample) / Self.sampleRate))s sliceSamples=\(audioSamples.count) sliceDuration=\(String(format: "%.2f", sliceDurationSeconds))s rms=\(String(format: "%.4f", sliceRMS)) totalBuffer=\(currentBuffer.count) (\(String(format: "%.1f", Float(currentBuffer.count) / Self.sampleRate))s)")
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await engine.transcribe(audioArray: audioSamples, options: options)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        guard !Task.isCancelled else { return }

        let wordCount = result.text.split(separator: " ").count
        if elapsed > 0 && wordCount > 0 {
            service.updateTokensPerSecond(Double(wordCount) / elapsed)
        }

        // Track inference time with EMA for CPU-aware delay
        if movingAverageInferenceSeconds <= 0 {
            movingAverageInferenceSeconds = elapsed
        } else {
            movingAverageInferenceSeconds = Self.inferenceEmaAlpha * elapsed
                + (1.0 - Self.inferenceEmaAlpha) * movingAverageInferenceSeconds
        }

        NSLog("[TranscriptionCoordinator] chunk inference: %.1fs audio in %.2fs (ratio %.1fx, %d words, emaInf=%.3fs, delay=%.2fs)",
              sliceDurationSeconds, elapsed, Double(sliceDurationSeconds) / elapsed, wordCount,
              movingAverageInferenceSeconds, adaptiveDelay())
        logger.log("BUFFER RESULT model=\(service.selectedModel.id) elapsed=\(String(format: "%.3f", elapsed))s rtf=\(String(format: "%.2f", elapsed / Double(sliceDurationSeconds))) words=\(wordCount) segments=\(result.segments.count) emaInf=\(String(format: "%.3f", movingAverageInferenceSeconds))s delay=\(String(format: "%.2f", adaptiveDelay()))s text=\"\(String(result.text.prefix(200)))\"")

        hasCompletedFirstInference = true
        processTranscriptionResult(result, sliceOffset: sliceStartSeconds)
    }

    // MARK: - VAD & Delay

    private func isVoiceDetected(in energy: [Float], nextBufferInSeconds: Float) -> Bool {
        guard !energy.isEmpty else { return false }
        let recentEnergy = energy.suffix(10)
        let peakEnergy = recentEnergy.max() ?? 0
        let avgEnergy = recentEnergy.reduce(0, +) / Float(recentEnergy.count)
        return peakEnergy >= service.silenceThreshold || avgEnergy >= service.silenceThreshold * 0.5
    }

    private func adaptiveDelay() -> Double {
        // During silence, back off to save CPU
        if consecutiveSilenceCount > 5 {
            return min(service.realtimeDelayInterval * 3.0, 3.0)
        } else if consecutiveSilenceCount > 2 {
            return service.realtimeDelayInterval * 2.0
        }

        // Fast initial gate: show first words quickly (matches Android 0.35s)
        if !hasCompletedFirstInference {
            if service.selectedModel.engineType == .sherpaOnnxOffline && isOmnilingualModel {
                return Double(Self.omnilingualInitialMinNewAudioSeconds)
            }
            return Double(Self.initialMinNewAudioSeconds)
        }

        // For sherpa-onnx offline: CPU-aware delay (matches Android architecture)
        if service.selectedModel.engineType == .sherpaOnnxOffline {
            let baseDelay = isOmnilingualModel
                ? Double(Self.omnilingualBaseDelaySeconds)
                : Double(Self.sherpaBaseDelaySeconds)
            return computeCpuAwareDelay(baseDelay: baseDelay)
        }

        return service.realtimeDelayInterval
    }

    /// Compute delay based on actual inference time to maintain a target CPU duty cycle.
    /// If inference takes 0.17s and target duty is 24%, delay = 0.17/0.24 = 0.71s.
    /// This adapts automatically to device speed -- fast devices get shorter delays.
    private func computeCpuAwareDelay(baseDelay: Double) -> Double {
        let avg = movingAverageInferenceSeconds
        guard avg > 0 else { return baseDelay }
        let budgetDelay = avg / Double(Self.targetInferenceDutyCycle)
        return max(baseDelay, min(budgetDelay, Double(Self.maxCpuProtectDelaySeconds)))
    }

    // MARK: - Meters & Results

    /// Update render-facing meters at a fixed cadence with bounded payload size.
    /// This keeps live UI smooth while preventing large array churn on every loop.
    private func refreshRealtimeMeters(engine: ASREngine, force: Bool = false) {
        let now = CFAbsoluteTimeGetCurrent()
        if !force, now - lastUIMeterUpdateTimestamp < Self.uiMeterUpdateInterval { return }
        lastUIMeterUpdateTimestamp = now

        let sampleCount = service.effectiveAudioSamples.count
        let nextBufferSeconds = Double(sampleCount) / Double(Self.sampleRate)
        let nextEnergy = Array(service.effectiveRelativeEnergy.suffix(Self.displayEnergyFrameLimit))
        service.updateMeters(energy: nextEnergy, bufferSeconds: nextBufferSeconds)
    }

    func processTranscriptionResult(_ result: ASRResult, sliceOffset: Float = 0) {
        let newSegments = result.segments

        // Eager mode only works for multi-segment models (WhisperKit).
        // Single-segment models (SenseVoice, Moonshine) always return 1 segment
        // whose text changes every cycle, so segment comparison never confirms.
        let useEager = service.enableEagerMode && service.selectedModel.engineType != .sherpaOnnxOffline
        if useEager, !prevUnconfirmedSegments.isEmpty {
            var matchCount = 0
            for (prevSeg, newSeg) in zip(prevUnconfirmedSegments, newSegments) {
                if normalizeDisplayText(prevSeg.text)
                    == normalizeDisplayText(newSeg.text)
                {
                    matchCount += 1
                } else {
                    break
                }
            }

            if matchCount > 0 {
                let newlyConfirmed = Array(newSegments.prefix(matchCount))
                let updatedConfirmed = service.confirmedSegments + newlyConfirmed

                if let lastConfirmed = newlyConfirmed.last {
                    lastConfirmedSegmentEndSeconds = sliceOffset + lastConfirmed.end
                }

                let updatedUnconfirmed = Array(newSegments.dropFirst(matchCount))
                service.updateSegments(confirmed: updatedConfirmed, unconfirmed: updatedUnconfirmed)
            } else {
                service.updateUnconfirmedSegments(newSegments)
            }
        } else {
            service.updateUnconfirmedSegments(newSegments)
        }

        prevUnconfirmedSegments = service.unconfirmedSegments

        // Build confirmed text: completed chunks + within-chunk confirmed segments
        let withinChunkConfirmed = normalizedJoinedText(from: service.confirmedSegments)
        let nextConfirmedText = [completedChunksText, withinChunkConfirmed]
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
        let nextHypothesisText = normalizedJoinedText(from: service.unconfirmedSegments)

        let changed = service.confirmedText != nextConfirmedText || service.hypothesisText != nextHypothesisText
        service.updateTranscriptionText(confirmed: nextConfirmedText, hypothesis: nextHypothesisText)
        if changed {
            service.scheduleTranslationUpdate()
        }
    }

    // MARK: - Chunk Management

    func finalizeCurrentChunk() {
        let allSegments = service.confirmedSegments + service.unconfirmedSegments
        let chunkText = normalizedJoinedText(from: allSegments)
        if !chunkText.isEmpty {
            if completedChunksText.isEmpty {
                completedChunksText = chunkText
            } else {
                completedChunksText += "\n" + chunkText
            }
        }
        service.updateSegments(confirmed: [], unconfirmed: [])
        prevUnconfirmedSegments = []
        let nextConfirmedText = completedChunksText
        let changed = service.confirmedText != nextConfirmedText || !service.hypothesisText.isEmpty
        service.updateTranscriptionText(confirmed: nextConfirmedText, hypothesis: "")
        if changed {
            service.scheduleTranslationUpdate()
        }
    }

    // MARK: - Text Normalization

    func normalizedJoinedText(from segments: [ASRSegment]) -> String {
        segments.lazy
            .map { self.normalizeDisplayText($0.text) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    func normalizeDisplayText(_ text: String) -> String {
        text
            .components(separatedBy: "\n")
            .map { line in
                collapseInlineWhitespace(in: line)
                    .trimmingCharacters(in: .whitespaces)
            }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func collapseInlineWhitespace(in line: String) -> String {
        let range = NSRange(line.startIndex..<line.endIndex, in: line)
        return Self.inlineWhitespaceRegex.stringByReplacingMatches(
            in: line, options: [], range: range, withTemplate: " "
        )
    }

    // MARK: - Full Text Assembly

    func assembleFullText(confirmedSegments: [ASRSegment], unconfirmedSegments: [ASRSegment]) -> String {
        let currentChunkConfirmed = normalizedJoinedText(from: confirmedSegments)
        let currentChunkHypothesis = normalizedJoinedText(from: unconfirmedSegments)
        let currentChunk = [currentChunkConfirmed, currentChunkHypothesis]
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        let parts = [completedChunksText, currentChunk].filter { !$0.isEmpty }
        return parts.joined(separator: "\n")
    }

    // MARK: - State Reset

    func reset() {
        cancelAndTrackTranscriptionTask()
        lastBufferSize = 0
        lastConfirmedSegmentEndSeconds = 0
        prevUnconfirmedSegments = []
        consecutiveSilenceCount = 0
        hasCompletedFirstInference = false
        movingAverageInferenceSeconds = 0.0
        completedChunksText = ""
        lastUIMeterUpdateTimestamp = 0
    }
}
