#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace omnivoice {

struct GenerationConfig {
    int num_step = 32;
    float guidance_scale = 2.0f;
    float t_shift = 0.1f;
    float layer_penalty_factor = 5.0f;
    float position_temperature = 5.0f;
    float class_temperature = 0.0f;
    bool denoise = true;
    bool preprocess_prompt = true;
    bool postprocess_output = true;
    float audio_chunk_duration = 15.0f;
    float audio_chunk_threshold = 30.0f;
    std::optional<uint64_t> seed;
};

struct SynthesisParams {
    std::string text;
    std::string language;
    std::string instruct;
    bool auto_voice = false;
    float speed = 1.0f;
    std::optional<float> duration;
    std::string ref_audio_path;
    std::string ref_text;
    GenerationConfig generation;
};

struct Audio {
    int sample_rate = 24000;
    std::vector<float> samples;
};

enum class TraceEventKind {
    StageBegin,
    StageEnd,
    LlmProgress,
};

struct TraceEvent {
    TraceEventKind kind = TraceEventKind::StageBegin;
    std::string phase;
    std::string name;
    double seconds = 0.0;
    double audio_seconds = 0.0;
    int chunk_index = 0;
    int chunk_count = 0;
    int current = 0;
    int total = 0;
    int updated = 0;
    int total_positions = 0;
};

struct RuntimeOptions {
    std::string backend = "cpu";
    int device = 0;
    int threads = 4;
    std::function<void(const TraceEvent &)> trace;
};

class OmniVoiceRuntime {
public:
    OmniVoiceRuntime(const std::string & model_path, const RuntimeOptions & options);
    ~OmniVoiceRuntime();

    OmniVoiceRuntime(const OmniVoiceRuntime &) = delete;
    OmniVoiceRuntime & operator=(const OmniVoiceRuntime &) = delete;

    Audio generate(const SynthesisParams & params);
    int sample_rate() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<float> read_wav_mono(const std::string & path, int target_sample_rate, int * source_sample_rate = nullptr);
void write_wav_mono_f32(const std::string & path, const std::vector<float> & samples, int sample_rate);

} // namespace omnivoice
