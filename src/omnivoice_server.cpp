#include "omnivoice.h"

#include "ggml.h"
#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

void quiet_ggml_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::cerr << text;
}

// --- WAV encoding ---------------------------------------------------------

std::string encode_wav(const std::vector<float> & samples, int sample_rate) {
    const size_t n = samples.size();
    const uint16_t num_channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const uint16_t block_align = num_channels * bits_per_sample / 8;
    const uint32_t data_size = static_cast<uint32_t>(n * num_channels * bits_per_sample / 8);
    const uint32_t file_size = 36 + data_size;

    std::string buf;
    buf.reserve(44 + data_size);

    auto write_u8 = [&](uint8_t v) { buf.push_back(static_cast<char>(v)); };
    auto write_u16 = [&](uint16_t v) { write_u8(v & 0xFF); write_u8((v >> 8) & 0xFF); };
    auto write_u32 = [&](uint32_t v) { write_u8(v & 0xFF); write_u8((v >> 8) & 0xFF); write_u8((v >> 16) & 0xFF); write_u8((v >> 24) & 0xFF); };

    // RIFF header
    buf.append("RIFF", 4);
    write_u32(file_size);
    buf.append("WAVE", 4);

    // fmt chunk
    buf.append("fmt ", 4);
    write_u32(16); // chunk size
    write_u16(1);  // PCM format
    write_u16(num_channels);
    write_u32(static_cast<uint32_t>(sample_rate));
    write_u32(byte_rate);
    write_u16(block_align);
    write_u16(bits_per_sample);

    // data chunk
    buf.append("data", 4);
    write_u32(data_size);

    for (float s : samples) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t pcm = static_cast<int16_t>(clamped * 32767.0f);
        write_u8(static_cast<uint8_t>(pcm & 0xFF));
        write_u8(static_cast<uint8_t>((pcm >> 8) & 0xFF));
    }
    return buf;
}

// --- PCM raw output (24kHz 16-bit signed little-endian) -------------------

std::string encode_pcm(const std::vector<float> & samples) {
    std::string buf;
    buf.reserve(samples.size() * 2);
    for (float s : samples) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t pcm = static_cast<int16_t>(clamped * 32767.0f);
        buf.push_back(static_cast<char>(pcm & 0xFF));
        buf.push_back(static_cast<char>((pcm >> 8) & 0xFF));
    }
    return buf;
}

// --- Helpers --------------------------------------------------------------

std::string format_content_type(const std::string & format) {
    if (format == "mp3")  return "audio/mpeg";
    if (format == "pcm")  return "audio/pcm";
    return "audio/wav";
}

std::string json_error(const std::string & message, const std::string & type = "invalid_request_error") {
    return json{{"error", {{"message", message}, {"type", type}}}}.dump();
}

void set_json(httplib::Response & res, const json & value, int status = 200) {
    res.status = status;
    res.set_content(value.dump(), "application/json");
}

bool has_field(const json & j, const std::string & key) {
    return j.contains(key) && !j.at(key).is_null();
}

std::string get_opt_string(const json & j, const std::string & key, const std::string & def = "") {
    if (!has_field(j, key) || !j.at(key).is_string()) return def;
    return j.at(key).get<std::string>();
}

std::optional<double> get_opt_number(const json & j, const std::string & key) {
    if (!has_field(j, key)) return std::nullopt;
    if (!j.at(key).is_number()) return std::nullopt;
    return j.at(key).get<double>();
}

std::optional<bool> get_opt_bool(const json & j, const std::string & key) {
    if (!has_field(j, key)) return std::nullopt;
    if (j.at(key).is_boolean()) return j.at(key).get<bool>();
    if (j.at(key).is_number()) return j.at(key).get<double>() != 0.0;
    return std::nullopt;
}

std::string lower_extension(const fs::path & path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

bool is_reference_audio_path(const fs::path & path) {
    const std::string ext = lower_extension(path);
    return ext == ".wav" || ext == ".mp3" || ext == ".flac";
}

bool looks_like_preset_name(const std::string & name) {
    return !name.empty()
        && name.find('/') == std::string::npos
        && name.find('\\') == std::string::npos
        && name != "."
        && name != "..";
}

int parse_int_option(const std::string & value, const std::string & name) {
    size_t parsed = 0;
    int out = 0;
    try {
        out = std::stoi(value, &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid integer for " + name + ": " + value);
    }
    if (parsed != value.size()) throw std::runtime_error("invalid integer for " + name + ": " + value);
    return out;
}

// --- Voice preset system --------------------------------------------------
// Maps voice names to ref_audio/ref_text files on disk.
// Users can place WAV/MP3/FLAC + txt pairs in a voices/ directory.

struct VoicePreset {
    std::string ref_audio_path;
    std::string ref_text;
};

std::string voices_dir;

VoicePreset load_voice_preset(const std::string & name) {
    VoicePreset p;
    if (voices_dir.empty() || !looks_like_preset_name(name)) return p;
    const fs::path base = fs::path(voices_dir) / name;

    std::ifstream tf(base.string() + ".txt");
    if (tf) std::getline(tf, p.ref_text);

    for (const char * ext : {".wav", ".mp3", ".flac"}) {
        const fs::path audio_path = base.string() + ext;
        if (fs::is_regular_file(audio_path)) {
            p.ref_audio_path = audio_path.string();
            break;
        }
    }
    return p;
}

json list_voice_presets() {
    json out = {{"object", "list"}, {"data", json::array()}};
    if (voices_dir.empty()) return out;

    const fs::path dir(voices_dir);
    if (!fs::exists(dir) || !fs::is_directory(dir)) return out;

    for (const auto & entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file() || !is_reference_audio_path(entry.path())) continue;

        const std::string name = entry.path().stem().string();
        VoicePreset preset = load_voice_preset(name);
        out["data"].push_back({
            {"id", name},
            {"object", "voice"},
            {"has_ref_text", !preset.ref_text.empty()},
            {"ref_text", preset.ref_text},
        });
    }
    return out;
}

// --- Server ---------------------------------------------------------------

struct ServerConfig {
    std::string model_path;
    std::string host = "0.0.0.0";
    int port = 8000;
    std::string backend = "cpu";
    int device = 0;
    int threads = 4;
};

void usage() {
    std::cerr
        << "usage: omnivoice-server --model MODEL.gguf [options]\n"
        << "options:\n"
        << "  --host HOST         listen host (default: 0.0.0.0)\n"
        << "  --port PORT         listen port (default: 8000)\n"
        << "  --voices-dir DIR    directory with voice presets (name.wav/mp3/flac + name.txt)\n"
        << "  --device cpu|cuda[:N]\n"
        << "  --backend cpu|cuda\n"
        << "  --threads N         number of CPU threads (default: 4)\n";
}

} // namespace

int main(int argc, char ** argv) {
    ggml_log_set(quiet_ggml_log, nullptr);

    try {
    ServerConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need = [&](const char * name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (key == "--model") cfg.model_path = need("--model");
        else if (key == "--host") cfg.host = need("--host");
        else if (key == "--port") cfg.port = parse_int_option(need("--port"), "--port");
        else if (key == "--voices-dir") voices_dir = need("--voices-dir");
        else if (key == "--backend") {
            cfg.backend = need("--backend");
            if (cfg.backend == "rocm" || cfg.backend == "hip") cfg.backend = "cuda";
        } else if (key == "--device") {
            std::string dev = need("--device");
            if (dev == "cpu") cfg.backend = "cpu";
            else if (dev.rfind("cuda", 0) == 0) {
                cfg.backend = "cuda";
                auto pos = dev.find(':');
                if (pos != std::string::npos) cfg.device = parse_int_option(dev.substr(pos + 1), "--device");
            } else {
                throw std::runtime_error("unsupported --device: " + dev);
            }
        } else if (key == "--threads" || key == "-t") cfg.threads = parse_int_option(need("--threads"), "--threads");
        else if (key == "--help" || key == "-h") { usage(); return 0; }
        else { std::cerr << "unknown argument: " << key << "\n"; usage(); return 2; }
    }

    if (cfg.model_path.empty()) {
        std::cerr << "error: --model is required\n";
        usage();
        return 2;
    }

    // Load model once — stays in VRAM/RAM for the lifetime of the server
    std::cerr << "[init] loading model from " << cfg.model_path << " ...\n";
    omnivoice::RuntimeOptions options;
    options.backend = cfg.backend;
    options.device = cfg.device;
    options.threads = cfg.threads;

    omnivoice::OmniVoiceRuntime runtime(cfg.model_path, options);
    std::cerr << "[init] model loaded (backend=" << cfg.backend
              << " sample_rate=" << runtime.sample_rate() << ")\n";

    // Mutex to serialize generate() calls (GPU inference is not thread-safe)
    std::mutex gen_mutex;

    // --- HTTP server ---
    httplib::Server svr;

    // Health check
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        set_json(res, {{"status", "ok"}});
    });

    // GET /v1/models — list available models
    svr.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
        set_json(res, {
            {"object", "list"},
            {"data", json::array({
                {{"id", "omnivoice"}, {"object", "model"}, {"owned_by", "local"}},
            })},
        });
    });

    // GET /v1/audio/voices — list available voice presets
    svr.Get("/v1/audio/voices", [&](const httplib::Request &, httplib::Response & res) {
        try {
            set_json(res, list_voice_presets());
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(json_error(std::string("failed to list voices: ") + e.what()), "application/json");
        }
    });

    // POST /v1/audio/speech — the main endpoint
    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        // Parse JSON body
        if (req.body.empty()) {
            res.status = 400;
            res.set_content(json_error("request body is empty"), "application/json");
            return;
        }

        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error & e) {
            res.status = 400;
            res.set_content(json_error(std::string("invalid JSON: ") + e.what()), "application/json");
            return;
        }
        if (!body.is_object()) {
            res.status = 400;
            res.set_content(json_error("request body must be a JSON object"), "application/json");
            return;
        }

        // --- Required parameters ---
        std::string input_text = get_opt_string(body, "input");
        if (input_text.empty()) {
            res.status = 400;
            res.set_content(json_error("'input' is required and must be a non-empty string"), "application/json");
            return;
        }

        std::string voice = get_opt_string(body, "voice");
        std::string instructions = get_opt_string(body, "instructions");
        if (instructions.empty()) instructions = get_opt_string(body, "instruct");

        // --- Response format ---
        std::string response_format = get_opt_string(body, "response_format", "wav");
        if (response_format != "wav" && response_format != "pcm" && response_format != "mp3") {
            res.status = 400;
            res.set_content(json_error("unsupported response_format '" + response_format + "'; supported formats are 'wav', 'pcm', and 'mp3'"), "application/json");
            return;
        }

        // --- Speed ---
        float speed = static_cast<float>(get_opt_number(body, "speed").value_or(1.0));
        if (speed < 0.25f) speed = 0.25f;
        if (speed > 4.0f) speed = 4.0f;

        // --- Build SynthesisParams ---
        omnivoice::SynthesisParams params;
        params.text = input_text;
        params.speed = speed;

        // Language
        if (has_field(body, "language")) {
            params.language = get_opt_string(body, "language");
        }
        if (auto auto_voice = get_opt_bool(body, "auto_voice")) {
            params.auto_voice = *auto_voice;
        }

        // Voice resolution:
        // 1. If "voice" is a known preset name, load ref_audio + ref_text.
        // 2. Otherwise treat "voice" as an instruct string for voice design.
        // 3. Explicit ref_audio/ref_text overrides preset audio.
        // 4. Explicit instructions/instruct overrides or augments voice design.
        bool voice_is_instruct = false;

        if (!voice.empty()) {
            // Check if it's a preset
            VoicePreset preset = load_voice_preset(voice);
            if (!preset.ref_audio_path.empty()) {
                params.ref_audio_path = preset.ref_audio_path;
                params.ref_text = preset.ref_text;
            } else {
                // Treat voice as instruct string (voice design attributes)
                params.instruct = voice;
                voice_is_instruct = true;
            }
        }

        // Explicit ref_audio / ref_text override (OmniVoice extensions)
        if (has_field(body, "ref_audio")) {
            // ref_audio can be a WAV file path or a voice preset name.
            std::string ref = get_opt_string(body, "ref_audio");
            if (!ref.empty()) {
                // Check if it's a file path
                if (fs::is_regular_file(ref)) {
                    if (!is_reference_audio_path(ref)) {
                        res.status = 400;
                        res.set_content(json_error("'ref_audio' must point to a WAV, MP3, or FLAC file"), "application/json");
                        return;
                    }
                    params.ref_audio_path = ref;
                } else {
                    // Try as preset name
                    VoicePreset p = load_voice_preset(ref);
                    if (!p.ref_audio_path.empty()) {
                        params.ref_audio_path = p.ref_audio_path;
                        if (params.ref_text.empty()) params.ref_text = p.ref_text;
                    } else {
                        res.status = 400;
                        res.set_content(json_error("'ref_audio' is neither a readable WAV path nor a configured voice preset"), "application/json");
                        return;
                    }
                }
            }
        }
        if (has_field(body, "ref_text")) {
            params.ref_text = get_opt_string(body, "ref_text");
        }

        if (!params.ref_audio_path.empty() && params.ref_text.empty()) {
            res.status = 400;
            res.set_content(json_error("'ref_text' is required when using reference audio"), "application/json");
            return;
        }

        // Instructions override voice if explicitly provided
        if (!instructions.empty() && voice_is_instruct) {
            // Both voice and instructions — merge them
            params.instruct = voice + ", " + instructions;
        } else if (!instructions.empty()) {
            params.instruct = instructions;
        }

        // --- GenerationConfig (OmniVoice extensions) ---
        if (auto v = get_opt_number(body, "num_step"))
            params.generation.num_step = static_cast<int>(*v);

        if (auto v = get_opt_number(body, "guidance_scale"))
            params.generation.guidance_scale = static_cast<float>(*v);

        if (auto v = get_opt_number(body, "t_shift"))
            params.generation.t_shift = static_cast<float>(*v);

        if (auto v = get_opt_number(body, "layer_penalty_factor"))
            params.generation.layer_penalty_factor = static_cast<float>(*v);

        if (auto v = get_opt_number(body, "position_temperature"))
            params.generation.position_temperature = static_cast<float>(*v);

        if (auto v = get_opt_number(body, "class_temperature"))
            params.generation.class_temperature = static_cast<float>(*v);

        if (auto v = get_opt_bool(body, "denoise"))
            params.generation.denoise = *v;

        if (auto v = get_opt_bool(body, "preprocess_prompt"))
            params.generation.preprocess_prompt = *v;

        if (auto v = get_opt_bool(body, "postprocess_output"))
            params.generation.postprocess_output = *v;

        if (auto v = get_opt_number(body, "duration"))
            params.duration = static_cast<float>(*v);

        if (auto v = get_opt_number(body, "audio_chunk_duration"))
            params.generation.audio_chunk_duration = static_cast<float>(*v);

        if (auto v = get_opt_number(body, "audio_chunk_threshold"))
            params.generation.audio_chunk_threshold = static_cast<float>(*v);

        if (auto v = get_opt_number(body, "seed"))
            params.generation.seed = static_cast<uint64_t>(*v);

        // --- Generate ---
        std::cerr << "[request] input=\"" << input_text.substr(0, 80)
                  << (input_text.size() > 80 ? "..." : "")
                  << "\" voice=\"" << voice
                  << "\" format=" << response_format
                  << " speed=" << speed << "\n";

        omnivoice::Audio audio;
        const auto gen_start = std::chrono::steady_clock::now();
        try {
            std::lock_guard<std::mutex> lock(gen_mutex);
            audio = runtime.generate(params);
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(json_error(std::string("generation failed: ") + e.what()), "application/json");
            std::cerr << "[error] " << e.what() << "\n";
            return;
        }
        const double gen_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - gen_start).count();

        // --- Encode response ---
        std::string audio_bytes;
        if (response_format == "mp3") {
            audio_bytes = omnivoice::encode_mp3_mono_f32(audio.samples, audio.sample_rate);
        } else if (response_format == "pcm") {
            audio_bytes = encode_pcm(audio.samples);
        } else {
            audio_bytes = encode_wav(audio.samples, audio.sample_rate);
        }

        const double audio_seconds = audio.sample_rate > 0
            ? double(audio.samples.size()) / double(audio.sample_rate)
            : 0.0;
        const double rtf = audio_seconds > 0.0 ? gen_seconds / audio_seconds : 0.0;
        std::cerr << std::fixed << std::setprecision(3)
                  << "[response] gen=" << gen_seconds << "s"
                  << " audio=" << audio_seconds << "s"
                  << " rtf=" << rtf
                  << " size=" << audio_bytes.size() << " (" << response_format << ")\n";

        res.status = 200;
        res.set_content(audio_bytes, format_content_type(response_format));
    });

    // --- Start ---
    std::cerr << "[init] listening on " << cfg.host << ":" << cfg.port << "\n";
    if (!svr.listen(cfg.host, cfg.port)) {
        std::cerr << "error: failed to listen on " << cfg.host << ":" << cfg.port << "\n";
        return 1;
    }
    return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
