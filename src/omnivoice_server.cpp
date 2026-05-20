#include "omnivoice.h"

#include "ggml.h"
#include "httplib.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal JSON parsing without nlohmann — we'll parse manually to avoid adding
// another dependency. If nlohmann/json.hpp is available, use it; otherwise,
// a lightweight inline parser is provided below.
//
// For now we use a simple approach: include httplib (already header-only) and
// do JSON parsing with a tiny helper.

namespace {

void quiet_ggml_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::cerr << text;
}

// --- Minimal JSON value ---------------------------------------------------

enum class JsonType { Null, Bool, Number, String, Array, Object };

struct JsonValue {
    JsonType type = JsonType::Null;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    bool is_string() const { return type == JsonType::String; }
    bool is_number() const { return type == JsonType::Number; }
    bool is_bool() const { return type == JsonType::Bool; }
    bool is_object() const { return type == JsonType::Object; }
    bool is_array() const { return type == JsonType::Array; }
    bool is_null() const { return type == JsonType::Null; }

    const std::string & as_string() const { return str_val; }
    double as_number() const { return num_val; }
    float as_float() const { return static_cast<float>(num_val); }
    int as_int() const { return static_cast<int>(num_val); }
    bool as_bool() const { return bool_val; }

    bool has(const std::string & key) const {
        if (type != JsonType::Object) return false;
        for (auto & [k, v] : obj) if (k == key) return true;
        return false;
    }

    const JsonValue & operator[](const std::string & key) const {
        static const JsonValue null_val;
        if (type != JsonType::Object) return null_val;
        for (auto & [k, v] : obj) if (k == key) return v;
        return null_val;
    }

    const JsonValue & operator[](size_t idx) const {
        static const JsonValue null_val;
        if (type != JsonType::Array || idx >= arr.size()) return null_val;
        return arr[idx];
    }

    size_t size() const { return type == JsonType::Array ? arr.size() : 0; }

    // Encode audio samples as 16-bit PCM in a WAV buffer
    std::string to_wav_bytes(const std::vector<float> & samples, int sample_rate) const {
        // not a JSON method — placed here for convenience
        (void)type;
        return std::string();
    }
};

// --- Minimal JSON parser --------------------------------------------------

struct JsonParser {
    const std::string & src;
    size_t pos = 0;

    explicit JsonParser(const std::string & s) : src(s) {}

    void skip_ws() {
        while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\n' || src[pos] == '\r')) ++pos;
    }

    JsonValue parse_string() {
        if (pos >= src.size() || src[pos] != '"') return {};
        ++pos;
        std::string out;
        while (pos < src.size() && src[pos] != '"') {
            if (src[pos] == '\\' && pos + 1 < src.size()) {
                ++pos;
                switch (src[pos]) {
                    case '"':  out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u': {
                        out += "\\u";
                        for (int i = 0; i < 4 && pos + 1 < src.size(); ++i) { ++pos; out += src[pos]; }
                        break;
                    }
                    default: out += src[pos]; break;
                }
            } else {
                out += src[pos];
            }
            ++pos;
        }
        if (pos < src.size()) ++pos; // skip closing "
        JsonValue v;
        v.type = JsonType::String;
        v.str_val = std::move(out);
        return v;
    }

    JsonValue parse_number() {
        size_t start = pos;
        if (pos < src.size() && src[pos] == '-') ++pos;
        while (pos < src.size() && (src[pos] >= '0' && src[pos] <= '9')) ++pos;
        if (pos < src.size() && src[pos] == '.') { ++pos; while (pos < src.size() && src[pos] >= '0' && src[pos] <= '9') ++pos; }
        if (pos < src.size() && (src[pos] == 'e' || src[pos] == 'E')) { ++pos; if (pos < src.size() && (src[pos] == '+' || src[pos] == '-')) ++pos; while (pos < src.size() && src[pos] >= '0' && src[pos] <= '9') ++pos; }
        JsonValue v;
        v.type = JsonType::Number;
        v.num_val = std::stod(src.substr(start, pos - start));
        return v;
    }

    JsonValue parse_array() {
        if (pos >= src.size() || src[pos] != '[') return {};
        ++pos;
        JsonValue v;
        v.type = JsonType::Array;
        skip_ws();
        if (pos < src.size() && src[pos] == ']') { ++pos; return v; }
        while (pos < src.size()) {
            skip_ws();
            v.arr.push_back(parse_value());
            skip_ws();
            if (pos < src.size() && src[pos] == ',') { ++pos; continue; }
            if (pos < src.size() && src[pos] == ']') { ++pos; break; }
            break;
        }
        return v;
    }

    JsonValue parse_object() {
        if (pos >= src.size() || src[pos] != '{') return {};
        ++pos;
        JsonValue v;
        v.type = JsonType::Object;
        skip_ws();
        if (pos < src.size() && src[pos] == '}') { ++pos; return v; }
        while (pos < src.size()) {
            skip_ws();
            auto key = parse_string();
            skip_ws();
            if (pos < src.size() && src[pos] == ':') ++pos;
            skip_ws();
            auto val = parse_value();
            v.obj.emplace_back(std::move(key.str_val), std::move(val));
            skip_ws();
            if (pos < src.size() && src[pos] == ',') { ++pos; continue; }
            if (pos < src.size() && src[pos] == '}') { ++pos; break; }
            break;
        }
        return v;
    }

    JsonValue parse_value() {
        skip_ws();
        if (pos >= src.size()) return {};
        char c = src[pos];
        if (c == '"') return parse_string();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't') { pos += 4; JsonValue v; v.type = JsonType::Bool; v.bool_val = true; return v; }
        if (c == 'f') { pos += 5; JsonValue v; v.type = JsonType::Bool; v.bool_val = false; return v; }
        if (c == 'n') { pos += 4; return {}; }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        return {};
    }
};

JsonValue parse_json(const std::string & s) {
    JsonParser p(s);
    return p.parse_value();
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
    if (format == "opus") return "audio/opus";
    if (format == "aac")  return "audio/aac";
    if (format == "flac") return "audio/flac";
    if (format == "pcm")  return "audio/pcm";
    return "audio/wav";
}

std::string json_error(const std::string & message, const std::string & type = "invalid_request_error") {
    return "{\"error\":{\"message\":\"" + message + "\",\"type\":\"" + type + "\"}}";
}

std::string get_opt_string(const JsonValue & j, const std::string & key, const std::string & def = "") {
    if (!j.has(key) || !j[key].is_string()) return def;
    return j[key].as_string();
}

double get_opt_number(const JsonValue & j, const std::string & key, double def = 0.0) {
    if (!j.has(key) || !j[key].is_number()) return def;
    return j[key].as_number();
}

bool get_opt_bool(const JsonValue & j, const std::string & key, bool def = false) {
    if (!j.has(key)) return def;
    if (j[key].is_bool()) return j[key].as_bool();
    if (j[key].is_number()) return j[key].as_number() != 0.0;
    return def;
}

// --- Voice preset system --------------------------------------------------
// Maps voice names to ref_audio/ref_text files on disk.
// Users can place WAV+txt pairs in a voices/ directory.

struct VoicePreset {
    std::string ref_audio_path;
    std::string ref_text;
};

std::string voices_dir;

VoicePreset load_voice_preset(const std::string & name) {
    VoicePreset p;
    if (voices_dir.empty()) return p;
    std::string base = voices_dir;
    if (!base.empty() && base.back() != '/') base += '/';
    base += name;
    // Try reading .txt for ref_text
    std::string txt_path = base + ".txt";
    std::ifstream tf(txt_path);
    if (tf) { std::getline(tf, p.ref_text); }
    // Check for audio file
    for (const char * ext : {".wav", ".mp3", ".ogg", ".flac"}) {
        std::string audio_path = base + ext;
        std::ifstream af(audio_path, std::ios::binary);
        if (af) { p.ref_audio_path = audio_path; break; }
    }
    return p;
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
        << "  --voices-dir DIR    directory with voice presets (name.wav + name.txt)\n"
        << "  --device cpu|cuda[:N]\n"
        << "  --backend cpu|cuda\n"
        << "  --threads N         number of CPU threads (default: 4)\n";
}

} // namespace

int main(int argc, char ** argv) {
    ggml_log_set(quiet_ggml_log, nullptr);

    ServerConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need = [&](const char * name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (key == "--model") cfg.model_path = need("--model");
        else if (key == "--host") cfg.host = need("--host");
        else if (key == "--port") cfg.port = std::stoi(need("--port"));
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
                if (pos != std::string::npos) cfg.device = std::stoi(dev.substr(pos + 1));
            }
        } else if (key == "--threads" || key == "-t") cfg.threads = std::stoi(need("--threads"));
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
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // GET /v1/models — list available models
    svr.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(
            "{\"object\":\"list\",\"data\":[{\"id\":\"omnivoice\",\"object\":\"model\",\"owned_by\":\"local\"}]}",
            "application/json");
    });

    // GET /v1/audio/voices — list available voice presets
    svr.Get("/v1/audio/voices", [&](const httplib::Request &, httplib::Response & res) {
        std::string json = "{\"object\":\"list\",\"data\":[";

        if (!voices_dir.empty()) {
            bool first = true;
            for (const auto & entry : std::filesystem::directory_iterator(voices_dir)) {
                if (!entry.is_regular_file()) continue;
                std::string path = entry.path().string();
                // Only look at audio files
                bool is_audio = false;
                for (const char * ext : {".wav", ".mp3", ".ogg", ".flac"}) {
                    size_t len = strlen(ext);
                    if (path.size() >= len && path.compare(path.size() - len, len, ext) == 0) {
                        is_audio = true;
                        break;
                    }
                }
                if (!is_audio) continue;

                std::string name = entry.path().stem().string();
                VoicePreset preset = load_voice_preset(name);

                if (!first) json += ",";
                first = false;
                json += "{\"id\":\"" + name + "\"";
                json += ",\"object\":\"voice\"";
                json += ",\"has_ref_text\":" + std::string(preset.ref_text.empty() ? "false" : "true");
                json += ",\"ref_text\":\"" + preset.ref_text + "\"";
                json += "}";
            }
        }

        json += "]}";
        res.set_content(json, "application/json");
    });

    // POST /v1/audio/speech — the main endpoint
    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        // Parse JSON body
        if (req.body.empty()) {
            res.status = 400;
            res.set_content(json_error("request body is empty"), "application/json");
            return;
        }

        JsonValue body = parse_json(req.body);
        if (body.is_null()) {
            res.status = 400;
            res.set_content(json_error("invalid JSON"), "application/json");
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

        // --- Response format ---
        std::string response_format = get_opt_string(body, "response_format", "wav");
        // Only wav and pcm are natively supported by OmniVoice
        if (response_format != "wav" && response_format != "pcm") {
            response_format = "wav";
        }

        // --- Speed ---
        float speed = static_cast<float>(get_opt_number(body, "speed", 1.0));
        if (speed < 0.25f) speed = 0.25f;
        if (speed > 4.0f) speed = 4.0f;

        // --- Build SynthesisParams ---
        omnivoice::SynthesisParams params;
        params.text = input_text;
        params.speed = speed;

        // Language
        if (body.has("language")) {
            params.language = get_opt_string(body, "language");
        }

        // Voice resolution:
        // 1. If "voice" is a known preset name → load ref_audio + ref_text
        // 2. If "voice" is a descriptive string → map to instruct (voice design)
        // 3. If "ref_audio" is provided (base64 or path) → voice cloning
        // 4. If "instructions" is provided → use as instruct
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
        if (body.has("ref_audio")) {
            // ref_audio can be a file path or a voice preset name
            std::string ref = get_opt_string(body, "ref_audio");
            if (!ref.empty()) {
                // Check if it's a file path
                std::ifstream test(ref);
                if (test) {
                    params.ref_audio_path = ref;
                } else {
                    // Try as preset name
                    VoicePreset p = load_voice_preset(ref);
                    if (!p.ref_audio_path.empty()) {
                        params.ref_audio_path = p.ref_audio_path;
                    }
                }
            }
        }
        if (body.has("ref_text")) {
            params.ref_text = get_opt_string(body, "ref_text");
        }

        // Instructions override voice if explicitly provided
        if (!instructions.empty() && voice_is_instruct) {
            // Both voice and instructions — merge them
            params.instruct = voice + ", " + instructions;
        } else if (!instructions.empty()) {
            params.instruct = instructions;
        }

        // --- GenerationConfig (OmniVoice extensions) ---
        if (body.has("num_step"))
            params.generation.num_step = static_cast<int>(get_opt_number(body, "num_step", 32));

        if (body.has("guidance_scale"))
            params.generation.guidance_scale = static_cast<float>(get_opt_number(body, "guidance_scale", 2.0));

        if (body.has("t_shift"))
            params.generation.t_shift = static_cast<float>(get_opt_number(body, "t_shift", 0.1));

        if (body.has("layer_penalty_factor"))
            params.generation.layer_penalty_factor = static_cast<float>(get_opt_number(body, "layer_penalty_factor", 5.0));

        if (body.has("position_temperature"))
            params.generation.position_temperature = static_cast<float>(get_opt_number(body, "position_temperature", 5.0));

        if (body.has("class_temperature"))
            params.generation.class_temperature = static_cast<float>(get_opt_number(body, "class_temperature", 0.0));

        if (body.has("denoise"))
            params.generation.denoise = get_opt_bool(body, "denoise", true);

        if (body.has("preprocess_prompt"))
            params.generation.preprocess_prompt = get_opt_bool(body, "preprocess_prompt", true);

        if (body.has("postprocess_output"))
            params.generation.postprocess_output = get_opt_bool(body, "postprocess_output", true);

        if (body.has("duration"))
            params.duration = static_cast<float>(get_opt_number(body, "duration", 0.0));

        if (body.has("audio_chunk_duration"))
            params.generation.audio_chunk_duration = static_cast<float>(get_opt_number(body, "audio_chunk_duration", 15.0));

        if (body.has("audio_chunk_threshold"))
            params.generation.audio_chunk_threshold = static_cast<float>(get_opt_number(body, "audio_chunk_threshold", 30.0));

        if (body.has("seed"))
            params.generation.seed = static_cast<uint64_t>(get_opt_number(body, "seed", 0));

        // --- Generate ---
        std::cerr << "[request] input=\"" << input_text.substr(0, 80)
                  << (input_text.size() > 80 ? "..." : "")
                  << "\" voice=\"" << voice
                  << "\" format=" << response_format
                  << " speed=" << speed << "\n";

        omnivoice::Audio audio;
        try {
            std::lock_guard<std::mutex> lock(gen_mutex);
            audio = runtime.generate(params);
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(json_error(std::string("generation failed: ") + e.what()), "application/json");
            std::cerr << "[error] " << e.what() << "\n";
            return;
        }

        // --- Encode response ---
        std::string audio_bytes;
        if (response_format == "pcm") {
            audio_bytes = encode_pcm(audio.samples);
        } else {
            audio_bytes = encode_wav(audio.samples, audio.sample_rate);
        }

        const double output_seconds = audio.sample_rate > 0
            ? double(audio.samples.size()) / double(audio.sample_rate)
            : 0.0;
        std::cerr << "[response] " << output_seconds << "s audio, "
                  << audio_bytes.size() << " bytes (" << response_format << ")\n";

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
}
