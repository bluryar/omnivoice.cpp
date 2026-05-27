#include "omnivoice.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;

void quiet_ggml_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::cerr << text;
}

bool parse_bool(const std::string & v) {
    return v == "1" || v == "true" || v == "yes" || v == "y";
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

using Clock = std::chrono::steady_clock;

double elapsed_s(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct TracePrinter {
    std::map<std::string, double> phase_seconds;
    double reference_audio_seconds = 0.0;
    bool progress_active = false;

    void clear_progress_line() {
        if (!progress_active) return;
        std::cerr << "\r" << std::string(120, ' ') << "\r";
        progress_active = false;
    }

    static std::string chunk_label(const omnivoice::TraceEvent & event) {
        if (event.chunk_count <= 1) return "";
        return " chunk " + std::to_string(event.chunk_index) + "/" + std::to_string(event.chunk_count);
    }

    void stage_begin(const omnivoice::TraceEvent & event) {
        clear_progress_line();
        std::cerr << "[stage] " << event.phase << "/" << event.name << chunk_label(event) << " ...\n";
    }

    void stage_end(const omnivoice::TraceEvent & event) {
        clear_progress_line();
        phase_seconds[event.phase] += event.seconds;
        if (event.phase == "reference_encode" && event.audio_seconds > 0.0) {
            reference_audio_seconds = std::max(reference_audio_seconds, event.audio_seconds);
        }
        std::cerr << "[stage] " << event.phase << "/" << event.name << chunk_label(event)
                  << " done " << std::fixed << std::setprecision(3) << event.seconds << "s\n";
    }

    void llm_progress(const omnivoice::TraceEvent & event) {
        const int total = std::max(1, event.total);
        const int current = std::max(0, std::min(event.current, total));
        const double ratio = double(current) / double(total);
        constexpr int width = 30;
        const int fill = std::max(0, std::min(width, int(ratio * width)));

        std::string bar;
        bar.reserve(width);
        bar.append(static_cast<size_t>(fill), '=');
        if (fill < width) {
            bar.push_back('>');
            bar.append(static_cast<size_t>(width - fill - 1), ' ');
        }

        std::cerr << "\r[llm] " << event.name << chunk_label(event)
                  << " [" << bar << "] "
                  << current << "/" << total << " steps";
        if (event.total_positions > 0) {
            std::cerr << "  " << event.updated << "/" << event.total_positions << " positions";
        }
        std::cerr << std::flush;
        progress_active = true;
        if (current >= total) {
            std::cerr << "\n";
            progress_active = false;
        }
    }

    void operator()(const omnivoice::TraceEvent & event) {
        switch (event.kind) {
            case omnivoice::TraceEventKind::StageBegin:
                stage_begin(event);
                break;
            case omnivoice::TraceEventKind::StageEnd:
                stage_end(event);
                break;
            case omnivoice::TraceEventKind::LlmProgress:
                llm_progress(event);
                break;
        }
    }
};

void usage() {
    std::cerr
        << "usage: omnivoice-cli --model MODEL.gguf --text TEXT --output out.wav|out.mp3 [options]\n"
        << "options:\n"
        << "  --response-format wav|mp3  output format, inferred from --output when omitted\n"
        << "  --language LANG --instruct TEXT --auto-voice true|false\n"
        << "  --num-step N --guidance-scale F --speed F --duration F --t-shift F\n"
        << "  --denoise true|false --postprocess-output true|false --preprocess-prompt true|false\n"
        << "  --layer-penalty-factor F --position-temperature F --class-temperature F\n"
        << "  --device cpu|cuda[:N] --backend cpu|cuda --seed N --threads N\n";
}

} // namespace

int main(int argc, char ** argv) {
    ggml_log_set(quiet_ggml_log, nullptr);
    try {
        std::string model;
        std::string output;
        std::string response_format;
        omnivoice::SynthesisParams params;
        omnivoice::RuntimeOptions options;

        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            auto need = [&](const char * name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
                return argv[++i];
            };
            if (key == "--model") model = need("--model");
            else if (key == "--text") params.text = need("--text");
            else if (key == "--output" || key == "--out") output = need("--output");
            else if (key == "--response-format" || key == "--response_format" || key == "--format") response_format = lower(need("--response-format"));
            else if (key == "--ref-audio" || key == "--ref_audio") params.ref_audio_path = need("--ref-audio");
            else if (key == "--ref-text" || key == "--ref_text") params.ref_text = need("--ref-text");
            else if (key == "--language") params.language = need("--language");
            else if (key == "--instruct") params.instruct = need("--instruct");
            else if (key == "--auto-voice" || key == "--auto_voice") params.auto_voice = parse_bool(need("--auto-voice"));
            else if (key == "--num-step" || key == "--num_step") params.generation.num_step = std::stoi(need("--num-step"));
            else if (key == "--guidance-scale" || key == "--guidance_scale") params.generation.guidance_scale = std::stof(need("--guidance-scale"));
            else if (key == "--speed") params.speed = std::stof(need("--speed"));
            else if (key == "--duration") params.duration = std::stof(need("--duration"));
            else if (key == "--t-shift" || key == "--t_shift") params.generation.t_shift = std::stof(need("--t-shift"));
            else if (key == "--denoise") params.generation.denoise = parse_bool(need("--denoise"));
            else if (key == "--postprocess-output" || key == "--postprocess_output") params.generation.postprocess_output = parse_bool(need("--postprocess-output"));
            else if (key == "--preprocess-prompt" || key == "--preprocess_prompt") params.generation.preprocess_prompt = parse_bool(need("--preprocess-prompt"));
            else if (key == "--layer-penalty-factor" || key == "--layer_penalty_factor") params.generation.layer_penalty_factor = std::stof(need("--layer-penalty-factor"));
            else if (key == "--position-temperature" || key == "--position_temperature") params.generation.position_temperature = std::stof(need("--position-temperature"));
            else if (key == "--class-temperature" || key == "--class_temperature") params.generation.class_temperature = std::stof(need("--class-temperature"));
            else if (key == "--backend") {
                options.backend = need("--backend");
                if (options.backend == "rocm" || options.backend == "hip") options.backend = "cuda";
            } else if (key == "--device") {
                std::string dev = need("--device");
                if (dev == "cpu") options.backend = "cpu";
                else if (dev.rfind("cuda", 0) == 0) {
                    options.backend = "cuda";
                    auto pos = dev.find(':');
                    if (pos != std::string::npos) options.device = std::stoi(dev.substr(pos + 1));
                } else {
                    throw std::runtime_error("unsupported --device: " + dev);
                }
            } else if (key == "--seed") params.generation.seed = static_cast<uint64_t>(std::stoull(need("--seed")));
            else if (key == "--threads" || key == "-t") options.threads = std::stoi(need("--threads"));
            else if (key == "--help" || key == "-h") {
                usage();
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + key);
            }
        }

        if (model.empty() || params.text.empty() || output.empty()) {
            usage();
            return 2;
        }
        if (!params.ref_audio_path.empty() && params.ref_text.empty()) {
            throw std::runtime_error("--ref-text is required with --ref-audio");
        }

        TracePrinter trace_printer;
        options.trace = [&](const omnivoice::TraceEvent & event) {
            trace_printer(event);
        };

        std::cerr << "[stage] model_load ...\n";
        const auto load_start = Clock::now();
        omnivoice::OmniVoiceRuntime runtime(model, options);
        std::cerr << "[stage] model_load done " << std::fixed << std::setprecision(3) << elapsed_s(load_start) << "s\n";

        std::cerr << "[stage] generate ...\n";
        const auto generate_start = Clock::now();
        omnivoice::Audio audio = runtime.generate(params);
        const double generate_seconds = elapsed_s(generate_start);
        std::cerr << "[stage] generate done " << std::fixed << std::setprecision(3) << generate_seconds << "s\n";

        const std::string output_ext = lower(fs::path(output).extension().string());
        if (response_format.empty()) {
            response_format = output_ext == ".mp3" ? "mp3" : "wav";
        }
        if (response_format != "wav" && response_format != "mp3") {
            throw std::runtime_error("unsupported --response-format: " + response_format);
        }

        const auto write_start = Clock::now();
        if (response_format == "mp3") {
            std::cerr << "[stage] write_mp3 ...\n";
            omnivoice::write_mp3_mono_f32(output, audio.samples, audio.sample_rate);
            std::cerr << "[stage] write_mp3 done " << std::fixed << std::setprecision(3) << elapsed_s(write_start) << "s\n";
        } else {
            std::cerr << "[stage] write_wav ...\n";
            omnivoice::write_wav_mono_f32(output, audio.samples, audio.sample_rate);
            std::cerr << "[stage] write_wav done " << std::fixed << std::setprecision(3) << elapsed_s(write_start) << "s\n";
        }

        const double output_seconds = audio.sample_rate > 0
            ? double(audio.samples.size()) / double(audio.sample_rate)
            : 0.0;
        const auto print_rtf = [](const std::string & name, double seconds, double audio_seconds) {
            std::cerr << "[rtf] " << name << "=";
            if (audio_seconds > 0.0) {
                std::cerr << std::fixed << std::setprecision(3) << (seconds / audio_seconds);
            } else {
                std::cerr << "n/a";
            }
            std::cerr << "  seconds=" << std::fixed << std::setprecision(3) << seconds;
        };

        std::cerr << "[rtf] output_audio_s=" << std::fixed << std::setprecision(3) << output_seconds << "\n";
        print_rtf("total", generate_seconds, output_seconds);
        std::cerr << "\n";
        print_rtf("llm", trace_printer.phase_seconds["llm"], output_seconds);
        std::cerr << "\n";
        if (trace_printer.reference_audio_seconds > 0.0) {
            print_rtf("reference_encode", trace_printer.phase_seconds["reference_encode"], trace_printer.reference_audio_seconds);
            std::cerr << "  ref_audio_s=" << std::fixed << std::setprecision(3) << trace_printer.reference_audio_seconds << "\n";
        } else {
            std::cerr << "[rtf] reference_encode=n/a\n";
        }
        print_rtf("decode", trace_printer.phase_seconds["decode"], output_seconds);
        std::cerr << "\n";
        std::cerr << "saved " << response_format << " audio to " << output << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
