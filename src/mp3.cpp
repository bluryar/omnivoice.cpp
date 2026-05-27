#include "omnivoice.h"

extern "C" {
#include "layer3.h"
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace omnivoice {
namespace {

int16_t f32_to_s16(float sample) {
    const float clamped = std::max(-1.0f, std::min(1.0f, sample));
    return static_cast<int16_t>(std::lround(clamped * 32767.0f));
}

class ShineEncoder {
public:
    explicit ShineEncoder(shine_config_t & config) : encoder_(shine_initialise(&config)) {
        if (!encoder_) throw std::runtime_error("failed to initialize MP3 encoder");
    }

    ~ShineEncoder() {
        shine_close(encoder_);
    }

    ShineEncoder(const ShineEncoder &) = delete;
    ShineEncoder & operator=(const ShineEncoder &) = delete;

    shine_t get() const {
        return encoder_;
    }

private:
    shine_t encoder_ = nullptr;
};

void append_shine_data(std::string & out, unsigned char * data, int written) {
    if (data && written > 0) out.append(reinterpret_cast<const char *>(data), static_cast<size_t>(written));
}

} // namespace

std::string encode_mp3_mono_f32(const std::vector<float> & samples, int sample_rate, int bitrate_kbps) {
    if (sample_rate <= 0) throw std::runtime_error("MP3 sample rate must be positive");
    if (bitrate_kbps <= 0) throw std::runtime_error("MP3 bitrate must be positive");

    shine_config_t config = {};
    config.wave.channels = PCM_MONO;
    config.wave.samplerate = sample_rate;
    shine_set_config_mpeg_defaults(&config.mpeg);
    config.mpeg.mode = MONO;
    config.mpeg.bitr = bitrate_kbps;

    if (shine_check_config(sample_rate, bitrate_kbps) < 0) {
        throw std::runtime_error("unsupported MP3 sample rate/bitrate combination");
    }

    ShineEncoder encoder(config);
    const int samples_per_pass = shine_samples_per_pass(encoder.get());
    if (samples_per_pass <= 0) throw std::runtime_error("invalid MP3 encoder frame size");

    std::string out;
    out.reserve(samples.size() * 2);
    std::vector<int16_t> frame(static_cast<size_t>(samples_per_pass), 0);

    size_t offset = 0;
    while (offset < samples.size()) {
        const size_t n = std::min(static_cast<size_t>(samples_per_pass), samples.size() - offset);
        std::fill(frame.begin(), frame.end(), 0);
        for (size_t i = 0; i < n; ++i) frame[i] = f32_to_s16(samples[offset + i]);

        int written = 0;
        unsigned char * data = shine_encode_buffer_interleaved(encoder.get(), frame.data(), &written);
        append_shine_data(out, data, written);
        offset += n;
    }

    int written = 0;
    unsigned char * data = shine_flush(encoder.get(), &written);
    append_shine_data(out, data, written);
    return out;
}

void write_mp3_mono_f32(const std::string & path, const std::vector<float> & samples, int sample_rate, int bitrate_kbps) {
    const std::string mp3 = encode_mp3_mono_f32(samples, sample_rate, bitrate_kbps);
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("failed to write MP3: " + path);
    os.write(mp3.data(), static_cast<std::streamsize>(mp3.size()));
}

} // namespace omnivoice
