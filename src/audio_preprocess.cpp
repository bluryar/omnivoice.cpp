#include "omnivoice_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <numeric>

namespace omnivoice {
namespace {

float rms_window(const std::vector<float> & audio, int start, int end) {
    if (end <= start) return 0.0f;
    double sum = 0.0;
    for (int i = start; i < end; ++i) sum += double(audio[static_cast<size_t>(i)]) * double(audio[static_cast<size_t>(i)]);
    return static_cast<float>(std::sqrt(sum / double(end - start)));
}

std::vector<std::pair<int, int>> active_segments(const std::vector<float> & audio, int sample_rate, float threshold_db) {
    const int step = std::max(1, int(std::lround(sample_rate * 10.0 / 1000.0)));
    const float threshold = std::pow(10.0f, threshold_db / 20.0f);
    const int n_windows = (int(audio.size()) + step - 1) / step;
    std::vector<std::pair<int, int>> segments;
    int start = -1;
    for (int i = 0; i < n_windows; ++i) {
        const int a = i * step;
        const int b = std::min<int>(audio.size(), a + step);
        const bool active = rms_window(audio, a, b) >= threshold;
        if (active && start < 0) {
            start = a;
        } else if (!active && start >= 0) {
            segments.push_back({start, a});
            start = -1;
        }
    }
    if (start >= 0) segments.push_back({start, int(audio.size())});
    return segments;
}

std::vector<float> trim_edges(const std::vector<float> & audio, int sample_rate, int lead_ms, int trail_ms, float threshold_db) {
    auto segs = active_segments(audio, sample_rate, threshold_db);
    if (segs.empty()) return {};
    const int lead = int(std::lround(sample_rate * lead_ms / 1000.0));
    const int trail = int(std::lround(sample_rate * trail_ms / 1000.0));
    const int start = std::max(0, segs.front().first - lead);
    const int end = std::min<int>(audio.size(), segs.back().second + trail);
    return std::vector<float>(audio.begin() + start, audio.begin() + end);
}

uint32_t utf8_decode_one_local(const std::string & s, size_t * pos) {
    const unsigned char c = static_cast<unsigned char>(s[*pos]);
    if (c < 0x80) {
        *pos += 1;
        return c;
    }
    if ((c >> 5) == 0x6 && *pos + 1 < s.size()) {
        uint32_t cp = ((c & 0x1f) << 6) | (static_cast<unsigned char>(s[*pos + 1]) & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c >> 4) == 0xe && *pos + 2 < s.size()) {
        uint32_t cp = ((c & 0x0f) << 12)
            | ((static_cast<unsigned char>(s[*pos + 1]) & 0x3f) << 6)
            | (static_cast<unsigned char>(s[*pos + 2]) & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c >> 3) == 0x1e && *pos + 3 < s.size()) {
        uint32_t cp = ((c & 0x07) << 18)
            | ((static_cast<unsigned char>(s[*pos + 1]) & 0x3f) << 12)
            | ((static_cast<unsigned char>(s[*pos + 2]) & 0x3f) << 6)
            | (static_cast<unsigned char>(s[*pos + 3]) & 0x3f);
        *pos += 4;
        return cp;
    }
    *pos += 1;
    return 0xfffd;
}

bool is_sentence_punctuation(uint32_t cp) {
    switch (cp) {
        case '.':
        case ',':
        case ';':
        case ':':
        case '!':
        case '?':
        case 0x3002:
        case 0xff0c:
        case 0xff1b:
        case 0xff1a:
        case 0xff01:
        case 0xff1f:
            return true;
        default:
            return false;
    }
}

bool is_terminal_punctuation(uint32_t cp) {
    if (is_sentence_punctuation(cp)) return true;
    switch (cp) {
        case ')':
        case ']':
        case '}':
        case '"':
        case '\'':
        case 0x201d:
        case 0x2019:
        case 0x300d:
        case 0x300f:
        case 0x3011:
        case 0xff09:
            return true;
        default:
            return false;
    }
}

uint32_t last_codepoint(const std::string & text) {
    uint32_t last = 0;
    size_t pos = 0;
    while (pos < text.size()) last = utf8_decode_one_local(text, &pos);
    return last;
}

} // namespace

std::vector<float> remove_silence(const std::vector<float> & audio, int sample_rate, int mid_sil_ms, int lead_sil_ms, int trail_sil_ms) {
    std::vector<float> cur = audio;
    if (mid_sil_ms > 0) {
        auto segs = active_segments(cur, sample_rate, -50.0f);
        if (segs.empty()) return {};
        const int keep = int(std::lround(sample_rate * mid_sil_ms / 1000.0));
        std::vector<std::pair<int, int>> expanded;
        for (auto [s, e] : segs) {
            expanded.push_back({std::max(0, s - keep), std::min<int>(cur.size(), e + keep)});
        }
        std::vector<std::pair<int, int>> merged;
        for (auto seg : expanded) {
            if (merged.empty() || seg.first > merged.back().second) {
                merged.push_back(seg);
            } else {
                merged.back().second = std::max(merged.back().second, seg.second);
            }
        }
        std::vector<float> kept;
        for (auto [s, e] : merged) kept.insert(kept.end(), cur.begin() + s, cur.begin() + e);
        cur.swap(kept);
    }
    return trim_edges(cur, sample_rate, lead_sil_ms, trail_sil_ms, -50.0f);
}

std::vector<float> fade_and_pad_audio(const std::vector<float> & audio, int sample_rate) {
    if (audio.empty()) return audio;
    std::vector<float> out = audio;
    const int fade = int(0.1f * sample_rate);
    const int pad = int(0.1f * sample_rate);
    const int k = std::min<int>(fade, out.size() / 2);
    for (int i = 0; i < k; ++i) {
        const float fi = k <= 1 ? 1.0f : float(i) / float(k - 1);
        out[static_cast<size_t>(i)] *= fi;
        out[out.size() - 1 - static_cast<size_t>(i)] *= fi;
    }
    std::vector<float> padded(static_cast<size_t>(pad), 0.0f);
    padded.insert(padded.end(), out.begin(), out.end());
    padded.insert(padded.end(), static_cast<size_t>(pad), 0.0f);
    return padded;
}

std::vector<float> cross_fade_chunks(const std::vector<std::vector<float>> & chunks, int sample_rate) {
    if (chunks.empty()) return {};
    if (chunks.size() == 1) return chunks[0];
    const int total_n = int(0.3f * sample_rate);
    const int fade_n = total_n / 3;
    const int silence_n = fade_n;
    std::vector<float> merged = chunks[0];
    for (size_t ci = 1; ci < chunks.size(); ++ci) {
        std::vector<float> current = chunks[ci];
        const int fout = std::min<int>(fade_n, merged.size());
        for (int i = 0; i < fout; ++i) {
            const float scale = fout <= 1 ? 0.0f : 1.0f - float(i) / float(fout - 1);
            merged[merged.size() - fout + static_cast<size_t>(i)] *= scale;
        }
        merged.insert(merged.end(), static_cast<size_t>(silence_n), 0.0f);
        const int fin = std::min<int>(fade_n, current.size());
        for (int i = 0; i < fin; ++i) {
            const float scale = fin <= 1 ? 1.0f : float(i) / float(fin - 1);
            current[static_cast<size_t>(i)] *= scale;
        }
        merged.insert(merged.end(), current.begin(), current.end());
    }
    return merged;
}

std::vector<float> resample_linear(const std::vector<float> & input, int src_rate, int dst_rate) {
    if (src_rate == dst_rate || input.empty()) return input;
    const double ratio = double(dst_rate) / double(src_rate);
    const size_t out_n = std::max<size_t>(1, size_t(std::llround(double(input.size()) * ratio)));
    std::vector<float> out(out_n);
    for (size_t i = 0; i < out_n; ++i) {
        const double src = double(i) / ratio;
        const size_t j = std::min<size_t>(input.size() - 1, size_t(std::floor(src)));
        const size_t k = std::min<size_t>(input.size() - 1, j + 1);
        const float t = float(src - double(j));
        out[i] = input[j] * (1.0f - t) + input[k] * t;
    }
    return out;
}

std::string add_punctuation(const std::string & text) {
    std::string out = text;
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) out.pop_back();
    if (out.empty()) return out;
    if (!is_terminal_punctuation(last_codepoint(out))) {
        out += contains_cjk(out) ? "。" : ".";
    }
    return out;
}

std::vector<std::string> chunk_text_punctuation(const std::string & text, int chunk_len, int min_chunk_len) {
    std::vector<std::string> sentences;
    std::string cur;
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t start = pos;
        const uint32_t cp = utf8_decode_one_local(text, &pos);
        cur.append(text, start, pos - start);
        if (is_sentence_punctuation(cp)) {
            sentences.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) sentences.push_back(cur);

    std::vector<std::string> chunks;
    cur.clear();
    for (const std::string & s : sentences) {
        if (!cur.empty() && utf8_length(cur) + utf8_length(s) > chunk_len) {
            chunks.push_back(cur);
            cur.clear();
        }
        cur += s;
    }
    if (!cur.empty()) chunks.push_back(cur);
    if (chunks.size() >= 2 && utf8_length(chunks[0]) < min_chunk_len) {
        chunks[1] = chunks[0] + chunks[1];
        chunks.erase(chunks.begin());
    }
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(), [](const std::string & s) { return s.empty(); }), chunks.end());
    return chunks;
}

} // namespace omnivoice
