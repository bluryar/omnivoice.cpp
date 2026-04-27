#pragma once

#include "omnivoice.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace omnivoice {

struct Tensor2i {
    int rows = 0;
    int cols = 0;
    std::vector<int32_t> data;

    int32_t & operator()(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    int32_t operator()(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
};

struct InferenceInputs {
    Tensor2i input_ids;
    std::vector<float> audio_mask;
    int style_length = 0;
    int text_length = 0;
    int ref_audio_length = 0;
    int target_length = 0;

    int total_length() const { return input_ids.cols; }
    int audio_start() const { return total_length() - ref_audio_length - target_length; }
    int target_start() const { return total_length() - target_length; }
};

struct VoiceClonePrompt {
    Tensor2i ref_audio_tokens;
    std::string ref_text;
    std::optional<float> ref_rms;
    double ref_audio_seconds = 0.0;
};

struct Qwen3Spec {
    int vocab_size = 0;
    int context_length = 0;
    int embedding_length = 0;
    int feed_forward_length = 0;
    int block_count = 0;
    int head_count = 0;
    int head_count_kv = 0;
    int head_dim = 0;
    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 1000000.0f;
};

struct OmniVoiceSpec {
    Qwen3Spec qwen3;
    int num_audio_codebook = 0;
    int audio_vocab_size = 0;
    int audio_mask_id = 0;
    std::vector<int> audio_codebook_weights;
};

struct HiggsSpec {
    int sample_rate = 24000;
    int semantic_sample_rate = 16000;
    int downsample_factor = 320;
    std::vector<float> target_bandwidths;
    std::vector<int> strides;
    std::vector<int> channel_ratios;
    std::vector<int> block_dilations;
    int codebook_size = 1024;
    int codebook_dim = 64;
    int kernel_size = 3;
    int unit_kernel_size = 3;
    int acoustic_hidden_size = 256;
    std::vector<int> acoustic_downsampling_ratios;
    int acoustic_decoder_hidden_size = 1024;
    std::vector<int> acoustic_upsampling_ratios;
    int acoustic_hop_length = 960;
    int semantic_hidden_size = 768;
    std::vector<int> semantic_conv_dims;
    std::vector<int> semantic_conv_kernels;
    std::vector<int> semantic_conv_strides;
    float semantic_layer_norm_eps = 1.0e-5f;
    int semantic_intermediate_size = 3072;
    int semantic_head_count = 12;
    int semantic_block_count = 12;
    int semantic_pos_conv_embeddings = 128;
    int semantic_pos_conv_groups = 16;

    int hop_length() const;
    int frame_rate() const;
    int semantic_head_dim() const;
    int semantic_downsample_factor() const;
    int num_quantizers() const;
    int num_quantizers_for_bandwidth(std::optional<float> bandwidth) const;
    int acoustic_encode_output_length(int input_length) const;
    int semantic_encode_output_length(int input_length) const;
    int semantic_feature_extractor_output_length(int input_length) const;
};

class Tokenizer {
public:
    Tokenizer(std::vector<std::string> tokens, std::vector<std::string> merges);
    std::vector<int32_t> encode(const std::string & text, bool add_special_tokens = true) const;
    bool has_token(const std::string & token) const;
    int32_t token_id(const std::string & token) const;

private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<std::string> special_tokens_;

    std::vector<int32_t> encode_ordinary(const std::string & text) const;
    std::vector<std::string> pretokenize(const std::string & text) const;
    std::vector<int32_t> bpe(const std::string & piece) const;
};

std::string resolve_language(const std::string & language);
std::string resolve_instruct(const std::string & instruct, bool use_zh);
bool contains_cjk(const std::string & text);
int utf8_length(const std::string & text);
float text_duration_weight(const std::string & text);
std::string add_punctuation(const std::string & text);
std::vector<std::string> chunk_text_punctuation(const std::string & text, int chunk_len, int min_chunk_len);

std::vector<float> remove_silence(const std::vector<float> & audio, int sample_rate, int mid_sil_ms, int lead_sil_ms, int trail_sil_ms);
std::vector<float> fade_and_pad_audio(const std::vector<float> & audio, int sample_rate);
std::vector<float> cross_fade_chunks(const std::vector<std::vector<float>> & chunks, int sample_rate);
std::vector<float> resample_linear(const std::vector<float> & input, int src_rate, int dst_rate);

} // namespace omnivoice
