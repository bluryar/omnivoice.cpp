#include "omnivoice_internal.h"

#include "ggml-cpu.h"

#if defined(GGML_USE_CUDA) || defined(GGML_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>

namespace omnivoice {
namespace {

constexpr int QWEN3_ROPE_MODE_NEOX = 2;

int conv1d_out_len(int input, int kernel, int stride, int padding, int dilation = 1) {
    return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

std::string tensor_name(const std::string & prefix, const std::string & suffix) {
    return prefix + suffix;
}

struct BackendHandle {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    std::string name;
    int threads = 4;

    BackendHandle(const RuntimeOptions & options) : name(options.backend), threads(options.threads) {
        ggml_time_init();
        if (name == "cuda") {
#if defined(GGML_USE_CUDA) || defined(GGML_CUDA)
            backend = ggml_backend_cuda_init(options.device);
            buft = ggml_backend_cuda_buffer_type(options.device);
#else
            throw std::runtime_error("CUDA backend is not available in this build");
#endif
        } else if (name == "cpu") {
            backend = ggml_backend_cpu_init();
            buft = ggml_backend_cpu_buffer_type();
            ggml_backend_cpu_set_n_threads(backend, threads);
        } else {
            throw std::runtime_error("unsupported backend: " + name);
        }
        if (!backend || !buft) throw std::runtime_error("failed to initialize backend: " + name);
    }

    ~BackendHandle() {
        if (backend) ggml_backend_free(backend);
    }
};

struct GraphHandle {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocr = nullptr;

    GraphHandle(ggml_backend_buffer_type_t buft, size_t max_nodes, size_t extra_meta = 64ull * 1024ull * 1024ull) {
        ggml_init_params params{};
        params.mem_size = ggml_graph_overhead_custom(max_nodes, false) + extra_meta;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("ggml_init failed");
        graph = ggml_new_graph_custom(ctx, max_nodes, false);
        allocr = ggml_gallocr_new(buft);
    }

    ~GraphHandle() {
        if (allocr) ggml_gallocr_free(allocr);
        if (ctx) ggml_free(ctx);
    }

    void reserve_alloc() {
        if (!ggml_gallocr_reserve(allocr, graph)) throw std::runtime_error("ggml_gallocr_reserve failed");
        if (!ggml_gallocr_alloc_graph(allocr, graph)) throw std::runtime_error("ggml_gallocr_alloc_graph failed");
    }
};

struct Model {
    BackendHandle backend;
    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    OmniVoiceSpec spec;
    HiggsSpec higgs;
    Tokenizer tokenizer;
    std::function<void(const TraceEvent &)> trace;

    Model(const std::string & path, const RuntimeOptions & options)
        : backend(options), tokenizer(load_tokens(path), load_merges(path)), trace(options.trace) {
        load(path);
    }

    ~Model() {
        if (weight_buf) ggml_backend_buffer_free(weight_buf);
        if (weight_ctx) ggml_free(weight_ctx);
    }

    ggml_tensor * t(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("GGUF missing tensor: " + name);
        return it->second;
    }

    static std::vector<std::string> load_string_array(const std::string & path, const char * key) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = nullptr;
        gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
        if (!ctx) throw std::runtime_error("failed to open GGUF metadata: " + path);
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) {
            gguf_free(ctx);
            return {};
        }
        std::vector<std::string> out;
        const size_t n = gguf_get_arr_n(ctx, kid);
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) out.emplace_back(gguf_get_arr_str(ctx, kid, i));
        gguf_free(ctx);
        return out;
    }

    static std::vector<std::string> load_tokens(const std::string & path) {
        auto tokens = load_string_array(path, "tokenizer.ggml.tokens");
        if (tokens.empty()) throw std::runtime_error("GGUF missing tokenizer.ggml.tokens");
        return tokens;
    }

    static std::vector<std::string> load_merges(const std::string & path) {
        return load_string_array(path, "tokenizer.ggml.merges");
    }

    static int64_t get_i(gguf_context * ctx, const char * key) {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) throw std::runtime_error(std::string("missing GGUF metadata key: ") + key);
        switch (gguf_get_kv_type(ctx, kid)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, kid);
            case GGUF_TYPE_INT32: return gguf_get_val_i32(ctx, kid);
            case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx, kid));
            case GGUF_TYPE_INT64: return gguf_get_val_i64(ctx, kid);
            default: throw std::runtime_error(std::string("metadata key is not integer: ") + key);
        }
    }

    static float get_f(gguf_context * ctx, const char * key) {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) throw std::runtime_error(std::string("missing GGUF metadata key: ") + key);
        switch (gguf_get_kv_type(ctx, kid)) {
            case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, kid);
            case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(ctx, kid));
            case GGUF_TYPE_UINT32: return static_cast<float>(gguf_get_val_u32(ctx, kid));
            case GGUF_TYPE_INT32: return static_cast<float>(gguf_get_val_i32(ctx, kid));
            default: throw std::runtime_error(std::string("metadata key is not float: ") + key);
        }
    }

    static std::vector<int> get_i_arr(gguf_context * ctx, const char * key) {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) throw std::runtime_error(std::string("missing GGUF metadata key: ") + key);
        std::vector<int> out;
        const size_t n = gguf_get_arr_n(ctx, kid);
        out.reserve(n);
        const gguf_type ty = gguf_get_arr_type(ctx, kid);
        const void * data = gguf_get_arr_data(ctx, kid);
        for (size_t i = 0; i < n; ++i) {
            if (ty == GGUF_TYPE_INT32) out.push_back(static_cast<const int32_t *>(data)[i]);
            else if (ty == GGUF_TYPE_UINT32) out.push_back(static_cast<int>(static_cast<const uint32_t *>(data)[i]));
            else if (ty == GGUF_TYPE_INT64) out.push_back(static_cast<int>(static_cast<const int64_t *>(data)[i]));
            else throw std::runtime_error(std::string("metadata array is not integer: ") + key);
        }
        return out;
    }

    static std::vector<float> get_f_arr(gguf_context * ctx, const char * key) {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) throw std::runtime_error(std::string("missing GGUF metadata key: ") + key);
        std::vector<float> out;
        const size_t n = gguf_get_arr_n(ctx, kid);
        out.reserve(n);
        const gguf_type ty = gguf_get_arr_type(ctx, kid);
        const void * data = gguf_get_arr_data(ctx, kid);
        for (size_t i = 0; i < n; ++i) {
            if (ty == GGUF_TYPE_FLOAT32) out.push_back(static_cast<const float *>(data)[i]);
            else if (ty == GGUF_TYPE_FLOAT64) out.push_back(static_cast<float>(static_cast<const double *>(data)[i]));
            else throw std::runtime_error(std::string("metadata array is not float: ") + key);
        }
        return out;
    }

    void parse_specs(gguf_context * ctx) {
        spec.qwen3.vocab_size = int(get_i(ctx, "qwen3.vocab_size"));
        spec.qwen3.context_length = int(get_i(ctx, "qwen3.context_length"));
        spec.qwen3.embedding_length = int(get_i(ctx, "qwen3.embedding_length"));
        spec.qwen3.feed_forward_length = int(get_i(ctx, "qwen3.feed_forward_length"));
        spec.qwen3.block_count = int(get_i(ctx, "qwen3.block_count"));
        spec.qwen3.head_count = int(get_i(ctx, "qwen3.attention.head_count"));
        spec.qwen3.head_count_kv = int(get_i(ctx, "qwen3.attention.head_count_kv"));
        spec.qwen3.head_dim = int(get_i(ctx, "qwen3.attention.key_length"));
        spec.qwen3.rms_norm_eps = get_f(ctx, "qwen3.attention.layer_norm_rms_epsilon");
        spec.qwen3.rope_theta = get_f(ctx, "qwen3.rope.freq_base");
        spec.num_audio_codebook = int(get_i(ctx, "omnivoice.num_audio_codebook"));
        spec.audio_vocab_size = int(get_i(ctx, "omnivoice.audio_vocab_size"));
        spec.audio_mask_id = int(get_i(ctx, "omnivoice.audio_mask_id"));
        spec.audio_codebook_weights = get_i_arr(ctx, "omnivoice.audio_codebook_weights");

        higgs.sample_rate = int(get_i(ctx, "omnivoice.higgs.sample_rate"));
        higgs.semantic_sample_rate = int(get_i(ctx, "omnivoice.higgs.semantic_sample_rate"));
        higgs.downsample_factor = int(get_i(ctx, "omnivoice.higgs.downsample_factor"));
        higgs.target_bandwidths = get_f_arr(ctx, "omnivoice.higgs.target_bandwidths");
        higgs.strides = get_i_arr(ctx, "omnivoice.higgs.strides");
        higgs.channel_ratios = get_i_arr(ctx, "omnivoice.higgs.channel_ratios");
        higgs.block_dilations = get_i_arr(ctx, "omnivoice.higgs.block_dilations");
        higgs.codebook_size = int(get_i(ctx, "omnivoice.higgs.codebook_size"));
        higgs.codebook_dim = int(get_i(ctx, "omnivoice.higgs.codebook_dim"));
        higgs.kernel_size = int(get_i(ctx, "omnivoice.higgs.kernel_size"));
        higgs.unit_kernel_size = int(get_i(ctx, "omnivoice.higgs.unit_kernel_size"));
        higgs.acoustic_hidden_size = int(get_i(ctx, "omnivoice.higgs.acoustic_hidden_size"));
        higgs.acoustic_downsampling_ratios = get_i_arr(ctx, "omnivoice.higgs.acoustic_downsampling_ratios");
        higgs.acoustic_decoder_hidden_size = int(get_i(ctx, "omnivoice.higgs.acoustic_decoder_hidden_size"));
        higgs.acoustic_upsampling_ratios = get_i_arr(ctx, "omnivoice.higgs.acoustic_upsampling_ratios");
        higgs.acoustic_hop_length = int(get_i(ctx, "omnivoice.higgs.acoustic_hop_length"));
        higgs.semantic_hidden_size = int(get_i(ctx, "omnivoice.higgs.semantic_hidden_size"));
        higgs.semantic_conv_dims = get_i_arr(ctx, "omnivoice.higgs.semantic_conv_dims");
        higgs.semantic_conv_kernels = get_i_arr(ctx, "omnivoice.higgs.semantic_conv_kernels");
        higgs.semantic_conv_strides = get_i_arr(ctx, "omnivoice.higgs.semantic_conv_strides");
        higgs.semantic_layer_norm_eps = get_f(ctx, "omnivoice.higgs.semantic_layer_norm_eps");
        higgs.semantic_intermediate_size = int(get_i(ctx, "omnivoice.higgs.semantic_intermediate_size"));
        higgs.semantic_head_count = int(get_i(ctx, "omnivoice.higgs.semantic_head_count"));
        higgs.semantic_block_count = int(get_i(ctx, "omnivoice.higgs.semantic_block_count"));
        higgs.semantic_pos_conv_embeddings = int(get_i(ctx, "omnivoice.higgs.semantic_pos_conv_embeddings"));
        higgs.semantic_pos_conv_groups = int(get_i(ctx, "omnivoice.higgs.semantic_pos_conv_groups"));
    }

    void load(const std::string & path) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = &weight_ctx;
        gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
        if (!gguf || !weight_ctx) throw std::runtime_error("failed to load GGUF: " + path);
        parse_specs(gguf);

        weight_buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend.backend);
        if (!weight_buf) throw std::runtime_error("failed to allocate GGUF weights on backend");

        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("failed to open GGUF weights: " + path);
        const int64_t n_tensors = gguf_get_n_tensors(gguf);
        const size_t data_offset = gguf_get_data_offset(gguf);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf, i);
            ggml_tensor * tensor = ggml_get_tensor(weight_ctx, name);
            if (!tensor) throw std::runtime_error(std::string("GGUF tensor not in context: ") + name);
            const size_t nbytes = ggml_nbytes(tensor);
            std::vector<uint8_t> data(nbytes);
            f.seekg(static_cast<std::streamoff>(data_offset + gguf_get_tensor_offset(gguf, i)));
            f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(nbytes));
            if (!f) throw std::runtime_error(std::string("failed to read tensor data: ") + name);
            ggml_backend_tensor_set(tensor, data.data(), 0, nbytes);
            tensors.emplace(name, tensor);
        }
        gguf_free(gguf);
    }
};

} // namespace

int HiggsSpec::hop_length() const {
    int x = 1;
    for (int v : acoustic_upsampling_ratios) x *= v;
    return x;
}

int HiggsSpec::frame_rate() const {
    return int(std::ceil(double(sample_rate) / double(hop_length())));
}

int HiggsSpec::semantic_head_dim() const {
    return semantic_hidden_size / semantic_head_count;
}

int HiggsSpec::semantic_downsample_factor() const {
    return int(double(hop_length()) / (double(sample_rate) / double(semantic_sample_rate)) / double(downsample_factor));
}

int HiggsSpec::num_quantizers() const {
    const int codebook_nbits = int(std::ceil(std::log2(double(codebook_size))));
    return int(1000.0 * double(target_bandwidths.back()) / (double(frame_rate()) * double(codebook_nbits)));
}

int HiggsSpec::num_quantizers_for_bandwidth(std::optional<float> bandwidth) const {
    if (!bandwidth) return num_quantizers();
    const double bw_per_q = std::log2(double(codebook_size)) * double(frame_rate()) / 1000.0;
    return std::max(1, int(std::floor(double(*bandwidth) / bw_per_q)));
}

int HiggsSpec::acoustic_encode_output_length(int input_length) const {
    int length = conv1d_out_len(input_length, 7, 1, 3);
    for (int stride : acoustic_downsampling_ratios) length = conv1d_out_len(length, 2 * stride, stride, int(std::ceil(stride / 2.0)));
    return conv1d_out_len(length, 3, 1, 1);
}

int HiggsSpec::semantic_encode_output_length(int input_length) const {
    int length = conv1d_out_len(input_length, kernel_size, 1, kernel_size / 2);
    for (int stride : strides) {
        int kernel = stride == 1 ? 3 : 2 * stride;
        length = conv1d_out_len(length, kernel, stride, (kernel - 1) / 2);
    }
    return length;
}

int HiggsSpec::semantic_feature_extractor_output_length(int input_length) const {
    int length = input_length;
    for (size_t i = 0; i < semantic_conv_kernels.size(); ++i) {
        length = conv1d_out_len(length, semantic_conv_kernels[i], semantic_conv_strides[i], 0);
    }
    return length;
}

namespace {

using Clock = std::chrono::steady_clock;

void emit_trace(
        const Model & model,
        TraceEventKind kind,
        const std::string & phase,
        const std::string & name,
        double seconds = 0.0,
        double audio_seconds = 0.0,
        int chunk_index = 0,
        int chunk_count = 0,
        int current = 0,
        int total = 0,
        int updated = 0,
        int total_positions = 0) {
    if (!model.trace) return;
    TraceEvent event;
    event.kind = kind;
    event.phase = phase;
    event.name = name;
    event.seconds = seconds;
    event.audio_seconds = audio_seconds;
    event.chunk_index = chunk_index;
    event.chunk_count = chunk_count;
    event.current = current;
    event.total = total;
    event.updated = updated;
    event.total_positions = total_positions;
    model.trace(event);
}

class ScopedStage {
public:
    ScopedStage(
            const Model & model,
            std::string phase,
            std::string name,
            int chunk_index = 0,
            int chunk_count = 0,
            double audio_seconds = 0.0)
        : model_(model),
          phase_(std::move(phase)),
          name_(std::move(name)),
          chunk_index_(chunk_index),
          chunk_count_(chunk_count),
          audio_seconds_(audio_seconds),
          start_(Clock::now()) {
        emit_trace(model_, TraceEventKind::StageBegin, phase_, name_, 0.0, audio_seconds_, chunk_index_, chunk_count_);
    }

    ~ScopedStage() {
        end();
    }

    void set_audio_seconds(double audio_seconds) {
        audio_seconds_ = audio_seconds;
    }

    void end() {
        if (ended_) return;
        ended_ = true;
        const double seconds = std::chrono::duration<double>(Clock::now() - start_).count();
        emit_trace(model_, TraceEventKind::StageEnd, phase_, name_, seconds, audio_seconds_, chunk_index_, chunk_count_);
    }

private:
    const Model & model_;
    std::string phase_;
    std::string name_;
    int chunk_index_ = 0;
    int chunk_count_ = 0;
    double audio_seconds_ = 0.0;
    Clock::time_point start_;
    bool ended_ = false;
};

ggml_tensor * named(ggml_tensor * t, const char * name) {
    ggml_set_name(t, name);
    return t;
}

ggml_tensor * broadcast_bias_2d(ggml_context * ctx, ggml_tensor * bias) {
    return ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
}

ggml_tensor * broadcast_bias_3d(ggml_context * ctx, ggml_tensor * bias) {
    return ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
}

ggml_tensor * broadcast_mask_1d_to_2d(ggml_context * ctx, ggml_tensor * mask) {
    return ggml_reshape_2d(ctx, mask, 1, mask->ne[0]);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * inp, ggml_tensor * bias) {
    ggml_tensor * out = ggml_mul_mat(ctx, weight, inp);
    if (bias) out = ggml_add(ctx, out, broadcast_bias_2d(ctx, bias));
    return out;
}

ggml_tensor * slice_3d_time(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    ggml_tensor * view = ggml_view_3d(ctx, tensor, length, tensor->ne[1], tensor->ne[2], tensor->nb[1], tensor->nb[2], start * ggml_element_size(tensor));
    return ggml_cont_3d(ctx, view, view->ne[0], view->ne[1], view->ne[2]);
}

ggml_tensor * slice_3d_channels(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    ggml_tensor * view = ggml_view_3d(ctx, tensor, tensor->ne[0], length, tensor->ne[2], tensor->nb[1], tensor->nb[2], start * tensor->nb[1]);
    return ggml_cont_3d(ctx, view, view->ne[0], view->ne[1], view->ne[2]);
}

ggml_tensor * slice_conv_out_channels(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    ggml_tensor * view = ggml_view_3d(ctx, tensor, tensor->ne[0], tensor->ne[1], length, tensor->nb[1], tensor->nb[2], start * tensor->nb[2]);
    return ggml_cont_3d(ctx, view, view->ne[0], view->ne[1], view->ne[2]);
}

ggml_tensor * slice_2d_tokens(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    return ggml_view_2d(ctx, tensor, tensor->ne[0], length, tensor->nb[1], start * tensor->nb[1]);
}

ggml_tensor * slice_3d_ne1(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    return ggml_view_3d(ctx, tensor, tensor->ne[0], length, tensor->ne[2], tensor->nb[1], tensor->nb[2], start * tensor->nb[1]);
}

ggml_tensor * conv_1d_raw(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * bias, ggml_tensor * inp, int stride, int padding, int dilation = 1) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, inp, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * a = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    ggml_tensor * b = ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    out = ggml_reshape_3d(ctx, out, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
    if (bias) out = ggml_add(ctx, out, broadcast_bias_3d(ctx, bias));
    return out;
}

ggml_tensor * conv_1d(ggml_context * ctx, const Model & model, const std::string & prefix, ggml_tensor * inp, int stride, int padding, int dilation = 1, bool bias = true) {
    return conv_1d_raw(ctx, model.t(prefix + ".weight"), bias ? model.t(prefix + ".bias") : nullptr, inp, stride, padding, dilation);
}

ggml_tensor * conv_transpose_1d(ggml_context * ctx, const Model & model, const std::string & prefix, ggml_tensor * inp, int stride, int padding, int output_padding) {
    ggml_tensor * full = ggml_conv_transpose_1d(ctx, model.t(prefix + ".weight"), inp, stride, 0, 1);
    const int64_t crop_left = padding;
    const int64_t crop_right = padding - output_padding;
    ggml_tensor * out = slice_3d_time(ctx, full, crop_left, full->ne[0] - crop_left - crop_right);
    return ggml_add(ctx, out, broadcast_bias_3d(ctx, model.t(prefix + ".bias")));
}

ggml_tensor * snake_1d(ggml_context * ctx, ggml_tensor * inp, ggml_tensor * alpha, ggml_tensor * one, ggml_tensor * eps) {
    ggml_tensor * alpha_full = ggml_repeat(ctx, alpha, inp);
    ggml_tensor * one_full = ggml_repeat(ctx, one, inp);
    ggml_tensor * eps_full = ggml_repeat(ctx, eps, inp);
    ggml_tensor * alpha_eps = ggml_add(ctx, alpha_full, eps_full);
    ggml_tensor * inv_alpha = ggml_div(ctx, one_full, alpha_eps);
    ggml_tensor * sin_sq = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, alpha_full, inp)));
    return ggml_add(ctx, inp, ggml_mul(ctx, inv_alpha, sin_sq));
}

ggml_tensor * residual_unit(ggml_context * ctx, const Model & model, const std::string & prefix, ggml_tensor * inp, ggml_tensor * one, ggml_tensor * eps, int dilation) {
    ggml_tensor * cur = snake_1d(ctx, inp, model.t(prefix + ".snake1.alpha"), one, eps);
    cur = conv_1d(ctx, model, prefix + ".conv1", cur, 1, ((7 - 1) * dilation) / 2, dilation);
    cur = snake_1d(ctx, cur, model.t(prefix + ".snake2.alpha"), one, eps);
    cur = conv_1d(ctx, model, prefix + ".conv2", cur, 1, 0);
    return ggml_add(ctx, inp, cur);
}

ggml_tensor * decoder_block(ggml_context * ctx, const Model & model, int i, ggml_tensor * inp, ggml_tensor * one, ggml_tensor * eps, int stride) {
    const std::string prefix = "a.ad.block." + std::to_string(i);
    ggml_tensor * cur = snake_1d(ctx, inp, model.t(prefix + ".snake1.alpha"), one, eps);
    cur = conv_transpose_1d(ctx, model, prefix + ".conv_t1", cur, stride, int(std::ceil(stride / 2.0)), stride % 2);
    cur = residual_unit(ctx, model, prefix + ".res_unit1", cur, one, eps, 1);
    cur = residual_unit(ctx, model, prefix + ".res_unit2", cur, one, eps, 3);
    cur = residual_unit(ctx, model, prefix + ".res_unit3", cur, one, eps, 9);
    return cur;
}

ggml_tensor * broadcast_mul_2d(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * target) {
    return ggml_mul(ctx, target, ggml_reshape_2d(ctx, weight, weight->ne[0], 1));
}

ggml_tensor * broadcast_mul_3d(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * target) {
    return ggml_mul(ctx, target, ggml_reshape_3d(ctx, weight, weight->ne[0], 1, 1));
}

ggml_tensor * broadcast_mul_channels_3d(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * target) {
    return ggml_mul(ctx, target, ggml_reshape_3d(ctx, weight, 1, weight->ne[0], 1));
}

ggml_tensor * rms_norm_2d(ggml_context * ctx, ggml_tensor * inp, ggml_tensor * weight, float eps) {
    return broadcast_mul_2d(ctx, weight, ggml_rms_norm(ctx, inp, eps));
}

ggml_tensor * rms_norm_3d(ggml_context * ctx, ggml_tensor * inp, ggml_tensor * weight, float eps) {
    return broadcast_mul_3d(ctx, weight, ggml_rms_norm(ctx, inp, eps));
}

ggml_tensor * norm_affine_2d(ggml_context * ctx, ggml_tensor * inp, ggml_tensor * weight, ggml_tensor * bias, float eps) {
    ggml_tensor * cur = ggml_norm(ctx, inp, eps);
    cur = broadcast_mul_2d(ctx, weight, cur);
    return ggml_add(ctx, cur, broadcast_bias_2d(ctx, bias));
}

ggml_tensor * channel_norm_affine_3d(ggml_context * ctx, ggml_tensor * inp, ggml_tensor * weight, ggml_tensor * bias, float eps) {
    ggml_tensor * cur = ggml_reshape_2d(ctx, inp, inp->ne[0], inp->ne[1] * inp->ne[2]);
    cur = ggml_norm(ctx, cur, eps);
    cur = ggml_reshape_3d(ctx, cur, inp->ne[0], inp->ne[1], inp->ne[2]);
    cur = broadcast_mul_channels_3d(ctx, weight, cur);
    return ggml_add(ctx, cur, broadcast_bias_3d(ctx, bias));
}

ggml_tensor * time_channel_3d_to_matrix(ggml_context * ctx, ggml_tensor * tensor) {
    ggml_tensor * matrix = ggml_reshape_2d(ctx, tensor, tensor->ne[0], tensor->ne[1] * tensor->ne[2]);
    matrix = ggml_transpose(ctx, matrix);
    return ggml_cont_2d(ctx, matrix, matrix->ne[0], matrix->ne[1]);
}

ggml_tensor * matrix_to_time_channel_3d(ggml_context * ctx, ggml_tensor * tensor) {
    ggml_tensor * transposed = ggml_transpose(ctx, tensor);
    transposed = ggml_cont_2d(ctx, transposed, transposed->ne[0], transposed->ne[1]);
    return ggml_reshape_3d(ctx, transposed, transposed->ne[0], transposed->ne[1], 1);
}

ggml_tensor * slice_2d_columns(ggml_context * ctx, ggml_tensor * tensor, int step) {
    const int64_t length = (tensor->ne[1] + step - 1) / step;
    ggml_tensor * view = ggml_view_2d(ctx, tensor, tensor->ne[0], length, tensor->nb[1] * step, 0);
    return ggml_cont_2d(ctx, view, view->ne[0], view->ne[1]);
}

ggml_tensor * grouped_conv_1d_raw(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias, ggml_tensor * inp, int stride, int padding, int dilation, int groups) {
    const int64_t in_per_group = inp->ne[1] / groups;
    const int64_t out_per_group = weight->ne[2] / groups;
    ggml_tensor * out = nullptr;
    for (int group = 0; group < groups; ++group) {
        ggml_tensor * inp_group = slice_3d_channels(ctx, inp, group * in_per_group, in_per_group);
        ggml_tensor * weight_group = slice_conv_out_channels(ctx, weight, group * out_per_group, out_per_group);
        ggml_tensor * cur = conv_1d_raw(ctx, weight_group, nullptr, inp_group, stride, padding, dilation);
        out = out ? ggml_concat(ctx, out, cur, 1) : cur;
    }
    if (bias) out = ggml_add(ctx, out, broadcast_bias_3d(ctx, bias));
    return out;
}

ggml_tensor * grouped_conv_1d(ggml_context * ctx, const Model & model, const std::string & weight_name, ggml_tensor * bias, ggml_tensor * inp, int stride, int padding, int dilation, int groups) {
    return grouped_conv_1d_raw(ctx, model.t(weight_name), bias, inp, stride, padding, dilation, groups);
}

ggml_tensor * semantic_residual_unit(ggml_context * ctx, const Model & model, const std::string & prefix, ggml_tensor * inp, int dilation) {
    const int64_t kernel = model.t(prefix + ".conv1.weight")->ne[0];
    ggml_tensor * cur = ggml_elu(ctx, inp);
    cur = conv_1d(ctx, model, prefix + ".conv1", cur, 1, int(((kernel - 1) * dilation) / 2), dilation, false);
    cur = ggml_elu(ctx, cur);
    cur = conv_1d(ctx, model, prefix + ".conv2", cur, 1, 0, 1, false);
    return ggml_add(ctx, inp, cur);
}

ggml_tensor * semantic_encoder_block(ggml_context * ctx, const Model & model, int i, ggml_tensor * inp, int stride) {
    const std::string prefix = "a.se.conv_blocks." + std::to_string(i);
    ggml_tensor * cur = inp;
    for (size_t j = 0; j < model.higgs.block_dilations.size(); ++j) {
        cur = semantic_residual_unit(ctx, model, prefix + ".res_units." + std::to_string(j), cur, model.higgs.block_dilations[j]);
    }
    const int kernel = stride == 1 ? 3 : 2 * stride;
    return conv_1d(ctx, model, prefix + ".conv", cur, stride, (kernel - 1) / 2);
}

ggml_tensor * semantic_encoder(ggml_context * ctx, const Model & model, ggml_tensor * inp) {
    ggml_tensor * cur = conv_1d(ctx, model, "a.se.conv", inp, 1, model.higgs.kernel_size / 2, 1, false);
    for (size_t i = 0; i < model.higgs.strides.size(); ++i) {
        cur = semantic_encoder_block(ctx, model, int(i), cur, model.higgs.strides[i]);
    }
    return cur;
}

ggml_tensor * encoder_block(ggml_context * ctx, const Model & model, int i, ggml_tensor * inp, ggml_tensor * one, ggml_tensor * eps, int stride) {
    const std::string prefix = "a.ae.block." + std::to_string(i);
    ggml_tensor * cur = residual_unit(ctx, model, prefix + ".res_unit1", inp, one, eps, 1);
    cur = residual_unit(ctx, model, prefix + ".res_unit2", cur, one, eps, 3);
    cur = residual_unit(ctx, model, prefix + ".res_unit3", cur, one, eps, 9);
    cur = snake_1d(ctx, cur, model.t(prefix + ".snake1.alpha"), one, eps);
    return conv_1d(ctx, model, prefix + ".conv1", cur, stride, int(std::ceil(stride / 2.0)));
}

ggml_tensor * acoustic_encoder(ggml_context * ctx, const Model & model, ggml_tensor * inp, ggml_tensor * one, ggml_tensor * eps) {
    ggml_tensor * cur = conv_1d(ctx, model, "a.ae.conv1", inp, 1, 3);
    for (size_t i = 0; i < model.higgs.acoustic_downsampling_ratios.size(); ++i) {
        cur = encoder_block(ctx, model, int(i), cur, one, eps, model.higgs.acoustic_downsampling_ratios[i]);
    }
    cur = snake_1d(ctx, cur, model.t("a.ae.snake1.alpha"), one, eps);
    return conv_1d(ctx, model, "a.ae.conv2", cur, 1, 1);
}

ggml_tensor * hubert_feature_extractor(ggml_context * ctx, const Model & model, ggml_tensor * inp) {
    ggml_tensor * cur = inp;
    for (size_t i = 0; i < model.higgs.semantic_conv_dims.size(); ++i) {
        const std::string prefix = "a.sm.fe.conv_layers." + std::to_string(i);
        cur = conv_1d(ctx, model, prefix + ".conv", cur, model.higgs.semantic_conv_strides[i], 0, 1, false);
        if (i == 0) {
            cur = channel_norm_affine_3d(
                ctx,
                cur,
                model.t(prefix + ".layer_norm.weight"),
                model.t(prefix + ".layer_norm.bias"),
                model.higgs.semantic_layer_norm_eps);
        }
        cur = ggml_gelu(ctx, cur);
    }
    return cur;
}

ggml_tensor * hubert_feature_projection(ggml_context * ctx, const Model & model, ggml_tensor * inp) {
    ggml_tensor * cur = time_channel_3d_to_matrix(ctx, inp);
    cur = norm_affine_2d(ctx, cur, model.t("a.sm.fp.layer_norm.weight"), model.t("a.sm.fp.layer_norm.bias"), model.higgs.semantic_layer_norm_eps);
    return linear(ctx, model.t("a.sm.fp.projection.weight"), cur, model.t("a.sm.fp.projection.bias"));
}

ggml_tensor * hubert_positional_conv(ggml_context * ctx, const Model & model, ggml_tensor * inp, ggml_tensor * weight) {
    ggml_tensor * cur = grouped_conv_1d_raw(
        ctx,
        weight,
        model.t("a.sm.encoder.pce.conv.bias"),
        inp,
        1,
        model.higgs.semantic_pos_conv_embeddings / 2,
        1,
        model.higgs.semantic_pos_conv_groups);
    if (model.higgs.semantic_pos_conv_embeddings % 2 == 0) cur = slice_3d_time(ctx, cur, 0, cur->ne[0] - 1);
    return ggml_gelu(ctx, cur);
}

ggml_tensor * as_4d(ggml_context * ctx, ggml_tensor * tensor);

ggml_tensor * hubert_attention(ggml_context * ctx, const Model & model, int i, ggml_tensor * hidden) {
    const std::string prefix = "a.sm.encoder.layers." + std::to_string(i);
    const int n_tokens = int(hidden->ne[1]);
    const int head_dim = model.higgs.semantic_head_dim();
    const int n_head = model.higgs.semantic_head_count;
    ggml_tensor * q = linear(ctx, model.t(prefix + ".attention.q_proj.weight"), hidden, model.t(prefix + ".attention.q_proj.bias"));
    ggml_tensor * k = linear(ctx, model.t(prefix + ".attention.k_proj.weight"), hidden, model.t(prefix + ".attention.k_proj.bias"));
    ggml_tensor * v = linear(ctx, model.t(prefix + ".attention.v_proj.weight"), hidden, model.t(prefix + ".attention.v_proj.bias"));
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_head, n_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_head, n_tokens);
    ggml_tensor * q4 = ggml_permute(ctx, as_4d(ctx, q), 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(ctx, as_4d(ctx, k), 0, 2, 1, 3);
    ggml_tensor * v4 = ggml_permute(ctx, as_4d(ctx, v), 0, 2, 1, 3);
    ggml_tensor * attn = ggml_mul_mat(ctx, k4, q4);
    attn = ggml_scale(ctx, attn, 1.0f / std::sqrt(float(head_dim)));
    attn = ggml_soft_max(ctx, attn);
    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_transpose(ctx, v4));
    attn = ggml_mul_mat(ctx, v_for_mm, attn);
    attn = ggml_permute(ctx, attn, 0, 2, 1, 3);
    attn = ggml_cont_2d(ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]);
    return linear(ctx, model.t(prefix + ".attention.out_proj.weight"), attn, model.t(prefix + ".attention.out_proj.bias"));
}

ggml_tensor * hubert_feed_forward(ggml_context * ctx, const Model & model, int i, ggml_tensor * hidden) {
    const std::string prefix = "a.sm.encoder.layers." + std::to_string(i) + ".ff";
    ggml_tensor * cur = linear(ctx, model.t(prefix + ".int.weight"), hidden, model.t(prefix + ".int.bias"));
    cur = ggml_gelu(ctx, cur);
    return linear(ctx, model.t(prefix + ".out.weight"), cur, model.t(prefix + ".out.bias"));
}

ggml_tensor * hubert_encoder_layer(ggml_context * ctx, const Model & model, int i, ggml_tensor * hidden) {
    const std::string prefix = "a.sm.encoder.layers." + std::to_string(i);
    ggml_tensor * cur = ggml_add(ctx, hidden, hubert_attention(ctx, model, i, hidden));
    cur = norm_affine_2d(ctx, cur, model.t(prefix + ".layer_norm.weight"), model.t(prefix + ".layer_norm.bias"), model.higgs.semantic_layer_norm_eps);
    cur = ggml_add(ctx, cur, hubert_feed_forward(ctx, model, i, cur));
    return norm_affine_2d(ctx, cur, model.t(prefix + ".final_layer_norm.weight"), model.t(prefix + ".final_layer_norm.bias"), model.higgs.semantic_layer_norm_eps);
}

ggml_tensor * encode_quantizer(ggml_context * ctx, const Model & model, int i, ggml_tensor * residual, ggml_tensor ** codes_out) {
    const std::string prefix = "a.q.quantizers." + std::to_string(i);
    ggml_tensor * projected = linear(ctx, model.t(prefix + ".project_in.weight"), residual, model.t(prefix + ".project_in.bias"));
    ggml_tensor * similarity = ggml_scale(ctx, ggml_mul_mat(ctx, model.t(prefix + ".codebook.embed"), projected), 2.0f);
    ggml_tensor * projected_sq = ggml_sum_rows(ctx, ggml_sqr(ctx, projected));
    similarity = ggml_sub(ctx, similarity, ggml_repeat(ctx, projected_sq, similarity));
    ggml_tensor * codebook_sq = ggml_sum_rows(ctx, ggml_sqr(ctx, model.t(prefix + ".codebook.embed")));
    codebook_sq = ggml_transpose(ctx, codebook_sq);
    codebook_sq = ggml_cont_2d(ctx, codebook_sq, codebook_sq->ne[0], codebook_sq->ne[1]);
    similarity = ggml_sub(ctx, similarity, ggml_repeat(ctx, codebook_sq, similarity));
    ggml_tensor * codes = ggml_argmax(ctx, similarity);
    if (codes_out) *codes_out = codes;
    ggml_tensor * quantized = ggml_get_rows(ctx, model.t(prefix + ".codebook.embed"), codes);
    return linear(ctx, model.t(prefix + ".project_out.weight"), quantized, model.t(prefix + ".project_out.bias"));
}

ggml_tensor * as_4d(ggml_context * ctx, ggml_tensor * tensor) {
    if (ggml_is_contiguous(tensor)) return ggml_reshape_4d(ctx, tensor, tensor->ne[0], tensor->ne[1], tensor->ne[2], 1);
    return ggml_view_4d(ctx, tensor, tensor->ne[0], tensor->ne[1], tensor->ne[2], 1, tensor->nb[1], tensor->nb[2], tensor->nb[2], 0);
}

ggml_tensor * repeat_kv_3d(ggml_context * ctx, ggml_tensor * tensor, int repeats) {
    if (repeats == 1) return tensor;
    ggml_tensor * viewed = ggml_is_contiguous(tensor)
        ? ggml_reshape_4d(ctx, tensor, tensor->ne[0], 1, tensor->ne[1], tensor->ne[2])
        : ggml_view_4d(ctx, tensor, tensor->ne[0], 1, tensor->ne[1], tensor->ne[2], tensor->nb[1], tensor->nb[1], tensor->nb[2], 0);
    ggml_tensor * repeated = ggml_repeat_4d(ctx, viewed, viewed->ne[0], repeats, viewed->ne[2], viewed->ne[3]);
    if (!ggml_is_contiguous(repeated)) repeated = ggml_cont_4d(ctx, repeated, repeated->ne[0], repeated->ne[1], repeated->ne[2], repeated->ne[3]);
    return ggml_reshape_3d(ctx, repeated, repeated->ne[0], repeated->ne[1] * repeated->ne[2], repeated->ne[3]);
}

ggml_tensor * qwen3_attention(ggml_context * ctx, const Model & model, int block, ggml_tensor * hidden, ggml_tensor * positions, ggml_tensor * kq_mask) {
    const auto & spec = model.spec;
    const std::string p = "blk." + std::to_string(block);
    const int n_tokens = int(hidden->ne[1]);
    const int head_dim = spec.qwen3.head_dim;
    const int n_head = spec.qwen3.head_count;
    const int n_head_kv = spec.qwen3.head_count_kv;
    const int n_rep = n_head / n_head_kv;
    const float scale = 1.0f / std::sqrt(float(head_dim));

    ggml_tensor * normed = rms_norm_2d(ctx, hidden, model.t(p + ".attn_norm.weight"), spec.qwen3.rms_norm_eps);
    ggml_tensor * q = ggml_mul_mat(ctx, model.t(p + ".attn_q.weight"), normed);
    ggml_tensor * k = ggml_mul_mat(ctx, model.t(p + ".attn_k.weight"), normed);
    ggml_tensor * v = ggml_mul_mat(ctx, model.t(p + ".attn_v.weight"), normed);
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, n_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, n_head_kv, n_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, n_head_kv, n_tokens);
    q = rms_norm_3d(ctx, q, model.t(p + ".attn_q_norm.weight"), spec.qwen3.rms_norm_eps);
    q = ggml_rope_ext(ctx, q, positions, nullptr, head_dim, QWEN3_ROPE_MODE_NEOX, spec.qwen3.context_length, spec.qwen3.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = rms_norm_3d(ctx, k, model.t(p + ".attn_k_norm.weight"), spec.qwen3.rms_norm_eps);
    k = ggml_rope_ext(ctx, k, positions, nullptr, head_dim, QWEN3_ROPE_MODE_NEOX, spec.qwen3.context_length, spec.qwen3.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = repeat_kv_3d(ctx, k, n_rep);
    v = repeat_kv_3d(ctx, v, n_rep);
    ggml_tensor * q4 = ggml_permute(ctx, as_4d(ctx, q), 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(ctx, as_4d(ctx, k), 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(ctx, k4, q4);
    kq = ggml_soft_max_ext(ctx, kq, kq_mask, scale, 0.0f);
    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, as_4d(ctx, v), 1, 2, 0, 3));
    ggml_tensor * kqv = ggml_mul_mat(ctx, v_for_mm, kq);
    ggml_tensor * attn = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    attn = ggml_cont_2d(ctx, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]);
    return ggml_mul_mat(ctx, model.t(p + ".attn_output.weight"), attn);
}

ggml_tensor * qwen3_mlp(ggml_context * ctx, const Model & model, int block, ggml_tensor * hidden) {
    const auto & spec = model.spec;
    const std::string p = "blk." + std::to_string(block);
    ggml_tensor * normed = rms_norm_2d(ctx, hidden, model.t(p + ".ffn_norm.weight"), spec.qwen3.rms_norm_eps);
    ggml_tensor * gate = ggml_mul_mat(ctx, model.t(p + ".ffn_gate.weight"), normed);
    ggml_tensor * up = ggml_mul_mat(ctx, model.t(p + ".ffn_up.weight"), normed);
    gate = ggml_silu(ctx, gate);
    ggml_tensor * gated = ggml_mul(ctx, gate, up);
    return ggml_mul_mat(ctx, model.t(p + ".ffn_down.weight"), gated);
}

ggml_tensor * qwen3_block(ggml_context * ctx, const Model & model, int block, ggml_tensor * hidden, ggml_tensor * positions, ggml_tensor * kq_mask) {
    ggml_tensor * attn = qwen3_attention(ctx, model, block, hidden, positions, kq_mask);
    ggml_tensor * cur = ggml_add(ctx, hidden, attn);
    ggml_tensor * mlp = qwen3_mlp(ctx, model, block, cur);
    return ggml_add(ctx, cur, mlp);
}

ggml_tensor * candidate_log_softmax(ggml_context * ctx, ggml_tensor * logits, int n_vocab) {
    ggml_tensor * max_logits = ggml_pool_2d(ctx, logits, GGML_OP_POOL_MAX, n_vocab, 1, n_vocab, 1, 0.0f, 0.0f);
    ggml_tensor * shifted = ggml_sub(ctx, logits, max_logits);
    ggml_tensor * sum_exp = ggml_sum_rows(ctx, ggml_exp(ctx, shifted));
    return ggml_sub(ctx, shifted, ggml_log(ctx, sum_exp));
}

std::vector<int32_t> full_text_ids(const InferenceInputs & in) {
    std::vector<int32_t> out(static_cast<size_t>(in.total_length()), 0);
    for (int i = 0; i < in.audio_start(); ++i) out[static_cast<size_t>(i)] = in.input_ids(0, i);
    return out;
}

std::vector<int32_t> full_shifted_audio_ids(const OmniVoiceSpec & spec, const InferenceInputs & in, int codebook) {
    std::vector<int32_t> out(static_cast<size_t>(in.total_length()), 0);
    const int offset = codebook * spec.audio_vocab_size;
    for (int t = in.audio_start(); t < in.total_length(); ++t) {
        out[static_cast<size_t>(t)] = in.input_ids(codebook, t) + offset;
    }
    return out;
}

std::vector<float> full_kq_mask(int n) {
    return std::vector<float>(static_cast<size_t>(n) * n, 1.0f);
}

std::vector<int32_t> arange_i32(int n) {
    std::vector<int32_t> out(static_cast<size_t>(n));
    std::iota(out.begin(), out.end(), 0);
    return out;
}

void tensor_set(ggml_tensor * tensor, const std::vector<int32_t> & data) {
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(int32_t));
}

void tensor_set(ggml_tensor * tensor, const std::vector<float> & data) {
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
}

struct SplitCandidateGraph {
    std::unique_ptr<GraphHandle> gh;
    ggml_tensor * cond_text = nullptr;
    ggml_tensor * uncond_text = nullptr;
    std::vector<ggml_tensor *> cond_audio;
    std::vector<ggml_tensor *> uncond_audio;
    ggml_tensor * cond_mask = nullptr;
    ggml_tensor * uncond_mask = nullptr;
    ggml_tensor * cond_pos = nullptr;
    ggml_tensor * uncond_pos = nullptr;
    ggml_tensor * cond_kq_mask = nullptr;
    ggml_tensor * uncond_kq_mask = nullptr;
    ggml_tensor * pred = nullptr;
    ggml_tensor * scores = nullptr;
};

std::unique_ptr<SplitCandidateGraph> build_split_candidate_graph(
    const Model & model,
    int cond_total,
    int cond_target_start,
    int target_length,
    int uncond_total,
    const GenerationConfig & config) {
    auto out = std::make_unique<SplitCandidateGraph>();
    out->gh = std::make_unique<GraphHandle>(model.backend.buft, 262144, 192ull * 1024ull * 1024ull);
    ggml_context * ctx = out->gh->ctx;
    const auto & spec = model.spec;

    auto build_branch = [&](const char * prefix, int total, int target_start, ggml_tensor ** text, std::vector<ggml_tensor *> & audio, ggml_tensor ** mask, ggml_tensor ** pos, ggml_tensor ** kq_mask) {
        *text = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total);
        ggml_set_input(*text);
        audio.clear();
        for (int i = 0; i < spec.num_audio_codebook; ++i) {
            ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total);
            ggml_set_input(a);
            audio.push_back(a);
        }
        *mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, total);
        *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total);
        *kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, total, total, 1, 1);
        ggml_set_input(*mask);
        ggml_set_input(*pos);
        ggml_set_input(*kq_mask);

        ggml_tensor * text_emb = ggml_get_rows(ctx, model.t("token_embd.weight"), *text);
        ggml_tensor * audio_emb = nullptr;
        for (ggml_tensor * a : audio) {
            ggml_tensor * e = ggml_get_rows(ctx, model.t("a.token_embd"), a);
            audio_emb = audio_emb ? ggml_add(ctx, audio_emb, e) : e;
        }
        ggml_tensor * mask2d = broadcast_mask_1d_to_2d(ctx, *mask);
        ggml_tensor * cur = ggml_add(ctx, text_emb, ggml_mul(ctx, ggml_sub(ctx, audio_emb, text_emb), mask2d));
        for (int i = 0; i < spec.qwen3.block_count; ++i) cur = qwen3_block(ctx, model, i, cur, *pos, *kq_mask);
        cur = rms_norm_2d(ctx, cur, model.t("output_norm.weight"), spec.qwen3.rms_norm_eps);
        cur = slice_2d_tokens(ctx, cur, target_start, target_length);
        ggml_tensor * logits_flat = ggml_mul_mat(ctx, model.t("a.output"), cur);
        return ggml_reshape_3d(ctx, logits_flat, spec.audio_vocab_size, spec.num_audio_codebook, target_length);
    };

    ggml_tensor * cond_out = build_branch("cond", cond_total, cond_target_start, &out->cond_text, out->cond_audio, &out->cond_mask, &out->cond_pos, &out->cond_kq_mask);
    ggml_tensor * uncond_out = build_branch("uncond", uncond_total, 0, &out->uncond_text, out->uncond_audio, &out->uncond_mask, &out->uncond_pos, &out->uncond_kq_mask);

    const int n_vocab = spec.audio_vocab_size;
    const int n_positions = spec.num_audio_codebook * target_length;
    ggml_tensor * cond_flat = ggml_reshape_2d(ctx, cond_out, n_vocab, n_positions);
    ggml_tensor * uncond_flat = ggml_reshape_2d(ctx, uncond_out, n_vocab, n_positions);
    ggml_tensor * cond_log = candidate_log_softmax(ctx, cond_flat, n_vocab);
    ggml_tensor * guided = cond_log;
    if (config.guidance_scale != 0.0f) {
        ggml_tensor * uncond_log = candidate_log_softmax(ctx, uncond_flat, n_vocab);
        guided = ggml_add(ctx, cond_log, ggml_scale(ctx, ggml_sub(ctx, cond_log, uncond_log), config.guidance_scale));
        guided = candidate_log_softmax(ctx, guided, n_vocab);
    }

    std::vector<float> mask_data(static_cast<size_t>(n_vocab) * n_positions, 0.0f);
    for (int p = 0; p < n_positions; ++p) mask_data[static_cast<size_t>(p) * n_vocab + spec.audio_mask_id] = -INFINITY;
    ggml_tensor * softmax_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_positions);
    ggml_set_input(softmax_mask);
    ggml_tensor * masked = ggml_add(ctx, guided, softmax_mask);
    out->pred = ggml_argmax(ctx, masked);
    out->scores = ggml_pool_2d(ctx, masked, GGML_OP_POOL_MAX, n_vocab, 1, n_vocab, 1, 0.0f, 0.0f);
    out->scores = ggml_reshape_1d(ctx, out->scores, n_positions);
    ggml_set_output(out->pred);
    ggml_set_output(out->scores);
    ggml_build_forward_expand(out->gh->graph, out->pred);
    ggml_build_forward_expand(out->gh->graph, out->scores);
    out->gh->reserve_alloc();
    ggml_backend_tensor_set(softmax_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    return out;
}

void upload_branch(const Model & model, SplitCandidateGraph & graph, const InferenceInputs & inputs, bool cond) {
    ggml_tensor * text = cond ? graph.cond_text : graph.uncond_text;
    ggml_tensor * mask = cond ? graph.cond_mask : graph.uncond_mask;
    ggml_tensor * pos = cond ? graph.cond_pos : graph.uncond_pos;
    ggml_tensor * kq = cond ? graph.cond_kq_mask : graph.uncond_kq_mask;
    std::vector<ggml_tensor *> & audio = cond ? graph.cond_audio : graph.uncond_audio;
    tensor_set(text, full_text_ids(inputs));
    tensor_set(mask, inputs.audio_mask);
    tensor_set(pos, arange_i32(inputs.total_length()));
    tensor_set(kq, full_kq_mask(inputs.total_length()));
    for (int i = 0; i < model.spec.num_audio_codebook; ++i) tensor_set(audio[static_cast<size_t>(i)], full_shifted_audio_ids(model.spec, inputs, i));
}

struct DecodeGraph {
    std::unique_ptr<GraphHandle> gh;
    std::vector<ggml_tensor *> code_inputs;
    ggml_tensor * output = nullptr;
};

std::unique_ptr<DecodeGraph> build_decode_graph(const Model & model, int code_length) {
    auto out = std::make_unique<DecodeGraph>();
    out->gh = std::make_unique<GraphHandle>(model.backend.buft, 32768, 96ull * 1024ull * 1024ull);
    ggml_context * ctx = out->gh->ctx;
    ggml_tensor * one = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 1);
    ggml_tensor * eps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 1);
    ggml_set_input(one);
    ggml_set_input(eps);
    out->code_inputs.reserve(model.spec.num_audio_codebook);
    ggml_tensor * quantized = nullptr;
    for (int i = 0; i < model.spec.num_audio_codebook; ++i) {
        const std::string p = "a.q.quantizers." + std::to_string(i);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, code_length);
        ggml_set_input(ids);
        out->code_inputs.push_back(ids);
        ggml_tensor * embed = ggml_get_rows(ctx, model.t(p + ".codebook.embed"), ids);
        ggml_tensor * proj = linear(ctx, model.t(p + ".project_out.weight"), embed, model.t(p + ".project_out.bias"));
        quantized = quantized ? ggml_add(ctx, quantized, proj) : proj;
    }
    ggml_tensor * fc2 = linear(ctx, model.t("a.fc2.weight"), quantized, model.t("a.fc2.bias"));
    ggml_tensor * cur = ggml_transpose(ctx, fc2);
    cur = ggml_cont_2d(ctx, cur, cur->ne[0], cur->ne[1]);
    cur = ggml_reshape_3d(ctx, cur, code_length, fc2->ne[0], 1);
    cur = conv_1d(ctx, model, "a.ad.conv1", cur, 1, 3);
    for (size_t i = 0; i < model.higgs.acoustic_upsampling_ratios.size(); ++i) {
        cur = decoder_block(ctx, model, int(i), cur, one, eps, model.higgs.acoustic_upsampling_ratios[i]);
    }
    cur = snake_1d(ctx, cur, model.t("a.ad.snake1.alpha"), one, eps);
    out->output = conv_1d(ctx, model, "a.ad.conv2", cur, 1, 3);
    ggml_set_output(out->output);
    ggml_build_forward_expand(out->gh->graph, out->output);
    out->gh->reserve_alloc();
    const float one_v = 1.0f;
    const float eps_v = 1.0e-9f;
    ggml_backend_tensor_set(one, &one_v, 0, sizeof(float));
    ggml_backend_tensor_set(eps, &eps_v, 0, sizeof(float));
    return out;
}

InferenceInputs prepare_unconditional(const Model & model, const Tensor2i & tokens) {
    InferenceInputs out;
    out.input_ids = tokens;
    out.audio_mask.assign(static_cast<size_t>(tokens.cols), 1.0f);
    out.target_length = tokens.cols;
    return out;
}

Tensor2i target_tokens(const InferenceInputs & in) {
    Tensor2i out{in.input_ids.rows, in.target_length, {}};
    out.data.resize(static_cast<size_t>(out.rows) * out.cols);
    for (int c = 0; c < out.rows; ++c) {
        for (int t = 0; t < out.cols; ++t) out(c, t) = in.input_ids(c, in.target_start() + t);
    }
    return out;
}

InferenceInputs with_target_tokens(const InferenceInputs & in, const Tensor2i & tokens) {
    InferenceInputs out = in;
    for (int c = 0; c < tokens.rows; ++c) {
        for (int t = 0; t < tokens.cols; ++t) out.input_ids(c, in.target_start() + t) = tokens(c, t);
    }
    return out;
}

std::vector<int> build_schedule(const OmniVoiceSpec & spec, int target_length, const GenerationConfig & config) {
    const int total = target_length * spec.num_audio_codebook;
    int rem = total;
    std::vector<int> schedule;
    schedule.reserve(config.num_step);
    auto timestep = [&](int i) {
        const float x = float(i) / float(config.num_step + 1);
        return config.t_shift * x / (1.0f + (config.t_shift - 1.0f) * x);
    };
    for (int step = 0; step < config.num_step; ++step) {
        int n;
        if (step == config.num_step - 1) n = rem;
        else n = std::min(int(std::ceil(total * (timestep(step + 1) - timestep(step)))), rem);
        schedule.push_back(n);
        rem -= n;
    }
    return schedule;
}

InferenceInputs prepare_inputs(const Model & model, const std::string & text, int target_tokens_count, const std::string & lang, const std::string & instruct, const VoiceClonePrompt * prompt, bool denoise) {
    std::string style;
    if (denoise && prompt) style += "<|denoise|>";
    style += "<|lang_start|>" + (lang.empty() ? std::string("None") : lang) + "<|lang_end|>";
    style += "<|instruct_start|>" + (instruct.empty() ? std::string("None") : instruct) + "<|instruct_end|>";
    std::string full_text = text;
    if (prompt && !prompt->ref_text.empty()) full_text = prompt->ref_text + " " + text;
    full_text.erase(std::remove(full_text.begin(), full_text.end(), '\n'), full_text.end());
    full_text.erase(std::remove(full_text.begin(), full_text.end(), '\r'), full_text.end());
    std::string wrapped = "<|text_start|>" + full_text + "<|text_end|>";
    auto style_ids = model.tokenizer.encode(style, true);
    auto text_ids = model.tokenizer.encode(wrapped, false);
    const int C = model.spec.num_audio_codebook;
    const int ref_len = prompt ? prompt->ref_audio_tokens.cols : 0;
    const int total = int(style_ids.size() + text_ids.size()) + ref_len + target_tokens_count;
    InferenceInputs out;
    out.input_ids = Tensor2i{C, total, std::vector<int32_t>(static_cast<size_t>(C) * total, model.spec.audio_mask_id)};
    out.audio_mask.assign(static_cast<size_t>(total), 0.0f);
    out.style_length = int(style_ids.size());
    out.text_length = int(text_ids.size());
    out.ref_audio_length = ref_len;
    out.target_length = target_tokens_count;
    for (int c = 0; c < C; ++c) {
        for (size_t i = 0; i < style_ids.size(); ++i) out.input_ids(c, int(i)) = style_ids[i];
        for (size_t i = 0; i < text_ids.size(); ++i) out.input_ids(c, int(style_ids.size() + i)) = text_ids[i];
        if (prompt) {
            for (int t = 0; t < ref_len; ++t) out.input_ids(c, int(style_ids.size() + text_ids.size()) + t) = prompt->ref_audio_tokens(c, t);
        }
    }
    for (int i = out.audio_start(); i < total; ++i) out.audio_mask[static_cast<size_t>(i)] = 1.0f;
    return out;
}

int estimate_target_tokens(const Model & model, const std::string & text, float speed, std::optional<float> duration, const VoiceClonePrompt * prompt) {
    if (duration) return std::max(1, int(*duration * model.higgs.frame_rate()));
    std::string ref_text = "Nice to meet you.";
    int ref_tokens = 25;
    if (prompt && prompt->ref_audio_tokens.cols > 0 && !prompt->ref_text.empty()) {
        ref_text = prompt->ref_text;
        ref_tokens = prompt->ref_audio_tokens.cols;
    }
    const float ref_weight = text_duration_weight(ref_text);
    float est = 0.0f;
    if (ref_tokens > 0 && ref_weight > 0.0f) {
        est = text_duration_weight(text) / (ref_weight / float(ref_tokens));
        if (est > 0.0f && est < 50.0f) est = 50.0f * std::pow(est / 50.0f, 1.0f / 3.0f);
    }
    if (speed > 0.0f && speed != 1.0f) est = est / speed;
    return std::max(1, int(est));
}

Tensor2i generate_codes(Model & model, const InferenceInputs & inputs, const GenerationConfig & config, int chunk_index, int chunk_count) {
    ScopedStage stage(model, "llm", "llm_decode", chunk_index, chunk_count, double(inputs.target_length) / double(model.higgs.frame_rate()));
    Tensor2i tokens = target_tokens(inputs);
    std::mt19937_64 rng(config.seed.value_or(0));
    const auto schedule = build_schedule(model.spec, inputs.target_length, config);
    std::unique_ptr<SplitCandidateGraph> graph;
    const int total_positions = model.spec.num_audio_codebook * inputs.target_length;
    int updated_positions = 0;
    for (size_t step = 0; step < schedule.size(); ++step) {
        const int update_count = schedule[step];
        if (update_count <= 0) {
            emit_trace(
                model,
                TraceEventKind::LlmProgress,
                "llm",
                "llm_decode",
                0.0,
                double(inputs.target_length) / double(model.higgs.frame_rate()),
                chunk_index,
                chunk_count,
                int(step + 1),
                int(schedule.size()),
                updated_positions,
                total_positions);
            continue;
        }
        InferenceInputs cond = with_target_tokens(inputs, tokens);
        InferenceInputs uncond = prepare_unconditional(model, tokens);
        if (!graph) {
            graph = build_split_candidate_graph(model, cond.total_length(), cond.target_start(), cond.target_length, uncond.total_length(), config);
        }
        upload_branch(model, *graph, cond, true);
        upload_branch(model, *graph, uncond, false);
        const ggml_status st = ggml_backend_graph_compute(model.backend.backend, graph->gh->graph);
        if (st != GGML_STATUS_SUCCESS) throw std::runtime_error("split candidate graph compute failed");
        const int positions = model.spec.num_audio_codebook * cond.target_length;
        std::vector<int32_t> pred_flat(static_cast<size_t>(positions));
        std::vector<float> score_flat(static_cast<size_t>(positions));
        ggml_backend_tensor_get(graph->pred, pred_flat.data(), 0, pred_flat.size() * sizeof(int32_t));
        ggml_backend_tensor_get(graph->scores, score_flat.data(), 0, score_flat.size() * sizeof(float));

        std::vector<int32_t> pred(static_cast<size_t>(positions));
        std::vector<float> scores(static_cast<size_t>(positions));
        for (int t = 0; t < cond.target_length; ++t) {
            for (int c = 0; c < model.spec.num_audio_codebook; ++c) {
                const int src = t * model.spec.num_audio_codebook + c;
                const int dst = c * cond.target_length + t;
                pred[dst] = pred_flat[src];
                scores[dst] = score_flat[src] - float(c) * config.layer_penalty_factor;
            }
        }
        if (config.position_temperature > 0.0f) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (float & s : scores) {
                const float u = std::max(1.0e-10f, dist(rng));
                const float g = -std::log(-std::log(u + 1.0e-10f) + 1.0e-10f);
                s = s / config.position_temperature + g;
            }
        }
        for (int c = 0; c < tokens.rows; ++c) {
            for (int t = 0; t < tokens.cols; ++t) {
                if (tokens(c, t) != model.spec.audio_mask_id) scores[static_cast<size_t>(c * tokens.cols + t)] = -INFINITY;
            }
        }
        std::vector<int> idx(static_cast<size_t>(positions));
        std::iota(idx.begin(), idx.end(), 0);
        idx.erase(std::remove_if(idx.begin(), idx.end(), [&](int i) { return !std::isfinite(scores[static_cast<size_t>(i)]); }), idx.end());
        if (update_count < int(idx.size())) {
            std::nth_element(idx.begin(), idx.end() - update_count, idx.end(), [&](int a, int b) { return scores[static_cast<size_t>(a)] < scores[static_cast<size_t>(b)]; });
            idx.erase(idx.begin(), idx.end() - update_count);
        }
        for (int flat : idx) {
            const int c = flat / tokens.cols;
            const int t = flat % tokens.cols;
            tokens(c, t) = pred[static_cast<size_t>(flat)];
        }
        updated_positions = std::min(total_positions, updated_positions + update_count);
        emit_trace(
            model,
            TraceEventKind::LlmProgress,
            "llm",
            "llm_decode",
            0.0,
            double(inputs.target_length) / double(model.higgs.frame_rate()),
            chunk_index,
            chunk_count,
            int(step + 1),
            int(schedule.size()),
            updated_positions,
            total_positions);
    }
    return tokens;
}

std::vector<float> decode_codes(Model & model, const Tensor2i & codes, int chunk_index, int chunk_count) {
    ScopedStage stage(model, "decode", "higgs_decode", chunk_index, chunk_count, double(codes.cols) / double(model.higgs.frame_rate()));
    auto graph = build_decode_graph(model, codes.cols);
    for (int c = 0; c < codes.rows; ++c) {
        std::vector<int32_t> row(static_cast<size_t>(codes.cols));
        for (int t = 0; t < codes.cols; ++t) row[static_cast<size_t>(t)] = codes(c, t);
        tensor_set(graph->code_inputs[static_cast<size_t>(c)], row);
    }
    const ggml_status st = ggml_backend_graph_compute(model.backend.backend, graph->gh->graph);
    if (st != GGML_STATUS_SUCCESS) throw std::runtime_error("Higgs decode graph compute failed");
    const size_t n = ggml_nelements(graph->output);
    std::vector<float> out(n);
    ggml_backend_tensor_get(graph->output, out.data(), 0, n * sizeof(float));
    return out;
}

struct SemanticGraph {
    std::unique_ptr<GraphHandle> gh;
    ggml_tensor * input = nullptr;
    ggml_tensor * pos_conv_weight = nullptr;
    ggml_tensor * output = nullptr;
};

std::vector<float> materialize_hubert_pos_conv_weight(const Model & model) {
    ggml_tensor * weight_g = model.t("a.sm.encoder.pce.conv.pz.w0");
    ggml_tensor * weight_v = model.t("a.sm.encoder.pce.conv.pz.w1");
    if (weight_g->type != GGML_TYPE_F32 || weight_v->type != GGML_TYPE_F32) {
        throw std::runtime_error("HuBERT positional conv weight-norm tensors must be F32");
    }
    const int64_t ne0 = weight_v->ne[0];
    const int64_t ne1 = weight_v->ne[1];
    const int64_t ne2 = weight_v->ne[2];
    if (weight_g->ne[0] != ne0) throw std::runtime_error("HuBERT positional conv weight_g shape mismatch");
    std::vector<float> g(static_cast<size_t>(ne0));
    std::vector<float> v(static_cast<size_t>(ne0) * ne1 * ne2);
    ggml_backend_tensor_get(weight_g, g.data(), 0, g.size() * sizeof(float));
    ggml_backend_tensor_get(weight_v, v.data(), 0, v.size() * sizeof(float));

    std::vector<float> out(v.size());
    for (int64_t k = 0; k < ne0; ++k) {
        double sum = 0.0;
        for (int64_t c = 0; c < ne1; ++c) {
            for (int64_t o = 0; o < ne2; ++o) {
                const float x = v[static_cast<size_t>(k + c * ne0 + o * ne0 * ne1)];
                sum += double(x) * double(x);
            }
        }
        const float scale = sum > 0.0 ? g[static_cast<size_t>(k)] / float(std::sqrt(sum)) : 0.0f;
        for (int64_t c = 0; c < ne1; ++c) {
            for (int64_t o = 0; o < ne2; ++o) {
                const size_t idx = static_cast<size_t>(k + c * ne0 + o * ne0 * ne1);
                out[idx] = v[idx] * scale;
            }
        }
    }
    return out;
}

std::unique_ptr<SemanticGraph> build_semantic_graph(const Model & model, int input_length) {
    auto out = std::make_unique<SemanticGraph>();
    out->gh = std::make_unique<GraphHandle>(model.backend.buft, 262144, 192ull * 1024ull * 1024ull);
    ggml_context * ctx = out->gh->ctx;
    out->input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_length, 1, 1);
    ggml_tensor * raw_pos = model.t("a.sm.encoder.pce.conv.pz.w1");
    out->pos_conv_weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, raw_pos->ne[0], raw_pos->ne[1], raw_pos->ne[2]);
    ggml_set_input(out->input);
    ggml_set_input(out->pos_conv_weight);

    ggml_tensor * cur_3d = hubert_feature_extractor(ctx, model, out->input);
    ggml_tensor * cur = hubert_feature_projection(ctx, model, cur_3d);
    cur_3d = matrix_to_time_channel_3d(ctx, cur);
    ggml_tensor * pos = hubert_positional_conv(ctx, model, cur_3d, out->pos_conv_weight);
    cur_3d = ggml_add(ctx, cur_3d, pos);
    cur = time_channel_3d_to_matrix(ctx, cur_3d);
    cur = norm_affine_2d(ctx, cur, model.t("a.sm.encoder.layer_norm.weight"), model.t("a.sm.encoder.layer_norm.bias"), model.higgs.semantic_layer_norm_eps);

    ggml_tensor * hidden_sum = cur;
    for (int i = 0; i < model.higgs.semantic_block_count; ++i) {
        cur = hubert_encoder_layer(ctx, model, i, cur);
        hidden_sum = ggml_add(ctx, hidden_sum, cur);
    }
    cur = ggml_scale(ctx, hidden_sum, 1.0f / float(model.higgs.semantic_block_count + 1));
    if (model.higgs.semantic_downsample_factor() > 1) cur = slice_2d_columns(ctx, cur, model.higgs.semantic_downsample_factor());
    out->output = cur;
    ggml_set_output(out->output);
    ggml_build_forward_expand(out->gh->graph, out->output);
    out->gh->reserve_alloc();
    std::vector<float> pos_weight = materialize_hubert_pos_conv_weight(model);
    tensor_set(out->pos_conv_weight, pos_weight);
    return out;
}

std::vector<float> extract_semantic_features(Model & model, const std::vector<float> & waveform, int sample_rate) {
    std::vector<float> prepared = sample_rate == model.higgs.semantic_sample_rate
        ? waveform
        : resample_linear(waveform, sample_rate, model.higgs.semantic_sample_rate);
    prepared.insert(prepared.begin(), 160, 0.0f);
    prepared.insert(prepared.end(), 160, 0.0f);

    auto graph = build_semantic_graph(model, int(prepared.size()));
    tensor_set(graph->input, prepared);
    const ggml_status st = ggml_backend_graph_compute(model.backend.backend, graph->gh->graph);
    if (st != GGML_STATUS_SUCCESS) throw std::runtime_error("HuBERT semantic graph compute failed");

    const int hidden = int(graph->output->ne[0]);
    const int length = int(graph->output->ne[1]);
    std::vector<float> storage(static_cast<size_t>(hidden) * length);
    ggml_backend_tensor_get(graph->output, storage.data(), 0, storage.size() * sizeof(float));

    std::vector<float> time_major(static_cast<size_t>(length) * hidden);
    for (int t = 0; t < length; ++t) {
        for (int h = 0; h < hidden; ++h) {
            time_major[static_cast<size_t>(t) + static_cast<size_t>(h) * length] = storage[static_cast<size_t>(h) + static_cast<size_t>(t) * hidden];
        }
    }
    return time_major;
}

struct EncodeGraph {
    std::unique_ptr<GraphHandle> gh;
    ggml_tensor * waveform = nullptr;
    ggml_tensor * semantic = nullptr;
    std::vector<ggml_tensor *> code_outputs;
};

std::unique_ptr<EncodeGraph> build_encode_graph(const Model & model, int waveform_length, int semantic_length, int num_quantizers) {
    auto out = std::make_unique<EncodeGraph>();
    out->gh = std::make_unique<GraphHandle>(model.backend.buft, 131072, 128ull * 1024ull * 1024ull);
    ggml_context * ctx = out->gh->ctx;
    ggml_tensor * one = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 1);
    ggml_tensor * eps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 1);
    out->waveform = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, waveform_length, 1, 1);
    out->semantic = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, semantic_length, model.higgs.semantic_hidden_size, 1);
    ggml_set_input(one);
    ggml_set_input(eps);
    ggml_set_input(out->waveform);
    ggml_set_input(out->semantic);

    ggml_tensor * semantic_encoded = semantic_encoder(ctx, model, out->semantic);
    ggml_tensor * acoustic_encoded = acoustic_encoder(ctx, model, out->waveform, one, eps);
    if (acoustic_encoded->ne[0] != semantic_encoded->ne[0]) {
        throw std::runtime_error("acoustic and semantic encoder lengths differ while building Higgs encode graph");
    }

    ggml_tensor * embeddings = ggml_concat(ctx, acoustic_encoded, semantic_encoded, 1);
    ggml_tensor * cur = time_channel_3d_to_matrix(ctx, embeddings);
    cur = linear(ctx, model.t("a.fc.weight"), cur, model.t("a.fc.bias"));

    ggml_tensor * residual = cur;
    for (int i = 0; i < num_quantizers; ++i) {
        ggml_tensor * codes = nullptr;
        ggml_tensor * quantized = encode_quantizer(ctx, model, i, residual, &codes);
        ggml_set_output(codes);
        out->code_outputs.push_back(codes);
        residual = ggml_sub(ctx, residual, quantized);
    }
    ggml_build_forward_expand(out->gh->graph, out->code_outputs.back());
    for (ggml_tensor * codes : out->code_outputs) ggml_build_forward_expand(out->gh->graph, codes);
    out->gh->reserve_alloc();
    const float one_v = 1.0f;
    const float eps_v = 1.0e-9f;
    ggml_backend_tensor_set(one, &one_v, 0, sizeof(float));
    ggml_backend_tensor_set(eps, &eps_v, 0, sizeof(float));
    return out;
}

Tensor2i encode_codes_from_semantic_features(Model & model, std::vector<float> waveform, const std::vector<float> & semantic_features) {
    const int hidden = model.higgs.semantic_hidden_size;
    if (semantic_features.size() % static_cast<size_t>(hidden) != 0) throw std::runtime_error("semantic feature length is not divisible by hidden size");
    const int semantic_length = int(semantic_features.size() / hidden);
    int acoustic_len = model.higgs.acoustic_encode_output_length(int(waveform.size()));
    const int semantic_len = model.higgs.semantic_encode_output_length(semantic_length);
    if (acoustic_len != semantic_len) {
        const int pad = model.higgs.hop_length() / 2;
        waveform.insert(waveform.begin(), pad, 0.0f);
        waveform.insert(waveform.end(), pad, 0.0f);
        acoustic_len = model.higgs.acoustic_encode_output_length(int(waveform.size()));
        if (acoustic_len != semantic_len) {
            throw std::runtime_error("reference waveform length does not align with semantic features");
        }
    }

    const int num_quantizers = std::min(model.spec.num_audio_codebook, model.higgs.num_quantizers_for_bandwidth(std::nullopt));
    auto graph = build_encode_graph(model, int(waveform.size()), semantic_length, num_quantizers);
    tensor_set(graph->waveform, waveform);
    tensor_set(graph->semantic, semantic_features);
    const ggml_status st = ggml_backend_graph_compute(model.backend.backend, graph->gh->graph);
    if (st != GGML_STATUS_SUCCESS) throw std::runtime_error("Higgs encode graph compute failed");

    const int code_length = int(graph->code_outputs.front()->ne[0]);
    Tensor2i codes{num_quantizers, code_length, {}};
    codes.data.resize(static_cast<size_t>(num_quantizers) * code_length);
    for (int q = 0; q < num_quantizers; ++q) {
        std::vector<int32_t> row(static_cast<size_t>(code_length));
        ggml_backend_tensor_get(graph->code_outputs[static_cast<size_t>(q)], row.data(), 0, row.size() * sizeof(int32_t));
        for (int t = 0; t < code_length; ++t) codes(q, t) = row[static_cast<size_t>(t)];
    }
    return codes;
}

VoiceClonePrompt make_voice_clone_prompt(Model & model, const std::string & path, std::string ref_text, bool preprocess_prompt) {
    int source_rate = model.higgs.sample_rate;
    std::vector<float> wav;
    float rms = 0.0f;
    double ref_audio_seconds = 0.0;
    {
        ScopedStage stage(model, "reference_encode", "reference_read_preprocess");
        wav = read_wav_mono(path, model.higgs.sample_rate, &source_rate);
        for (float v : wav) rms += v * v;
        rms = wav.empty() ? 0.0f : std::sqrt(rms / float(wav.size()));
        if (rms > 0.0f && rms < 0.1f) {
            const float gain = 0.1f / rms;
            for (float & v : wav) v *= gain;
        }
        if (preprocess_prompt) {
            wav = remove_silence(wav, model.higgs.sample_rate, 200, 100, 200);
            ref_text = add_punctuation(ref_text);
        }
        const int clip = int(wav.size() % static_cast<size_t>(model.higgs.hop_length()));
        if (clip > 0) wav.resize(wav.size() - static_cast<size_t>(clip));
        if (wav.empty()) throw std::runtime_error("reference audio is empty after preprocessing");
        ref_audio_seconds = double(wav.size()) / double(model.higgs.sample_rate);
        stage.set_audio_seconds(ref_audio_seconds);
    }
    std::vector<float> semantic;
    {
        ScopedStage stage(model, "reference_encode", "reference_semantic_features", 0, 0, ref_audio_seconds);
        semantic = extract_semantic_features(model, wav, model.higgs.sample_rate);
    }
    Tensor2i codes;
    {
        ScopedStage stage(model, "reference_encode", "reference_higgs_encode", 0, 0, ref_audio_seconds);
        codes = encode_codes_from_semantic_features(model, wav, semantic);
    }
    return VoiceClonePrompt{std::move(codes), std::move(ref_text), rms, ref_audio_seconds};
}

} // namespace

struct OmniVoiceRuntime::Impl {
    Model model;
    explicit Impl(const std::string & model_path, const RuntimeOptions & options) : model(model_path, options) {}
};

OmniVoiceRuntime::OmniVoiceRuntime(const std::string & model_path, const RuntimeOptions & options)
    : impl_(std::make_unique<Impl>(model_path, options)) {}

OmniVoiceRuntime::~OmniVoiceRuntime() = default;

int OmniVoiceRuntime::sample_rate() const {
    return impl_->model.higgs.sample_rate;
}

Audio OmniVoiceRuntime::generate(const SynthesisParams & params) {
    Model & model = impl_->model;
    GenerationConfig config = params.generation;
    VoiceClonePrompt prompt;
    bool has_prompt = false;
    if (!params.ref_audio_path.empty()) {
        prompt = make_voice_clone_prompt(model, params.ref_audio_path, params.ref_text, config.preprocess_prompt);
        has_prompt = true;
    }
    std::string instruct = params.instruct;
    if (params.ref_audio_path.empty() && instruct.empty() && !params.auto_voice) instruct = "male, British accent";
    const std::string lang = resolve_language(params.language);
    instruct = resolve_instruct(instruct, contains_cjk(params.text));
    std::vector<std::string> chunk_texts;
    std::vector<int> chunk_targets;
    {
        ScopedStage stage(model, "planning", "generation_plan");
        const int target = estimate_target_tokens(model, params.text, params.speed, params.duration, has_prompt ? &prompt : nullptr);
        const int threshold = int(config.audio_chunk_threshold * model.higgs.frame_rate());
        if (target > threshold && config.audio_chunk_duration > 0.0f) {
            const float avg = float(target) / float(std::max(1, utf8_length(params.text)));
            const int chunk_len = std::max(1, int(config.audio_chunk_duration * model.higgs.frame_rate() / std::max(avg, 1.0e-6f)));
            chunk_texts = chunk_text_punctuation(params.text, chunk_len, 3);
        }
        if (chunk_texts.empty()) chunk_texts.push_back(params.text);
        int remaining = target;
        int remaining_weight = 0;
        for (const auto & c : chunk_texts) remaining_weight += std::max(1, utf8_length(c));
        for (size_t i = 0; i < chunk_texts.size(); ++i) {
            const int chunk_weight = std::max(1, utf8_length(chunk_texts[i]));
            int cur;
            if (i + 1 == chunk_texts.size()) cur = remaining;
            else {
                cur = int(std::round(float(remaining) * float(chunk_weight) / float(remaining_weight)));
                cur = std::max(1, std::min(cur, remaining - int(chunk_texts.size() - i - 1)));
            }
            chunk_targets.push_back(cur);
            remaining -= cur;
            remaining_weight -= chunk_weight;
        }
    }

    std::vector<std::vector<float>> chunk_wavs;
    for (size_t i = 0; i < chunk_texts.size(); ++i) {
        const VoiceClonePrompt * prompt_ptr = has_prompt ? &prompt : nullptr;
        InferenceInputs in;
        {
            ScopedStage stage(model, "prepare", "prepare_inputs", int(i + 1), int(chunk_texts.size()));
            in = prepare_inputs(model, chunk_texts[i], chunk_targets[i], lang, instruct, prompt_ptr, config.denoise);
        }
        Tensor2i codes = generate_codes(model, in, config, int(i + 1), int(chunk_texts.size()));
        if (!has_prompt && chunk_texts.size() > 1) {
            prompt.ref_audio_tokens = codes;
            prompt.ref_text = chunk_texts[i];
            has_prompt = true;
        }
        chunk_wavs.push_back(decode_codes(model, codes, int(i + 1), int(chunk_texts.size())));
    }
    std::vector<float> waveform;
    {
        ScopedStage stage(model, "postprocess", "postprocess");
        waveform = chunk_wavs.size() == 1 ? chunk_wavs[0] : cross_fade_chunks(chunk_wavs, model.higgs.sample_rate);
        if (config.postprocess_output) waveform = remove_silence(waveform, model.higgs.sample_rate, 500, 100, 100);
        float peak = 0.0f;
        for (float v : waveform) peak = std::max(peak, std::abs(v));
        if (peak > 1.0e-6f) for (float & v : waveform) v = v / peak * 0.5f;
        waveform = fade_and_pad_audio(waveform, model.higgs.sample_rate);
    }
    return Audio{model.higgs.sample_rate, std::move(waveform)};
}

} // namespace omnivoice
