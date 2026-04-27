# OmniVoice GGML C++

Standalone C++ inference runtime for OmniVoice GGUF models, implemented directly
on top of the GGML C/C++ API.

This project is intentionally small: it contains the main inference path only.
It does not depend on Python, `tokenizers`, `soundfile`, `torchaudio`, PyTorch,
the `ggbond` Python wrapper, or the llama.cpp runtime.

## Status

This is an alpha standalone runtime.

Verified locally:

- CUDA text-to-speech with OmniVoice GGUF models.
- CUDA reference-audio voice cloning with WAV input and explicit `--ref-text`.
- CPU fallback smoke tests.
- Long Chinese text synthesis with UTF-8-aware chunking and chunked Higgs
  decode.

Not yet claimed:

- strict Python tokenizer parity across all Unicode normalization cases
- exact NumPy PCG64 seed parity
- exhaustive Python-vs-C++ generated-code parity
- production HTTP serving
- automatic ASR for missing reference transcripts

## Repository Layout

```text
CMakeLists.txt
include/omnivoice.h
src/
patches/
vendor/ggml/
```

`vendor/ggml` is an upstream `ggml-org/ggml` submodule pinned to tag `v0.9.11`.
The submodule is kept clean. OmniVoice CUDA decode currently needs the patch in
`patches/`.

## Requirements

- CMake 3.15+
- C++17 compiler
- Git submodule support
- CUDA toolkit for `-DGGML_CUDA=ON`
- An OmniVoice-compatible GGUF model file supplied by the user

The repository does not include model weights.

## Clone And Prepare

```bash
git clone <this-repository-url> omnivoice-ggml-cpp
cd omnivoice-ggml-cpp
git submodule update --init --recursive
```

Apply the OmniVoice CUDA fixes to the upstream ggml submodule:

```bash
patch_file="$PWD/patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch"
git -C vendor/ggml apply --check "$patch_file"
git -C vendor/ggml apply "$patch_file"
```

The patch is required for the verified long-text CUDA Higgs decode path on
ggml `v0.9.11`.

## Build

CUDA build:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

CPU-only build:

```bash
cmake -S . -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j8
```

The CLI binary is:

```bash
./build/omnivoice-cli
```

The CLI prints stage timing, LLM step progress, and RTF summaries to stderr.
This is intended to make latency attribution visible during local inference.

## Text-To-Speech

```bash
./build/omnivoice-cli \
  --model /path/to/omnivoice-q8_0.gguf \
  --text "你好，这是一个测试。" \
  --output /tmp/omnivoice-cpp-test.wav \
  --language Chinese \
  --instruct "female, low pitch" \
  --device cuda:0 \
  --num-step 32 \
  --seed 123
```

## Reference-Audio Voice Cloning

Reference audio is WAV-only in this version. `--ref-text` is required.

```bash
./build/omnivoice-cli \
  --model /path/to/omnivoice-q8_0.gguf \
  --text "请使用参考音色朗读这句话。" \
  --output /tmp/omnivoice-cpp-ref.wav \
  --ref-audio /path/to/reference.wav \
  --ref-text "参考音频中说出的文字。" \
  --language Chinese \
  --device cuda:0 \
  --num-step 32 \
  --seed 123
```

## CPU Fallback

```bash
./build-cpu/omnivoice-cli \
  --model /path/to/omnivoice-q8_0.gguf \
  --text "CPU 测试。" \
  --output /tmp/omnivoice-cpp-cpu.wav \
  --language Chinese \
  --device cpu \
  --threads 8 \
  --num-step 8 \
  --seed 123
```

## CLI Options

```text
--model                 GGUF model path
--text                  input text
--output, --out         output WAV path
--language              language name or code, for example Chinese, zh, English, en
--instruct              voice design prompt, for example "female, low pitch"
--auto-voice            true|false
--ref-audio             WAV reference audio
--ref-text              transcript for --ref-audio
--num-step              generation iteration count, default 32
--guidance-scale        classifier-free guidance scale, default 2.0
--speed                 speaking speed factor, default 1.0
--duration              fixed output duration in seconds
--position-temperature  position sampling temperature, default 5.0
--class-temperature     class sampling temperature, default 0.0
--device                cpu or cuda[:N]
--backend               cpu or cuda
--seed                  random seed
--threads               CPU thread count
```

`--num-step` is not the output length. It controls how many update iterations
are used for a fixed target audio span. To force a longer or shorter result,
use `--duration`.

Long text is split by punctuation when the estimated target length exceeds the
chunk threshold. Chunk planning is UTF-8 aware, so Chinese text is counted by
characters rather than bytes.

## Latency Output

Each run prints major stages such as model loading, reference audio encoding,
generation planning, LLM decode, Higgs decode, postprocessing, and WAV writing.
The LLM decode stage is rendered as a tqdm-style progress line.

At the end of synthesis, the CLI prints:

```text
[rtf] total=...
[rtf] llm=...
[rtf] reference_encode=...
[rtf] decode=...
```

`total`, `llm`, and `decode` use generated output audio duration as the RTF
denominator. `reference_encode` uses the preprocessed reference audio duration.

On a local RTX 4060 Ti run with a long Chinese prompt, WAV reference audio, and
`omnivoice-q8_0.gguf`, the current CUDA path produced about 77 seconds of audio
in about 15 seconds of generation time:

```text
[rtf] output_audio_s=77.140
[rtf] total=0.194  seconds=14.932
[rtf] llm=0.154  seconds=11.889
[rtf] reference_encode=0.043  seconds=0.355  ref_audio_s=8.160
[rtf] decode=0.035  seconds=2.664
```

For that case, LLM decode is the main latency source. Reference encoding and
Higgs decode are comparatively small.

## Models

Pass a local OmniVoice-compatible GGUF file via `--model`.

Known local smoke surfaces include:

- F32 GGUF
- Q8_0 GGUF
- Q5_1 linear-scope GGUF

Other GGML tensor types may work if the model artifact preserves the expected
OmniVoice tensor names and metadata.

## Limitations

- WAV input/output only.
- `--ref-text` is required with `--ref-audio`; no ASR fallback is included.
- No HTTP server or serving scheduler.
- No debug telemetry, performance counters, or experimental backend gates.
- The tokenizer and stochastic seed paths are close to the Python staging
  runtime but are not yet certified as exact parity for every edge case.

## Safety

Use this project only with proper consent and authorization. Do not use it for
unauthorized voice cloning, impersonation, fraud, scams, or other illegal or
unethical activity.

## License

This repository is licensed under the Apache License 2.0. See `LICENSE`.

Third-party notices are listed in `THIRD_PARTY_NOTICES.md`.
