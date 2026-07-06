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
