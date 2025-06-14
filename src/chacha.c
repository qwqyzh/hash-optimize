#include "mercha.h"
#include <immintrin.h>
#include <emmintrin.h>

static inline void chacha20_L4(__m128i *pa,__m128i *pb,__m128i *pc,__m128i *pd){
    register __m128i A asm("xmm4") = *pa;
    register __m128i B asm("xmm1") = *pb;
    register __m128i C asm("xmm3") = *pc;
    register __m128i D asm("xmm0") = *pd;
    asm volatile(
        ".p2align 4,,10               \n\t"
        ".p2align 3                  \n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm2  \n\t"
        "vpslld   $16, %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrld   $20, %%xmm1, %%xmm2  \n\t"
        "vpslld   $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $24, %%xmm0, %%xmm2  \n\t"
        "vpslld   $8,  %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrldq  $8,  %%xmm3, %%xmm6  \n\t"
        "vpsrld   $25, %%xmm1, %%xmm2  \n\t"
        "vpslld   $7,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpslldq  $8,  %%xmm3, %%xmm3  \n\t"
        "vpsrldq  $4,  %%xmm1, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $12, %%xmm0, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm0, %%xmm0 \n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm4\n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm5  \n\t"
        "vpslld   $16, %%xmm0, %%xmm1  \n\t"
        "vpor     %%xmm5, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm2, %%xmm2\n\t"
        "vpsrld   $20, %%xmm2, %%xmm0  \n\t"
        "vpslld   $12, %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm0, %%xmm2, %%xmm2\n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm5\n\t"
        "vpxor    %%xmm5, %%xmm1, %%xmm1\n\t"
        "vmovdqa  %%xmm5, %%xmm4      \n\t"
        "vpsrld   $24, %%xmm1, %%xmm0  \n\t"
        "vpslld   $8,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm6\n\t"
        "vpxor    %%xmm6, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $8,  %%xmm6, %%xmm3  \n\t"
        "vpsrld   $25, %%xmm2, %%xmm1  \n\t"
        "vpslld   $7,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpslldq  $8,  %%xmm6, %%xmm6  \n\t"
        "vpsrldq  $12, %%xmm2, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm1\n\t"
        "vpsrldq  $4,  %%xmm0, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm0, %%xmm0 \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm2  \n\t"
        "vpslld   $16, %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrld   $20, %%xmm1, %%xmm2  \n\t"
        "vpslld   $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $24, %%xmm0, %%xmm2  \n\t"
        "vpslld   $8,  %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrldq  $8,  %%xmm3, %%xmm6  \n\t"
        "vpsrld   $25, %%xmm1, %%xmm2  \n\t"
        "vpslld   $7,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpslldq  $8,  %%xmm3, %%xmm3  \n\t"
        "vpsrldq  $4,  %%xmm1, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $12, %%xmm0, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm0, %%xmm0 \n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm4\n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm5  \n\t"
        "vpslld   $16, %%xmm0, %%xmm1  \n\t"
        "vpor     %%xmm5, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm2, %%xmm2\n\t"
        "vpsrld   $20, %%xmm2, %%xmm0  \n\t"
        "vpslld   $12, %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm0, %%xmm2, %%xmm2\n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm5\n\t"
        "vpxor    %%xmm5, %%xmm1, %%xmm1\n\t"
        "vmovdqa  %%xmm5, %%xmm4      \n\t"
        "vpsrld   $24, %%xmm1, %%xmm0  \n\t"
        "vpslld   $8,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm6\n\t"
        "vpxor    %%xmm6, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $8,  %%xmm6, %%xmm3  \n\t"
        "vpsrld   $25, %%xmm2, %%xmm1  \n\t"
        "vpslld   $7,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpslldq  $8,  %%xmm6, %%xmm6  \n\t"
        "vpsrldq  $12, %%xmm2, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm1\n\t"
        "vpsrldq  $4,  %%xmm0, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm0, %%xmm0 \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm2  \n\t"
        "vpslld   $16, %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrld   $20, %%xmm1, %%xmm2  \n\t"
        "vpslld   $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $24, %%xmm0, %%xmm2  \n\t"
        "vpslld   $8,  %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrldq  $8,  %%xmm3, %%xmm6  \n\t"
        "vpsrld   $25, %%xmm1, %%xmm2  \n\t"
        "vpslld   $7,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpslldq  $8,  %%xmm3, %%xmm3  \n\t"
        "vpsrldq  $4,  %%xmm1, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $12, %%xmm0, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm0, %%xmm0 \n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm4\n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm5  \n\t"
        "vpslld   $16, %%xmm0, %%xmm1  \n\t"
        "vpor     %%xmm5, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm2, %%xmm2\n\t"
        "vpsrld   $20, %%xmm2, %%xmm0  \n\t"
        "vpslld   $12, %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm0, %%xmm2, %%xmm2\n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm5\n\t"
        "vpxor    %%xmm5, %%xmm1, %%xmm1\n\t"
        "vmovdqa  %%xmm5, %%xmm4      \n\t"
        "vpsrld   $24, %%xmm1, %%xmm0  \n\t"
        "vpslld   $8,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm6\n\t"
        "vpxor    %%xmm6, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $8,  %%xmm6, %%xmm3  \n\t"
        "vpsrld   $25, %%xmm2, %%xmm1  \n\t"
        "vpslld   $7,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpslldq  $8,  %%xmm6, %%xmm6  \n\t"
        "vpsrldq  $12, %%xmm2, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm1\n\t"
        "vpsrldq  $4,  %%xmm0, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm0, %%xmm0 \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm2  \n\t"
        "vpslld   $16, %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrld   $20, %%xmm1, %%xmm2  \n\t"
        "vpslld   $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $24, %%xmm0, %%xmm2  \n\t"
        "vpslld   $8,  %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrldq  $8,  %%xmm3, %%xmm6  \n\t"
        "vpsrld   $25, %%xmm1, %%xmm2  \n\t"
        "vpslld   $7,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpslldq  $8,  %%xmm3, %%xmm3  \n\t"
        "vpsrldq  $4,  %%xmm1, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $12, %%xmm0, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm0, %%xmm0 \n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm4\n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm5  \n\t"
        "vpslld   $16, %%xmm0, %%xmm1  \n\t"
        "vpor     %%xmm5, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm2, %%xmm2\n\t"
        "vpsrld   $20, %%xmm2, %%xmm0  \n\t"
        "vpslld   $12, %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm0, %%xmm2, %%xmm2\n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm5\n\t"
        "vpxor    %%xmm5, %%xmm1, %%xmm1\n\t"
        "vmovdqa  %%xmm5, %%xmm4      \n\t"
        "vpsrld   $24, %%xmm1, %%xmm0  \n\t"
        "vpslld   $8,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm6\n\t"
        "vpxor    %%xmm6, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $8,  %%xmm6, %%xmm3  \n\t"
        "vpsrld   $25, %%xmm2, %%xmm1  \n\t"
        "vpslld   $7,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpslldq  $8,  %%xmm6, %%xmm6  \n\t"
        "vpsrldq  $12, %%xmm2, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm1\n\t"
        "vpsrldq  $4,  %%xmm0, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm0, %%xmm0 \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm2  \n\t"
        "vpslld   $16, %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrld   $20, %%xmm1, %%xmm2  \n\t"
        "vpslld   $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $24, %%xmm0, %%xmm2  \n\t"
        "vpslld   $8,  %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrldq  $8,  %%xmm3, %%xmm6  \n\t"
        "vpsrld   $25, %%xmm1, %%xmm2  \n\t"
        "vpslld   $7,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpslldq  $8,  %%xmm3, %%xmm3  \n\t"
        "vpsrldq  $4,  %%xmm1, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $12, %%xmm0, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm0, %%xmm0 \n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm4\n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm5  \n\t"
        "vpslld   $16, %%xmm0, %%xmm1  \n\t"
        "vpor     %%xmm5, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm2, %%xmm2\n\t"
        "vpsrld   $20, %%xmm2, %%xmm0  \n\t"
        "vpslld   $12, %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm0, %%xmm2, %%xmm2\n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm5\n\t"
        "vpxor    %%xmm5, %%xmm1, %%xmm1\n\t"
        "vmovdqa  %%xmm5, %%xmm4      \n\t"
        "vpsrld   $24, %%xmm1, %%xmm0  \n\t"
        "vpslld   $8,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm6\n\t"
        "vpxor    %%xmm6, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $8,  %%xmm6, %%xmm3  \n\t"
        "vpsrld   $25, %%xmm2, %%xmm1  \n\t"
        "vpslld   $7,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpslldq  $8,  %%xmm6, %%xmm6  \n\t"
        "vpsrldq  $12, %%xmm2, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm1\n\t"
        "vpsrldq  $4,  %%xmm0, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm0, %%xmm0 \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm2  \n\t"
        "vpslld   $16, %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrld   $20, %%xmm1, %%xmm2  \n\t"
        "vpslld   $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $24, %%xmm0, %%xmm2  \n\t"
        "vpslld   $8,  %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrldq  $8,  %%xmm3, %%xmm6  \n\t"
        "vpsrld   $25, %%xmm1, %%xmm2  \n\t"
        "vpslld   $7,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpslldq  $8,  %%xmm3, %%xmm3  \n\t"
        "vpsrldq  $4,  %%xmm1, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $12, %%xmm0, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm0, %%xmm0 \n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm4\n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm5  \n\t"
        "vpslld   $16, %%xmm0, %%xmm1  \n\t"
        "vpor     %%xmm5, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm2, %%xmm2\n\t"
        "vpsrld   $20, %%xmm2, %%xmm0  \n\t"
        "vpslld   $12, %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm0, %%xmm2, %%xmm2\n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm5\n\t"
        "vpxor    %%xmm5, %%xmm1, %%xmm1\n\t"
        "vmovdqa  %%xmm5, %%xmm4      \n\t"
        "vpsrld   $24, %%xmm1, %%xmm0  \n\t"
        "vpslld   $8,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm6\n\t"
        "vpxor    %%xmm6, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $8,  %%xmm6, %%xmm3  \n\t"
        "vpsrld   $25, %%xmm2, %%xmm1  \n\t"
        "vpslld   $7,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpslldq  $8,  %%xmm6, %%xmm6  \n\t"
        "vpsrldq  $12, %%xmm2, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm1\n\t"
        "vpsrldq  $4,  %%xmm0, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm0, %%xmm0 \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm2  \n\t"
        "vpslld   $16, %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrld   $20, %%xmm1, %%xmm2  \n\t"
        "vpslld   $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $24, %%xmm0, %%xmm2  \n\t"
        "vpslld   $8,  %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrldq  $8,  %%xmm3, %%xmm6  \n\t"
        "vpsrld   $25, %%xmm1, %%xmm2  \n\t"
        "vpslld   $7,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpslldq  $8,  %%xmm3, %%xmm3  \n\t"
        "vpsrldq  $4,  %%xmm1, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $12, %%xmm0, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm0, %%xmm0 \n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm4\n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm5  \n\t"
        "vpslld   $16, %%xmm0, %%xmm1  \n\t"
        "vpor     %%xmm5, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm2, %%xmm2\n\t"
        "vpsrld   $20, %%xmm2, %%xmm0  \n\t"
        "vpslld   $12, %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm0, %%xmm2, %%xmm2\n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm5\n\t"
        "vpxor    %%xmm5, %%xmm1, %%xmm1\n\t"
        "vmovdqa  %%xmm5, %%xmm4      \n\t"
        "vpsrld   $24, %%xmm1, %%xmm0  \n\t"
        "vpslld   $8,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm6\n\t"
        "vpxor    %%xmm6, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $8,  %%xmm6, %%xmm3  \n\t"
        "vpsrld   $25, %%xmm2, %%xmm1  \n\t"
        "vpslld   $7,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpslldq  $8,  %%xmm6, %%xmm6  \n\t"
        "vpsrldq  $12, %%xmm2, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm1\n\t"
        "vpsrldq  $4,  %%xmm0, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm0, %%xmm0 \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm2  \n\t"
        "vpslld   $16, %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrld   $20, %%xmm1, %%xmm2  \n\t"
        "vpslld   $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $24, %%xmm0, %%xmm2  \n\t"
        "vpslld   $8,  %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrldq  $8,  %%xmm3, %%xmm6  \n\t"
        "vpsrld   $25, %%xmm1, %%xmm2  \n\t"
        "vpslld   $7,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpslldq  $8,  %%xmm3, %%xmm3  \n\t"
        "vpsrldq  $4,  %%xmm1, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $12, %%xmm0, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm0, %%xmm0 \n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm4\n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm5  \n\t"
        "vpslld   $16, %%xmm0, %%xmm1  \n\t"
        "vpor     %%xmm5, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm2, %%xmm2\n\t"
        "vpsrld   $20, %%xmm2, %%xmm0  \n\t"
        "vpslld   $12, %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm0, %%xmm2, %%xmm2\n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm5\n\t"
        "vpxor    %%xmm5, %%xmm1, %%xmm1\n\t"
        "vmovdqa  %%xmm5, %%xmm4      \n\t"
        "vpsrld   $24, %%xmm1, %%xmm0  \n\t"
        "vpslld   $8,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm6\n\t"
        "vpxor    %%xmm6, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $8,  %%xmm6, %%xmm3  \n\t"
        "vpsrld   $25, %%xmm2, %%xmm1  \n\t"
        "vpslld   $7,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpslldq  $8,  %%xmm6, %%xmm6  \n\t"
        "vpsrldq  $12, %%xmm2, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm1\n\t"
        "vpsrldq  $4,  %%xmm0, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm0, %%xmm0 \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm2  \n\t"
        "vpslld   $16, %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrld   $20, %%xmm1, %%xmm2  \n\t"
        "vpslld   $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $24, %%xmm0, %%xmm2  \n\t"
        "vpslld   $8,  %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrldq  $8,  %%xmm3, %%xmm6  \n\t"
        "vpsrld   $25, %%xmm1, %%xmm2  \n\t"
        "vpslld   $7,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpslldq  $8,  %%xmm3, %%xmm3  \n\t"
        "vpsrldq  $4,  %%xmm1, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $12, %%xmm0, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm0, %%xmm0 \n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm4\n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm5  \n\t"
        "vpslld   $16, %%xmm0, %%xmm1  \n\t"
        "vpor     %%xmm5, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm2, %%xmm2\n\t"
        "vpsrld   $20, %%xmm2, %%xmm0  \n\t"
        "vpslld   $12, %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm0, %%xmm2, %%xmm2\n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm5\n\t"
        "vpxor    %%xmm5, %%xmm1, %%xmm1\n\t"
        "vmovdqa  %%xmm5, %%xmm4      \n\t"
        "vpsrld   $24, %%xmm1, %%xmm0  \n\t"
        "vpslld   $8,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm6\n\t"
        "vpxor    %%xmm6, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $8,  %%xmm6, %%xmm3  \n\t"
        "vpsrld   $25, %%xmm2, %%xmm1  \n\t"
        "vpslld   $7,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpslldq  $8,  %%xmm6, %%xmm6  \n\t"
        "vpsrldq  $12, %%xmm2, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm1\n\t"
        "vpsrldq  $4,  %%xmm0, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm0, %%xmm0 \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm2  \n\t"
        "vpslld   $16, %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrld   $20, %%xmm1, %%xmm2  \n\t"
        "vpslld   $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm4, %%xmm4\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $24, %%xmm0, %%xmm2  \n\t"
        "vpslld   $8,  %%xmm0, %%xmm0  \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm1, %%xmm1\n\t"
        "vpsrldq  $8,  %%xmm3, %%xmm6  \n\t"
        "vpsrld   $25, %%xmm1, %%xmm2  \n\t"
        "vpslld   $7,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm2, %%xmm1, %%xmm1\n\t"
        "vpslldq  $8,  %%xmm3, %%xmm3  \n\t"
        "vpsrldq  $4,  %%xmm1, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $12, %%xmm0, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm0, %%xmm0 \n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm4\n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpxor    %%xmm4, %%xmm0, %%xmm0\n\t"
        "vpsrld   $16, %%xmm0, %%xmm5  \n\t"
        "vpslld   $16, %%xmm0, %%xmm1  \n\t"
        "vpor     %%xmm5, %%xmm1, %%xmm1\n\t"
        "vpaddd   %%xmm1, %%xmm3, %%xmm3\n\t"
        "vpxor    %%xmm3, %%xmm2, %%xmm2\n\t"
        "vpsrld   $20, %%xmm2, %%xmm0  \n\t"
        "vpslld   $12, %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm0, %%xmm2, %%xmm2\n\t"
        "vpaddd   %%xmm2, %%xmm4, %%xmm5\n\t"
        "vpxor    %%xmm5, %%xmm1, %%xmm1\n\t"
        "vmovdqa  %%xmm5, %%xmm4      \n\t"
        "vpsrld   $24, %%xmm1, %%xmm0  \n\t"
        "vpslld   $8,  %%xmm1, %%xmm1  \n\t"
        "vpor     %%xmm0, %%xmm1, %%xmm0\n\t"
        "vpaddd   %%xmm0, %%xmm3, %%xmm6\n\t"
        "vpxor    %%xmm6, %%xmm2, %%xmm2\n\t"
        "vpsrldq  $8,  %%xmm6, %%xmm3  \n\t"
        "vpsrld   $25, %%xmm2, %%xmm1  \n\t"
        "vpslld   $7,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm2\n\t"
        "vpslldq  $8,  %%xmm6, %%xmm6  \n\t"
        "vpsrldq  $12, %%xmm2, %%xmm1 \n\t"
        "vpslldq  $4,  %%xmm2, %%xmm2  \n\t"
        "vpor     %%xmm3, %%xmm6, %%xmm3\n\t"
        "vpor     %%xmm1, %%xmm2, %%xmm1\n\t"
        "vpsrldq  $4,  %%xmm0, %%xmm2  \n\t"
        "vpslldq  $12, %%xmm0, %%xmm0 \n\t"
        "vpor     %%xmm2, %%xmm0, %%xmm0\n\t"
        :
        : /* no separate inputs */
        : "xmm2","xmm5","xmm6","memory"
    );
    *pa = A; *pb = B; *pc = C; *pd = D;
}



void chacha20_encrypt(const uint8_t key[32], const uint8_t nonce[12], uint32_t initial_counter, uint8_t *buffer, size_t length) {
    uint32_t key_words[8];
    uint32_t nonce_words[3];
    for (int i = 0; i < 8; i++) {
        key_words[i] = (uint32_t)key[i*4 + 0]      |
                      ((uint32_t)key[i*4 + 1] << 8)  |
                      ((uint32_t)key[i*4 + 2] << 16) |
                      ((uint32_t)key[i*4 + 3] << 24);
    }
    for (int i = 0; i < 3; i++) {
        nonce_words[i] = (uint32_t)nonce[i*4 + 0]      |
                        ((uint32_t)nonce[i*4 + 1] << 8)  |
                        ((uint32_t)nonce[i*4 + 2] << 16) |
                        ((uint32_t)nonce[i*4 + 3] << 24);
    }

    uint32_t state[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,             
        key_words[0], key_words[1], key_words[2], key_words[3],     
        key_words[4], key_words[5], key_words[6], key_words[7],     
        initial_counter,                                            
        nonce_words[0], nonce_words[1], nonce_words[2]              
    };

    uint8_t key_stream[64];
    int t = length >> 6;
    #pragma omp parallel for private(key_stream) schedule(static) proc_bind(spread)
    for(int j=0; j<=t; j+=4) {
        size_t offset = j << 6;
        uint32_t working_state[16];
        memcpy(working_state, state, 64);
        working_state[12] += j;
        __m128i a = _mm_loadu_si128((__m128i *)&working_state[0]);    
        __m128i b = _mm_loadu_si128((__m128i *)&working_state[4]);
        __m128i c = _mm_loadu_si128((__m128i *)&working_state[8]);    
        __m128i d = _mm_loadu_si128((__m128i *)&working_state[12]);
        chacha20_L4(&a,&b,&c,&d);
        _mm_storeu_si128((__m128i *)&working_state[0], a);
        _mm_storeu_si128((__m128i *)&working_state[4], b);                                
        _mm_storeu_si128((__m128i *)&working_state[8], c);
        _mm_storeu_si128((__m128i *)&working_state[12], d);
    
        working_state[12] += j;
        __m256i vs0 = _mm256_loadu_si256((const __m256i *)state);          
        __m256i vs1 = _mm256_loadu_si256((const __m256i *)(state + 8));        
        __m256i vw0 = _mm256_loadu_si256((const __m256i *)working_state);      
        __m256i vw1 = _mm256_loadu_si256((const __m256i *)(working_state + 8));  
        vw0 = _mm256_add_epi32(vw0, vs0);
        vw1 = _mm256_add_epi32(vw1, vs1);
        _mm256_storeu_si256((__m256i *)working_state, vw0);
        _mm256_storeu_si256((__m256i *)(working_state + 8), vw1);
        _mm256_storeu_si256((__m256i *)key_stream, vw0);
        _mm256_storeu_si256((__m256i *)(key_stream + 32), vw1);
        __m256i ks = _mm256_loadu_si256((const __m256i*)(key_stream ));
        __m256i buf = _mm256_loadu_si256((const __m256i*)(buffer + offset));
        __m256i res = _mm256_xor_si256(ks, buf);
        _mm256_storeu_si256((__m256i*)(buffer + offset), res);
        ks = _mm256_loadu_si256((const __m256i*)(key_stream + 32));
        buf = _mm256_loadu_si256((const __m256i*)(buffer + offset + 32));
        res = _mm256_xor_si256(ks, buf);
        _mm256_storeu_si256((__m256i*)(buffer + offset + 32), res);

        offset = (j+1) << 6;
        memcpy(working_state, state, 64);
        working_state[12] += j+1;
        a = _mm_loadu_si128((__m128i *)&working_state[0]);    
        b = _mm_loadu_si128((__m128i *)&working_state[4]);
        c = _mm_loadu_si128((__m128i *)&working_state[8]);    
        d = _mm_loadu_si128((__m128i *)&working_state[12]);
        chacha20_L4(&a,&b,&c,&d);
        _mm_storeu_si128((__m128i *)&working_state[0], a);
        _mm_storeu_si128((__m128i *)&working_state[4], b);                                
        _mm_storeu_si128((__m128i *)&working_state[8], c);
        _mm_storeu_si128((__m128i *)&working_state[12], d);
    
        working_state[12] += j+1;
        vs0 = _mm256_loadu_si256((const __m256i *)state);          
        vs1 = _mm256_loadu_si256((const __m256i *)(state + 8));        
        vw0 = _mm256_loadu_si256((const __m256i *)working_state);      
        vw1 = _mm256_loadu_si256((const __m256i *)(working_state + 8));  
        vw0 = _mm256_add_epi32(vw0, vs0);
        vw1 = _mm256_add_epi32(vw1, vs1);
        _mm256_storeu_si256((__m256i *)working_state, vw0);
        _mm256_storeu_si256((__m256i *)(working_state + 8), vw1);
        _mm256_storeu_si256((__m256i *)key_stream, vw0);
        _mm256_storeu_si256((__m256i *)(key_stream + 32), vw1);
        ks = _mm256_loadu_si256((const __m256i*)(key_stream ));
        buf = _mm256_loadu_si256((const __m256i*)(buffer + offset));
        res = _mm256_xor_si256(ks, buf);
        _mm256_storeu_si256((__m256i*)(buffer + offset), res);
        ks = _mm256_loadu_si256((const __m256i*)(key_stream + 32));
        buf = _mm256_loadu_si256((const __m256i*)(buffer + offset + 32));
        res = _mm256_xor_si256(ks, buf);
        _mm256_storeu_si256((__m256i*)(buffer + offset + 32), res);

        offset = (j+2) << 6;
        memcpy(working_state, state, 64);
        working_state[12] += j+2;
        a = _mm_loadu_si128((__m128i *)&working_state[0]);    
        b = _mm_loadu_si128((__m128i *)&working_state[4]);
        c = _mm_loadu_si128((__m128i *)&working_state[8]);    
        d = _mm_loadu_si128((__m128i *)&working_state[12]);
        chacha20_L4(&a,&b,&c,&d);
        _mm_storeu_si128((__m128i *)&working_state[0], a);
        _mm_storeu_si128((__m128i *)&working_state[4], b);                                
        _mm_storeu_si128((__m128i *)&working_state[8], c);
        _mm_storeu_si128((__m128i *)&working_state[12], d);
    
        working_state[12] += j+2;
        vs0 = _mm256_loadu_si256((const __m256i *)state);          
        vs1 = _mm256_loadu_si256((const __m256i *)(state + 8));        
        vw0 = _mm256_loadu_si256((const __m256i *)working_state);      
        vw1 = _mm256_loadu_si256((const __m256i *)(working_state + 8));  
        vw0 = _mm256_add_epi32(vw0, vs0);
        vw1 = _mm256_add_epi32(vw1, vs1);
        _mm256_storeu_si256((__m256i *)working_state, vw0);
        _mm256_storeu_si256((__m256i *)(working_state + 8), vw1);
        _mm256_storeu_si256((__m256i *)key_stream, vw0);
        _mm256_storeu_si256((__m256i *)(key_stream + 32), vw1);
        ks = _mm256_loadu_si256((const __m256i*)(key_stream ));
        buf = _mm256_loadu_si256((const __m256i*)(buffer + offset));
        res = _mm256_xor_si256(ks, buf);
        _mm256_storeu_si256((__m256i*)(buffer + offset), res);
        ks = _mm256_loadu_si256((const __m256i*)(key_stream + 32));
        buf = _mm256_loadu_si256((const __m256i*)(buffer + offset + 32));
        res = _mm256_xor_si256(ks, buf);
        _mm256_storeu_si256((__m256i*)(buffer + offset + 32), res);

        offset = (j+3) << 6;
        memcpy(working_state, state, 64);
        working_state[12] += j+3;
        a = _mm_loadu_si128((__m128i *)&working_state[0]);    
        b = _mm_loadu_si128((__m128i *)&working_state[4]);
        c = _mm_loadu_si128((__m128i *)&working_state[8]);    
        d = _mm_loadu_si128((__m128i *)&working_state[12]);
        chacha20_L4(&a,&b,&c,&d);
        _mm_storeu_si128((__m128i *)&working_state[0], a);
        _mm_storeu_si128((__m128i *)&working_state[4], b);                                
        _mm_storeu_si128((__m128i *)&working_state[8], c);
        _mm_storeu_si128((__m128i *)&working_state[12], d);
    
        working_state[12] += j+3;
        vs0 = _mm256_loadu_si256((const __m256i *)state);          
        vs1 = _mm256_loadu_si256((const __m256i *)(state + 8));        
        vw0 = _mm256_loadu_si256((const __m256i *)working_state);      
        vw1 = _mm256_loadu_si256((const __m256i *)(working_state + 8));  
        vw0 = _mm256_add_epi32(vw0, vs0);
        vw1 = _mm256_add_epi32(vw1, vs1);
        _mm256_storeu_si256((__m256i *)working_state, vw0);
        _mm256_storeu_si256((__m256i *)(working_state + 8), vw1);
        _mm256_storeu_si256((__m256i *)key_stream, vw0);
        _mm256_storeu_si256((__m256i *)(key_stream + 32), vw1);
        ks = _mm256_loadu_si256((const __m256i*)(key_stream ));
        buf = _mm256_loadu_si256((const __m256i*)(buffer + offset));
        res = _mm256_xor_si256(ks, buf);
        _mm256_storeu_si256((__m256i*)(buffer + offset), res);
        ks = _mm256_loadu_si256((const __m256i*)(key_stream + 32));
        buf = _mm256_loadu_si256((const __m256i*)(buffer + offset + 32));
        res = _mm256_xor_si256(ks, buf);
        _mm256_storeu_si256((__m256i*)(buffer + offset + 32), res);
    }
}