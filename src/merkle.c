#include "mercha.h"
#include <immintrin.h>


static inline void merkle_round_inline(__m128i *pA,__m128i *pB,__m128i *pC,__m128i *pD){
    register __m128i A asm("xmm0") = *pA;
    register __m128i B asm("xmm4") = *pB;
    register __m128i C asm("xmm3") = *pC;
    register __m128i D asm("xmm1") = *pD;
    asm volatile(
        ".p2align 4,,10                \n\t"
        ".p2align 3                    \n\t"
        "vpaddd   %%xmm3, %%xmm1, %%xmm3\n\t"  // C += D
        "vpaddd   %%xmm0, %%xmm4, %%xmm0\n\t"  // A += B
        "vpsrld   $25, %%xmm3, %%xmm6  \n\t"  // t6 = C >> 25
        "vpaddd   %%xmm4, %%xmm1, %%xmm4\n\t"  // B += D
        "vpsrld   $25, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 25
        "vpslld   $7,  %%xmm3, %%xmm3  \n\t"  // C <<= 7
        "vpslld   $7,  %%xmm0, %%xmm0  \n\t"  // A <<= 7
        "vpor     %%xmm6, %%xmm3, %%xmm3\n\t"  // C = ROTL(C,7)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,7)
        "vpsrld   $23, %%xmm4, %%xmm6  \n\t"  // t6 = B >> 23
        "vpaddd   %%xmm0, %%xmm3, %%xmm0\n\t"  // A += C
        "vpslld   $9,  %%xmm4, %%xmm4  \n\t"  // B <<= 9
        "vpsrld   $23, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 23
        "vpslld   $9,  %%xmm0, %%xmm0  \n\t"  // A <<= 9
        "vpor     %%xmm6, %%xmm4, %%xmm4\n\t"  // B = ROTL(B,9)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,9)
        "vpaddd   %%xmm3, %%xmm1, %%xmm3\n\t"  // C += D
        "vpaddd   %%xmm0, %%xmm4, %%xmm0\n\t"  // A += B
        "vpsrld   $25, %%xmm3, %%xmm6  \n\t"  // t6 = C >> 25
        "vpaddd   %%xmm4, %%xmm1, %%xmm4\n\t"  // B += D
        "vpsrld   $25, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 25
        "vpslld   $7,  %%xmm3, %%xmm3  \n\t"  // C <<= 7
        "vpslld   $7,  %%xmm0, %%xmm0  \n\t"  // A <<= 7
        "vpor     %%xmm6, %%xmm3, %%xmm3\n\t"  // C = ROTL(C,7)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,7)
        "vpsrld   $23, %%xmm4, %%xmm6  \n\t"  // t6 = B >> 23
        "vpaddd   %%xmm0, %%xmm3, %%xmm0\n\t"  // A += C
        "vpslld   $9,  %%xmm4, %%xmm4  \n\t"  // B <<= 9
        "vpsrld   $23, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 23
        "vpslld   $9,  %%xmm0, %%xmm0  \n\t"  // A <<= 9
        "vpor     %%xmm6, %%xmm4, %%xmm4\n\t"  // B = ROTL(B,9)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,9)
        "vpaddd   %%xmm3, %%xmm1, %%xmm3\n\t"  // C += D
        "vpaddd   %%xmm0, %%xmm4, %%xmm0\n\t"  // A += B
        "vpsrld   $25, %%xmm3, %%xmm6  \n\t"  // t6 = C >> 25
        "vpaddd   %%xmm4, %%xmm1, %%xmm4\n\t"  // B += D
        "vpsrld   $25, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 25
        "vpslld   $7,  %%xmm3, %%xmm3  \n\t"  // C <<= 7
        "vpslld   $7,  %%xmm0, %%xmm0  \n\t"  // A <<= 7
        "vpor     %%xmm6, %%xmm3, %%xmm3\n\t"  // C = ROTL(C,7)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,7)
        "vpsrld   $23, %%xmm4, %%xmm6  \n\t"  // t6 = B >> 23
        "vpaddd   %%xmm0, %%xmm3, %%xmm0\n\t"  // A += C
        "vpslld   $9,  %%xmm4, %%xmm4  \n\t"  // B <<= 9
        "vpsrld   $23, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 23
        "vpslld   $9,  %%xmm0, %%xmm0  \n\t"  // A <<= 9
        "vpor     %%xmm6, %%xmm4, %%xmm4\n\t"  // B = ROTL(B,9)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,9)
        "vpaddd   %%xmm3, %%xmm1, %%xmm3\n\t"  // C += D
        "vpaddd   %%xmm0, %%xmm4, %%xmm0\n\t"  // A += B
        "vpsrld   $25, %%xmm3, %%xmm6  \n\t"  // t6 = C >> 25
        "vpaddd   %%xmm4, %%xmm1, %%xmm4\n\t"  // B += D
        "vpsrld   $25, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 25
        "vpslld   $7,  %%xmm3, %%xmm3  \n\t"  // C <<= 7
        "vpslld   $7,  %%xmm0, %%xmm0  \n\t"  // A <<= 7
        "vpor     %%xmm6, %%xmm3, %%xmm3\n\t"  // C = ROTL(C,7)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,7)
        "vpsrld   $23, %%xmm4, %%xmm6  \n\t"  // t6 = B >> 23
        "vpaddd   %%xmm0, %%xmm3, %%xmm0\n\t"  // A += C
        "vpslld   $9,  %%xmm4, %%xmm4  \n\t"  // B <<= 9
        "vpsrld   $23, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 23
        "vpslld   $9,  %%xmm0, %%xmm0  \n\t"  // A <<= 9
        "vpor     %%xmm6, %%xmm4, %%xmm4\n\t"  // B = ROTL(B,9)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,9)
        "vpaddd   %%xmm3, %%xmm1, %%xmm3\n\t"  // C += D
        "vpaddd   %%xmm0, %%xmm4, %%xmm0\n\t"  // A += B
        "vpsrld   $25, %%xmm3, %%xmm6  \n\t"  // t6 = C >> 25
        "vpaddd   %%xmm4, %%xmm1, %%xmm4\n\t"  // B += D
        "vpsrld   $25, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 25
        "vpslld   $7,  %%xmm3, %%xmm3  \n\t"  // C <<= 7
        "vpslld   $7,  %%xmm0, %%xmm0  \n\t"  // A <<= 7
        "vpor     %%xmm6, %%xmm3, %%xmm3\n\t"  // C = ROTL(C,7)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,7)
        "vpsrld   $23, %%xmm4, %%xmm6  \n\t"  // t6 = B >> 23
        "vpaddd   %%xmm0, %%xmm3, %%xmm0\n\t"  // A += C
        "vpslld   $9,  %%xmm4, %%xmm4  \n\t"  // B <<= 9
        "vpsrld   $23, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 23
        "vpslld   $9,  %%xmm0, %%xmm0  \n\t"  // A <<= 9
        "vpor     %%xmm6, %%xmm4, %%xmm4\n\t"  // B = ROTL(B,9)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,9)
        "vpaddd   %%xmm3, %%xmm1, %%xmm3\n\t"  // C += D
        "vpaddd   %%xmm0, %%xmm4, %%xmm0\n\t"  // A += B
        "vpsrld   $25, %%xmm3, %%xmm6  \n\t"  // t6 = C >> 25
        "vpaddd   %%xmm4, %%xmm1, %%xmm4\n\t"  // B += D
        "vpsrld   $25, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 25
        "vpslld   $7,  %%xmm3, %%xmm3  \n\t"  // C <<= 7
        "vpslld   $7,  %%xmm0, %%xmm0  \n\t"  // A <<= 7
        "vpor     %%xmm6, %%xmm3, %%xmm3\n\t"  // C = ROTL(C,7)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,7)
        "vpsrld   $23, %%xmm4, %%xmm6  \n\t"  // t6 = B >> 23
        "vpaddd   %%xmm0, %%xmm3, %%xmm0\n\t"  // A += C
        "vpslld   $9,  %%xmm4, %%xmm4  \n\t"  // B <<= 9
        "vpsrld   $23, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 23
        "vpslld   $9,  %%xmm0, %%xmm0  \n\t"  // A <<= 9
        "vpor     %%xmm6, %%xmm4, %%xmm4\n\t"  // B = ROTL(B,9)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,9)
        "vpaddd   %%xmm3, %%xmm1, %%xmm3\n\t"  // C += D
        "vpaddd   %%xmm0, %%xmm4, %%xmm0\n\t"  // A += B
        "vpsrld   $25, %%xmm3, %%xmm6  \n\t"  // t6 = C >> 25
        "vpaddd   %%xmm4, %%xmm1, %%xmm4\n\t"  // B += D
        "vpsrld   $25, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 25
        "vpslld   $7,  %%xmm3, %%xmm3  \n\t"  // C <<= 7
        "vpslld   $7,  %%xmm0, %%xmm0  \n\t"  // A <<= 7
        "vpor     %%xmm6, %%xmm3, %%xmm3\n\t"  // C = ROTL(C,7)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,7)
        "vpsrld   $23, %%xmm4, %%xmm6  \n\t"  // t6 = B >> 23
        "vpaddd   %%xmm0, %%xmm3, %%xmm0\n\t"  // A += C
        "vpslld   $9,  %%xmm4, %%xmm4  \n\t"  // B <<= 9
        "vpsrld   $23, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 23
        "vpslld   $9,  %%xmm0, %%xmm0  \n\t"  // A <<= 9
        "vpor     %%xmm6, %%xmm4, %%xmm4\n\t"  // B = ROTL(B,9)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,9)
        "vpaddd   %%xmm3, %%xmm1, %%xmm3\n\t"  // C += D
        "vpaddd   %%xmm0, %%xmm4, %%xmm0\n\t"  // A += B
        "vpsrld   $25, %%xmm3, %%xmm6  \n\t"  // t6 = C >> 25
        "vpaddd   %%xmm4, %%xmm1, %%xmm4\n\t"  // B += D
        "vpsrld   $25, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 25
        "vpslld   $7,  %%xmm3, %%xmm3  \n\t"  // C <<= 7
        "vpslld   $7,  %%xmm0, %%xmm0  \n\t"  // A <<= 7
        "vpor     %%xmm6, %%xmm3, %%xmm3\n\t"  // C = ROTL(C,7)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,7)
        "vpsrld   $23, %%xmm4, %%xmm6  \n\t"  // t6 = B >> 23
        "vpaddd   %%xmm0, %%xmm3, %%xmm0\n\t"  // A += C
        "vpslld   $9,  %%xmm4, %%xmm4  \n\t"  // B <<= 9
        "vpsrld   $23, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 23
        "vpslld   $9,  %%xmm0, %%xmm0  \n\t"  // A <<= 9
        "vpor     %%xmm6, %%xmm4, %%xmm4\n\t"  // B = ROTL(B,9)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,9)
        "vpaddd   %%xmm3, %%xmm1, %%xmm3\n\t"  // C += D
        "vpaddd   %%xmm0, %%xmm4, %%xmm0\n\t"  // A += B
        "vpsrld   $25, %%xmm3, %%xmm6  \n\t"  // t6 = C >> 25
        "vpaddd   %%xmm4, %%xmm1, %%xmm4\n\t"  // B += D
        "vpsrld   $25, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 25
        "vpslld   $7,  %%xmm3, %%xmm3  \n\t"  // C <<= 7
        "vpslld   $7,  %%xmm0, %%xmm0  \n\t"  // A <<= 7
        "vpor     %%xmm6, %%xmm3, %%xmm3\n\t"  // C = ROTL(C,7)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,7)
        "vpsrld   $23, %%xmm4, %%xmm6  \n\t"  // t6 = B >> 23
        "vpaddd   %%xmm0, %%xmm3, %%xmm0\n\t"  // A += C
        "vpslld   $9,  %%xmm4, %%xmm4  \n\t"  // B <<= 9
        "vpsrld   $23, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 23
        "vpslld   $9,  %%xmm0, %%xmm0  \n\t"  // A <<= 9
        "vpor     %%xmm6, %%xmm4, %%xmm4\n\t"  // B = ROTL(B,9)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,9)
        "vpaddd   %%xmm3, %%xmm1, %%xmm3\n\t"  // C += D
        "vpaddd   %%xmm0, %%xmm4, %%xmm0\n\t"  // A += B
        "vpsrld   $25, %%xmm3, %%xmm6  \n\t"  // t6 = C >> 25
        "vpaddd   %%xmm4, %%xmm1, %%xmm4\n\t"  // B += D
        "vpsrld   $25, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 25
        "vpslld   $7,  %%xmm3, %%xmm3  \n\t"  // C <<= 7
        "vpslld   $7,  %%xmm0, %%xmm0  \n\t"  // A <<= 7
        "vpor     %%xmm6, %%xmm3, %%xmm3\n\t"  // C = ROTL(C,7)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,7)
        "vpsrld   $23, %%xmm4, %%xmm6  \n\t"  // t6 = B >> 23
        "vpaddd   %%xmm0, %%xmm3, %%xmm0\n\t"  // A += C
        "vpslld   $9,  %%xmm4, %%xmm4  \n\t"  // B <<= 9
        "vpsrld   $23, %%xmm0, %%xmm5  \n\t"  // t5 = A >> 23
        "vpslld   $9,  %%xmm0, %%xmm0  \n\t"  // A <<= 9
        "vpor     %%xmm6, %%xmm4, %%xmm4\n\t"  // B = ROTL(B,9)
        "vpor     %%xmm5, %%xmm0, %%xmm0\n\t"  // A = ROTL(A,9)
        : "+x"(A), "+x"(B), "+x"(C), "+x"(D)
        :
        : "xmm5", "xmm6", "memory"
    );
    *pA = A;
    *pB = B;
    *pC = C;
    *pD = D;
}
void merkel_tree(const uint8_t *input, uint8_t *output, size_t length){
    uint8_t * cur_buf  = malloc(length);
    uint8_t * prev_buf = (uint8_t *)input;
    length /= 2;
    uint32_t state[16] = {0};
    while (length>=64) {
        #pragma omp parallel for private(state) schedule(static) proc_bind(spread)
        for (int i=0; i<length/64; ++i){
            const uint8_t *block1 = prev_buf + (2 * i) * 64;
            const uint8_t *block2 = prev_buf + (2 * i + 1) * 64;
            uint8_t *out_ptr = cur_buf + i * 64;
        
            __m256i w1 = _mm256_loadu_si256((const __m256i *)block1);
            __m256i w2 = _mm256_loadu_si256((const __m256i *)block2);
            __m256i rev_mask = _mm256_setr_epi32(7,6,5,4,3,2,1,0);
            __m256i w2_rev = _mm256_permutevar8x32_epi32(w2, rev_mask);
            __m256i w1_rev = _mm256_permutevar8x32_epi32(w1, rev_mask);
            __m256i state0 = _mm256_xor_si256(w1, w2_rev);
            __m256i state1 = _mm256_xor_si256(w2, w1_rev);
            
            _mm256_storeu_si256((__m256i *)state, state0);
            _mm256_storeu_si256((__m256i *)(state + 8), state1);
            
            __m128i v0 = _mm_loadu_si128((__m128i *)&state[0]);
            __m128i v1 = _mm_loadu_si128((__m128i *)&state[4]);
            __m128i v2 = _mm_loadu_si128((__m128i *)&state[8]);
            __m128i v3 = _mm_loadu_si128((__m128i *)&state[12]);
            merkle_round_inline(&v0, &v1, &v2, &v3);          
            _mm_storeu_si128((__m128i *)&state[0], v0);
            _mm_storeu_si128((__m128i *)&state[4], v1);
            _mm_storeu_si128((__m128i *)&state[8], v2);
            
            __m256i vLow  = _mm256_loadu_si256((const __m256i *)state);
            __m256i vHigh = _mm256_loadu_si256((const __m256i *)(state + 8));
            vHigh = _mm256_permutevar8x32_epi32(vHigh, rev_mask);
            vLow = _mm256_add_epi32(vLow, vHigh);
            _mm256_storeu_si256((__m256i *)state, vLow);
            memcpy(out_ptr, state, 64);
        }
        length /= 2;
        uint8_t *tmp = cur_buf;
        cur_buf = prev_buf;
        prev_buf = tmp;
    }
    memcpy(output, prev_buf, 64);
    if(input == prev_buf){
        free(cur_buf);
    }else{
        free(prev_buf);
    };
}