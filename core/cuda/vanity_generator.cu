#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

// Include necessary headers from the cuda-ecc-ed25519 library
#include "cuda-ecc-ed25519/ed25519.h"
#include "cuda-ecc-ed25519/sha512.h" // Assuming CreateKeyPair uses this internally or if needed separately
#include "cuda-ecc-ed25519/ge.h"     // Likely needed by ed25519.h
#include "cuda-ecc-ed25519/fe.h"     // Likely needed by ed25519.h
#include "cuda-ecc-ed25519/common.cu" // For common types/macros like uchar

// Define constants based on previous OpenCL kernel (adjust if needed)
#define MAX_PREFIXES 10 // Example: Max number of prefixes allowed
#define MAX_SUFFIXES 10 // Example: Max number of suffixes allowed
#define MAX_PREFIX_LEN 10 // Example: Max length of a prefix
#define MAX_SUFFIX_LEN 10 // Example: Max length of a suffix
#define PRIVATE_KEY_SIZE 32
#define PUBLIC_KEY_SIZE 32
#define SEED_SIZE 32
#define MAX_BASE58_LEN 45 // Max possible length for a 32-byte public key


// --- Ported Base58 Encoding Logic (from OpenCL kernel) ---
// Alphabet mapping (adjust if Solana uses a different one, but this is standard)
// Removed the alphabet_indices array, using direct mapping in encode function

// Base58 encoding function adapted for CUDA
// Input: 32-byte public key (in)
// Output: Base58 string (out), length (out_len)
__device__ void base58_encode_cuda(const unsigned char *in, unsigned int *out_len, unsigned char *out) {
    // Base58 alphabet
    const char b58_alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    unsigned int buffer[MAX_BASE58_LEN]; // Use integer buffer for calculations
    unsigned int digits = 0;

    // Count leading zeros
    unsigned int zeros = 0;
    while (zeros < PUBLIC_KEY_SIZE && in[zeros] == 0) {
        zeros++;
    }

    // Process non-zero bytes
    for (unsigned int i = zeros; i < PUBLIC_KEY_SIZE; i++) {
        unsigned int carry = (unsigned int)in[i];
        for (unsigned int j = 0; j < digits; j++) {
            carry += buffer[j] << 8; // Multiply by 256
            buffer[j] = carry % 58;
            carry /= 58;
        }
        while (carry > 0) {
            buffer[digits++] = carry % 58;
            carry /= 58;
        }
    }

    // Add leading '1's (representing zeros)
    for (unsigned int i = 0; i < zeros; i++) {
        out[i] = b58_alphabet[0]; // '1'
    }

    // Add encoded digits in reverse order
    for (unsigned int i = 0; i < digits; i++) {
        out[zeros + i] = b58_alphabet[buffer[digits - 1 - i]];
    }

    *out_len = zeros + digits;
    // Null-terminate for safety, though not strictly needed if length is used
    if (*out_len < MAX_BASE58_LEN) {
        out[*out_len] = '\0';
    } else {
         out[MAX_BASE58_LEN -1] = '\0'; // Ensure null termination within bounds
    }
}

// --- Ported Prefix/Suffix Checking Logic ---
// Adjusted CASE_SENSITIVE logic if needed, assuming case-sensitive for now
__device__ inline unsigned char adjust_case(unsigned char c) {
    // Simple ASCII lowercase conversion
    if (c >= 'A' && c <= 'Z') {
        return c + ('a' - 'A');
    }
    return c;
}


// Main Kernel Function
__global__ void find_vanity_address_kernel(
    curandState *states,
    unsigned char *results_out, // Output buffer for found private keys (SEED_SIZE * max_results)
    int *result_count,         // Atomic counter for number of results found
    const unsigned char* prefixes, // Flattened prefixes (num_prefixes * prefix_len)
    int prefix_len,
    int num_prefixes,
    const unsigned char* suffixes, // Flattened suffixes (num_suffixes * suffix_len)
    int suffix_len,
    int num_suffixes,
    unsigned long long kernel_launch_offset, // Base offset for this specific kernel launch
    int max_results,            // Stop searching after finding this many
    bool case_sensitive         // Flag for case sensitivity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long current_seed_offset = kernel_launch_offset + tid;

    // Initialize cuRAND state for this thread
    curandState localState = states[tid];

    // Local buffers
    unsigned char seed[SEED_SIZE];
    unsigned char public_key[PUBLIC_KEY_SIZE];
    unsigned char private_key[PRIVATE_KEY_SIZE]; // ed25519_CreateKeyPair takes private_key buffer but uses seed
    unsigned char base58_addr_buffer[MAX_BASE58_LEN];
    unsigned int base58_len = 0;

    // --- Generate Seed using cuRAND ---
    // Fill seed buffer with random bytes. Using float and scaling.
    // Adjust if a different cuRAND generation method is preferred.
    for (int i = 0; i < SEED_SIZE; i += 4) {
         float4 rand_f4 = curand_uniform4(&localState);
         unsigned int r1 = (unsigned int)(rand_f4.x * 256.0f);
         unsigned int r2 = (unsigned int)(rand_f4.y * 256.0f);
         unsigned int r3 = (unsigned int)(rand_f4.z * 256.0f);
         unsigned int r4 = (unsigned int)(rand_f4.w * 256.0f);
         if (i < SEED_SIZE) seed[i]   = (unsigned char)(r1 & 0xFF);
         if (i+1 < SEED_SIZE) seed[i+1] = (unsigned char)(r2 & 0xFF);
         if (i+2 < SEED_SIZE) seed[i+2] = (unsigned char)(r3 & 0xFF);
         if (i+3 < SEED_SIZE) seed[i+3] = (unsigned char)(r4 & 0xFF);
    }
    // Alternative: Generate one 64-bit value at a time using curand() if preferred

    // --- Derive Keypair using cuda-ecc-ed25519 library ---
    // NOTE: The library's CreateKeyPair likely handles the SHA512 hashing and bit clamping internally.
    // It expects seed as input and outputs pubkey and the *processed* private key.
    ed25519_CreateKeyPair(public_key, private_key, seed);

    // --- Encode Public Key to Base58 ---
    base58_encode_cuda(public_key, &base58_len, base58_addr_buffer);

    // --- Check Prefixes ---
    bool prefix_match = false;
    if (num_prefixes == 0) {
        prefix_match = true; // No prefixes means it matches by default
    } else {
        for (int p_idx = 0; p_idx < num_prefixes; ++p_idx) {
            bool current_prefix_mismatch = false;
            const unsigned char* current_prefix = prefixes + p_idx * prefix_len;
            if (base58_len < prefix_len) { // Cannot match if address is shorter than prefix
                 current_prefix_mismatch = true;
            } else {
                for (int i = 0; i < prefix_len; ++i) {
                    // Stop checking if prefix char is null terminator (flexible prefix length)
                    if (current_prefix[i] == '\0') break;

                    unsigned char addr_char = base58_addr_buffer[i];
                    unsigned char prefix_char = current_prefix[i];

                    if (!case_sensitive) {
                        addr_char = adjust_case(addr_char);
                        prefix_char = adjust_case(prefix_char);
                    }

                    if (addr_char != prefix_char) {
                        current_prefix_mismatch = true;
                        break; // Mismatch found for this prefix
                    }
                }
            }
            if (!current_prefix_mismatch) {
                prefix_match = true;
                break; // Found a matching prefix
            }
        }
    }

    // --- Check Suffixes ---
    bool suffix_match = false;
     if (!prefix_match) { // No need to check suffix if prefix didn't match
        suffix_match = false;
     } else if (num_suffixes == 0) {
        suffix_match = true; // No suffixes means it matches by default
    } else {
        for (int s_idx = 0; s_idx < num_suffixes; ++s_idx) {
            bool current_suffix_mismatch = false;
            const unsigned char* current_suffix = suffixes + s_idx * suffix_len;
            if (base58_len < suffix_len) { // Cannot match if address is shorter than suffix
                current_suffix_mismatch = true;
            } else {
                for (int i = 0; i < suffix_len; ++i) {
                     // Stop checking if suffix char is null terminator (flexible suffix length)
                    if (current_suffix[i] == '\0') break;

                    unsigned char addr_char = base58_addr_buffer[base58_len - suffix_len + i];
                    unsigned char suffix_char = current_suffix[i];

                     if (!case_sensitive) {
                        addr_char = adjust_case(addr_char);
                        suffix_char = adjust_case(suffix_char);
                    }

                    if (addr_char != suffix_char) {
                        current_suffix_mismatch = true;
                        break; // Mismatch found for this suffix
                    }
                }
            }
             if (!current_suffix_mismatch) {
                suffix_match = true;
                break; // Found a matching suffix
            }
        }
    }


    // --- Store Result if Match Found ---
    if (prefix_match && suffix_match) {
        int current_count = atomicAdd(result_count, 1);
        if (current_count < max_results) {
            // Copy the *original seed* to the output buffer
            unsigned char *result_slot = results_out + current_count * SEED_SIZE;
            for (int i = 0; i < SEED_SIZE; ++i) {
                result_slot[i] = seed[i];
            }
        } else {
             // Optional: Decrement count if we exceeded max_results due to race condition
             atomicSub(result_count, 1);
        }
    }

    // --- Update cuRAND state ---
    // Store the updated state back to global memory. Important!
     states[tid] = localState;
}

// Optional: Add a kernel for initializing cuRAND states if not done on host
__global__ void init_curand_states(unsigned long long seed, curandState *states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize state with a unique sequence for each thread
    curand_init(seed, tid, 0, &states[tid]);
} 