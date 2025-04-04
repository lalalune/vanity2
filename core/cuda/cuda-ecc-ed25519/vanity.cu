#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <vector>

#include <iostream>
#include <ctime>

#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "curand_kernel.h"
#include "ed25519.h"
#include "fixedint.h"
#include "gpu_common.h"
#include "gpu_ctx.h"

#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

#define MAX_PATTERN_LENGTH 10

/* -- Types ----------------------------------------------------------------- */

typedef struct {
	// CUDA Random States.
	curandState*    states[8];
    std::vector<std::string> patterns;
    int mode;
    int max_iterations;
    int stop_after_keys_found;
    int num_patterns;
    char dev_patterns[MAX_PATTERNS][MAX_PATTERN_LENGTH + 1];
    int dev_pattern_lengths[MAX_PATTERNS];
} config;

/* -- Prototypes, Because C++ ----------------------------------------------- */

void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void __global__ vanity_init(unsigned long long int* seed, curandState* state);
void __global__ vanity_scan(curandState* state, int* keys_found, int* gpu, int* execution_count,
                            const char (*patterns)[MAX_PATTERN_LENGTH + 1], const int* pattern_lengths, int num_patterns, int mode);
bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz);
bool parse_args(int argc, char const* argv[], config& vanity);

/* -- Entry Point ----------------------------------------------------------- */

int main(int argc, char const* argv[]) {
	ed25519_set_verbose(true);

	config vanity;
    vanity.mode = 0;
    vanity.max_iterations = 100000;
    vanity.stop_after_keys_found = 100;
    vanity.num_patterns = 0;

    if (!parse_args(argc, argv, vanity)) {
        fprintf(stderr, "Usage: %s --patterns <p1,p2,...> [--mode <prefix|suffix>] [--max_iterations <N>] [--stop_keys <K>]\\n", argv[0]);
        return 1;
    }

    printf("Configuration:\\n");
    printf("  Mode: %s\\n", vanity.mode == 0 ? "prefix" : "suffix");
    printf("  Max Iterations: %d\\n", vanity.max_iterations);
    printf("  Stop After Keys: %d\\n", vanity.stop_after_keys_found);
    printf("  Patterns (%d):\\n", vanity.num_patterns);
    for(int i = 0; i < vanity.num_patterns; ++i) {
        printf("    - %s (length %d)\\n", vanity.dev_patterns[i], vanity.dev_pattern_lengths[i]);
    }

	vanity_setup(vanity);
	vanity_run(vanity);
    return 0;
}

std::string getTimeStr(){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(30, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
}

unsigned long long int makeSeed() {
    unsigned long long int seed = 0;
    char *pseed = (char *)&seed;

    std::random_device rd;

    for(unsigned int b=0; b<sizeof(seed); b++) {
      auto r = rd();
      char *entropy = (char *)&r;
      pseed[b] = entropy[0];
    }

    return seed;
}

/* -- Vanity Step Functions ------------------------------------------------- */

void vanity_setup(config &vanity) {
	printf("GPU: Initializing Memory\n");
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	for (int i = 0; i < gpuCount; ++i) {
		cudaSetDevice(i);

		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, i);

		int blockSize       = 0,
		    minGridSize     = 0,
		    maxActiveBlocks = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

		printf("GPU: %d (%s <%d, %d, %d>) -- W: %d, P: %d, TPB: %d, MTD: (%dx, %dy, %dz), MGS: (%dx, %dy, %dz)\n",
			i,
			device.name,
			blockSize,
			minGridSize,
			maxActiveBlocks,
			device.warpSize,
			device.multiProcessorCount,
		       	device.maxThreadsPerBlock,
			device.maxThreadsDim[0],
			device.maxThreadsDim[1],
			device.maxThreadsDim[2],
			device.maxGridSize[0],
			device.maxGridSize[1],
			device.maxGridSize[2]
		);

		unsigned long long int rseed = makeSeed();
		printf("Initialising from entropy: %llu\n",rseed);

		unsigned long long int* dev_rseed;
	        cudaMalloc((void**)&dev_rseed, sizeof(unsigned long long int));		
                cudaMemcpy( dev_rseed, &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice ); 

		cudaMalloc((void **)&(vanity.states[i]), maxActiveBlocks * blockSize * sizeof(curandState));
		vanity_init<<<maxActiveBlocks, blockSize>>>(dev_rseed, vanity.states[i]);
	}

	printf("END: Initializing Memory\n");
}

void vanity_run(config &vanity) {
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	unsigned long long int  executions_total = 0;
	unsigned long long int  executions_this_iteration;
	int  executions_this_gpu;
        int* dev_executions_this_gpu[100];

        int  keys_found_total = 0;
        int  keys_found_this_iteration;
        int* dev_keys_found[100];

    char (*dev_patterns_gpu[100])[MAX_PATTERN_LENGTH + 1];
    int* dev_pattern_lengths_gpu[100];
    for (int g = 0; g < gpuCount; ++g) {
        cudaSetDevice(g);
        cudaMalloc((void**)&dev_patterns_gpu[g], sizeof(char[MAX_PATTERNS][MAX_PATTERN_LENGTH + 1]));
        cudaMalloc((void**)&dev_pattern_lengths_gpu[g], sizeof(int[MAX_PATTERNS]));
        cudaMemcpy(dev_patterns_gpu[g], vanity.dev_patterns, sizeof(char[MAX_PATTERNS][MAX_PATTERN_LENGTH + 1]), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_pattern_lengths_gpu[g], vanity.dev_pattern_lengths, sizeof(int[MAX_PATTERNS]), cudaMemcpyHostToDevice);
    }

	for (int i = 0; i < vanity.max_iterations; ++i) {
		auto start  = std::chrono::high_resolution_clock::now();

                executions_this_iteration=0;
                keys_found_this_iteration = 0;

		for (int g = 0; g < gpuCount; ++g) {
			cudaSetDevice(g);
			int blockSize       = 0,
			    minGridSize     = 0,
			    maxActiveBlocks = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (const void*)vanity_scan, 0, 0);
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, (const void*)vanity_scan, blockSize, 0);

			int* dev_g;
	                cudaMalloc((void**)&dev_g, sizeof(int));
                	cudaMemcpy( dev_g, &g, sizeof(int), cudaMemcpyHostToDevice );

	                cudaMalloc((void**)&dev_keys_found[g], sizeof(int));
                    cudaMemset(dev_keys_found[g], 0, sizeof(int));
	                cudaMalloc((void**)&dev_executions_this_gpu[g], sizeof(int));
                    cudaMemset(dev_executions_this_gpu[g], 0, sizeof(int));

			vanity_scan<<<maxActiveBlocks, blockSize>>>(
                vanity.states[g],
                dev_keys_found[g],
                dev_g,
                dev_executions_this_gpu[g],
                dev_patterns_gpu[g],
                dev_pattern_lengths_gpu[g],
                vanity.num_patterns,
                vanity.mode
            );
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "GPU %d Kernel launch error: %s\\n", g, cudaGetErrorString(err));
            }
            cudaFree(dev_g);

		}

		cudaDeviceSynchronize();
		auto finish = std::chrono::high_resolution_clock::now();

        int current_iter_keys = 0;
		for (int g = 0; g < gpuCount; ++g) {
                int keys_found_gpu = 0;
                cudaMemcpy( &keys_found_gpu, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost );
                current_iter_keys += keys_found_gpu;
                cudaFree(dev_keys_found[g]);

                int executions_gpu = 0;
                cudaMemcpy( &executions_gpu, dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost );
                executions_this_iteration += (unsigned long long int)executions_gpu * ATTEMPTS_PER_EXECUTION;
                cudaFree(dev_executions_this_gpu[g]);
		}
        keys_found_total += current_iter_keys;
        executions_total += executions_this_iteration;

		std::chrono::duration<double> elapsed = finish - start;
		printf("%s Iteration %d Attempts: %llu in %.3fs at %.2fcps - Found %d (Total Found %d, Total Attempts %llu)\n",
			getTimeStr().c_str(),
			i+1,
			executions_this_iteration,
			elapsed.count(),
            (executions_this_iteration > 0 && elapsed.count() > 0) ? (double)executions_this_iteration / elapsed.count() : 0.0,
            current_iter_keys,
			keys_found_total,
            executions_total
		);
        fflush(stdout);

        if ( keys_found_total >= vanity.stop_after_keys_found ) {
            printf("Found %d keys (>= %d), Done! \n", keys_found_total, vanity.stop_after_keys_found);
            fflush(stdout);
	        break;
		}
	}

    for (int g = 0; g < gpuCount; ++g) {
        cudaSetDevice(g);
        cudaFree(dev_patterns_gpu[g]);
        cudaFree(dev_pattern_lengths_gpu[g]);
    }

	printf("Iterations complete (%d total iterations run), Done!\n", vanity.max_iterations);
    fflush(stdout);
}

/* -- CUDA Vanity Functions ------------------------------------------------- */

void __global__ vanity_init(unsigned long long int* rseed, curandState* state) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);  
	curand_init(*rseed + id, id, 0, &state[id]);
}

void __global__ vanity_scan(curandState* state, int* keys_found, int* gpu, int* exec_count,
                            const char (*patterns)[MAX_PATTERN_LENGTH + 1], const int* pattern_lengths, int num_patterns, int mode)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x);

    atomicAdd(exec_count, 1);

	ge_p3 A;
	curandState localState     = state[id];
	unsigned char seed[32]     = {0};
	unsigned char public_key[32] = {0};

	for (unsigned int attempt = 0; attempt < ATTEMPTS_PER_EXECUTION; attempt++) {
		for (int i = 0; i < 32; i++) {
			seed[i] = curand(&localState);
		}

		ed25519_publickey(seed, public_key);

		char   b58[128];
		size_t b58sz = 128;
		b58enc(b58, &b58sz, public_key, 32);

		int b58_len = b58sz -1;
        bool match_found = false;
		for (int p_idx = 0; p_idx < num_patterns; ++p_idx) {
            int pattern_len = pattern_lengths[p_idx];
            if (pattern_len == 0 || pattern_len > b58_len) continue;

            bool current_match = false;
            if (mode == 0) {
                current_match = (strncmp(b58, patterns[p_idx], pattern_len) == 0);
            } else {
                current_match = (strncmp(b58 + b58_len - pattern_len, patterns[p_idx], pattern_len) == 0);
            }

			if (current_match) {
                match_found = true;
                atomicAdd(keys_found, 1);

                printf("MATCH: %s ", b58);
                for (int i = 0; i < 32; i++) printf("%02x", seed[i]);
                printf("\n");

                printf("[");
                for(int i=0; i<32; i++) printf("%d,", seed[i]);
                for(int i=0; i<31; i++) printf("%d,", public_key[i]);
                printf("%d", public_key[31]);
                printf("]\n");

                break;
			}
		}
	}

	state[id] = localState;
}

bool __device__ b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz) {
	const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

	const uint8_t *bin = data;
	int carry;
	size_t i, j, high, zcount = 0;
	size_t size;
	
	while (zcount < binsz && !bin[zcount])
		++zcount;
	
	size = (binsz - zcount) * 138 / 100 + 1;
	uint8_t buf[256];
	memset(buf, 0, size);
	
	for (i = zcount, high = size - 1; i < binsz; ++i, high = j)
	{
		for (carry = bin[i], j = size - 1; (j > high) || carry; --j)
		{
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (!j) {
				break;
			}
		}
	}
	
	for (j = 0; j < size && !buf[j]; ++j);
	
	if (*b58sz <= zcount + size - j) {
		*b58sz = zcount + size - j + 1;
		return false;
	}
	
	if (zcount) memset(b58, '1', zcount);
	for (i = zcount; j < size; ++i, ++j) b58[i] = b58digits_ordered[buf[j]];

	b58[i] = '\0';
	*b58sz = i + 1;
	
	return true;
}

bool parse_args(int argc, char const* argv[], config& vanity) {
    std::string patterns_str;
    bool patterns_found = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--patterns") == 0 && i + 1 < argc) {
            patterns_str = argv[++i];
            patterns_found = true;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "suffix") == 0) {
                vanity.mode = 1;
            } else if (strcmp(argv[i], "prefix") == 0) {
                vanity.mode = 0;
            } else {
                fprintf(stderr, "Error: Invalid mode '%s'. Use 'prefix' or 'suffix'.\\n", argv[i]);
                return false;
            }
        } else if (strcmp(argv[i], "--max_iterations") == 0 && i + 1 < argc) {
            vanity.max_iterations = atoi(argv[++i]);
            if (vanity.max_iterations <= 0) {
                 fprintf(stderr, "Error: Invalid max_iterations '%s'. Must be positive.\\n", argv[i]);
                 return false;
            }
        } else if (strcmp(argv[i], "--stop_keys") == 0 && i + 1 < argc) {
            vanity.stop_after_keys_found = atoi(argv[++i]);
             if (vanity.stop_after_keys_found <= 0) {
                 fprintf(stderr, "Error: Invalid stop_keys '%s'. Must be positive.\\n", argv[i]);
                 return false;
            }
        } else {
            fprintf(stderr, "Error: Unknown argument '%s'\\n", argv[i]);
            return false;
        }
    }

    if (!patterns_found) {
        fprintf(stderr, "Error: --patterns argument is required.\\n");
        return false;
    }

    std::stringstream ss(patterns_str);
    std::string pattern;
    int current_pattern = 0;
    while (std::getline(ss, pattern, ',')) {
        if (current_pattern >= MAX_PATTERNS) {
            fprintf(stderr, "Warning: Exceeded MAX_PATTERNS (%d). Ignoring further patterns.\\n", MAX_PATTERNS);
            break;
        }
        if (pattern.length() == 0) {
            fprintf(stderr, "Warning: Ignoring empty pattern.\\n");
            continue;
        }
        if (pattern.length() > MAX_PATTERN_LENGTH) {
             fprintf(stderr, "Error: Pattern '%s' exceeds MAX_PATTERN_LENGTH (%d).\\n", pattern.c_str(), MAX_PATTERN_LENGTH);
             return false;
        }
        strncpy(vanity.dev_patterns[current_pattern], pattern.c_str(), MAX_PATTERN_LENGTH);
        vanity.dev_patterns[current_pattern][pattern.length()] = '\0';
        vanity.dev_pattern_lengths[current_pattern] = pattern.length();
        vanity.patterns.push_back(pattern);
        current_pattern++;
    }
    vanity.num_patterns = current_pattern;

    if (vanity.num_patterns == 0) {
        fprintf(stderr, "Error: No valid patterns provided in --patterns string.\\n");
        return false;
    }

    return true;
}
