import logging
import time
from typing import List, Dict, Any, Tuple

import cupy

from core.config import CudaHostSetting, DEFAULT_CUDA_BATCH_SIZE
from core.cuda.manager import cuda_manager
from core.utils.helpers import check_character # Keep for prefix/suffix validation
from base58 import b58encode # Use installed base58 library
from solders.keypair import Keypair # Use solders for CPU verification

# Constants from the CUDA kernel (ensure they match)
SEED_SIZE = 32

class CudaSearcher:
    """Manages CUDA key generation and CPU-side validation for a single GPU."""
    def __init__(self, prefixes: List[str], suffixes: List[str], case_sensitive: bool = True):
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.case_sensitive = case_sensitive

        # Get CUDA kernel via manager
        try:
            self.kernel = cuda_manager.get_kernel()
            self.init_kernel = cuda_manager.get_init_kernel()
            logging.info("Successfully retrieved CUDA kernels.")
        except Exception as e:
            logging.error(f"Failed to get CUDA kernels: {e}")
            # Propagate or handle error appropriately
            raise

        # Prepare prefixes/suffixes for GPU
        self.gpu_prefixes, self.prefix_len, self.num_prefixes = self._prepare_affixes_for_gpu(prefixes)
        self.gpu_suffixes, self.suffix_len, self.num_suffixes = self._prepare_affixes_for_gpu(suffixes)

    def _prepare_affixes_for_gpu(self, affixes: List[str]) -> Tuple[cupy.ndarray | None, int, int]:
        """Pads and flattens affixes, transfers to GPU."""
        if not affixes:
            return None, 0, 0

        max_len = max(len(aff) for aff in affixes)
        num_affixes = len(affixes)

        # Create a flattened numpy array with padding
        # Use uint8 for individual characters
        host_array = cupy.zeros((num_affixes, max_len), dtype=cupy.uint8)
        for i, affix in enumerate(affixes):
            try:
                encoded = affix.encode('ascii') # Assuming ASCII affixes
            except UnicodeEncodeError:
                 logging.error(f"Affix '{affix}' contains non-ASCII characters. Skipping affix.")
                 # Or raise error depending on desired behavior
                 continue # Skip this affix
            host_array[i, :len(encoded)] = cupy.array(list(encoded), dtype=cupy.uint8)

        # Transfer to GPU
        gpu_array = host_array.ravel() # Flatten
        logging.debug(f"Prepared {num_affixes} affixes with max_len {max_len} for GPU.")
        return gpu_array, max_len, num_affixes

    def generate_vanity_keys(self,
                             gpu_id: int,
                             batch_size: int = DEFAULT_CUDA_BATCH_SIZE,
                             target_count: int = 1) -> List[Dict[str, str]]:
        """Generates vanity keys on a specific GPU using the all-in-one CUDA kernel."""
        results = []
        start_time = time.time()

        try:
            cupy.cuda.Device(gpu_id).use()
            logging.info(f"Using GPU {gpu_id}")

            # --- Prepare GPU Memory ---
            num_threads = 256 # Common block size
            # Grid size should cover *at least* target_count potentially, or a fixed large batch
            # Let's aim for a large number of threads to maximize GPU usage per call
            # Use a fixed large batch size for the kernel launch grid, independent of target_count
            kernel_batch_size = 1 << 24 # e.g., 16 million attempts per launch
            num_blocks = (kernel_batch_size + num_threads - 1) // num_threads
            total_threads = num_blocks * num_threads
            logging.debug(f"GPU {gpu_id}: Grid: {num_blocks} blocks, {num_threads} threads/block ({total_threads} total threads for kernel launch)")

            # WARNING: Assuming curandState size. This is brittle.
            state_size_bytes = 128 # Guessed size
            states_gpu = cupy.empty(total_threads * state_size_bytes, dtype=cupy.uint8)

            # Results buffer (stores found seeds, size based on target_count)
            results_gpu = cupy.zeros(target_count * SEED_SIZE, dtype=cupy.uint8)
            # Atomic counter for results
            result_count_gpu = cupy.zeros(1, dtype=cupy.int32)

            # Affixes (already prepared in __init__)
            prefixes_gpu = self.gpu_prefixes if self.gpu_prefixes is not None else cupy.empty(0, dtype=cupy.uint8)
            suffixes_gpu = self.gpu_suffixes if self.gpu_suffixes is not None else cupy.empty(0, dtype=cupy.uint8)

            # --- Initialize cuRAND States ---
            init_seed = int(time.time() * 1000) + gpu_id # Simple unique seed per GPU
            logging.info(f"GPU {gpu_id}: Initializing {total_threads} cuRAND states with seed {init_seed}...")
            self.init_kernel((num_blocks,), (num_threads,), (init_seed, states_gpu))
            cupy.cuda.runtime.deviceSynchronize() # Ensure init is complete
            logging.info(f"GPU {gpu_id}: cuRAND states initialized.")

            # --- Main Generation Kernel Launch ---
            total_attempts = 0
            kernel_launch_offset = 0 # Not strictly needed if seed is randomized

            logging.info(f"GPU {gpu_id}: Starting kernel launch for up to {target_count} address(es) with {total_threads} threads...")
            self.kernel(
                (num_blocks,), (num_threads,),
                (states_gpu,                  # cuRAND states
                 results_gpu,               # Output buffer for seeds
                 result_count_gpu,          # Atomic counter
                 prefixes_gpu,              # Prefixes data
                 self.prefix_len,           # Length of each prefix (padded)
                 self.num_prefixes,         # Number of prefixes
                 suffixes_gpu,              # Suffixes data
                 self.suffix_len,           # Length of each suffix (padded)
                 self.num_suffixes,         # Number of suffixes
                 kernel_launch_offset,      # Offset for this launch
                 target_count,              # Max results goal for this launch
                 self.case_sensitive        # Case sensitivity flag
                 )
            )
            # Kernel is asynchronous, wait for completion
            cupy.cuda.runtime.deviceSynchronize()
            total_attempts = total_threads # Record the number of threads launched

            # --- Retrieve Results ---
            final_count = result_count_gpu.get()[0]
            logging.info(f"Search on GPU {gpu_id} finished. Found {final_count} potential keys after {total_attempts} attempts.")

            if final_count > 0:
                # Ensure we don't read more than allocated or requested
                read_count = min(final_count, target_count)
                found_seeds_host = results_gpu[:read_count * SEED_SIZE].get()

                # Format results (re-deriving pubkey/address on CPU for verification/display)
                # This part might be skipped if only the seed is needed
                for i in range(read_count):
                    seed_bytes = bytes(found_seeds_host[i * SEED_SIZE : (i + 1) * SEED_SIZE])
                    try:
                        kp = Keypair.from_seed(seed_bytes)
                        address = str(kp.pubkey())
                        # Optional: Double-check prefix/suffix match on CPU
                        addr_lower = address.lower()
                        prefix_ok = True
                        if self.prefixes:
                             prefix_ok = any(addr_lower.startswith(p.lower()) for p in self.prefixes) if not self.case_sensitive else any(address.startswith(p) for p in self.prefixes)
                        suffix_ok = True
                        if self.suffixes:
                             suffix_ok = any(addr_lower.endswith(s.lower()) for s in self.suffixes) if not self.case_sensitive else any(address.endswith(s) for s in self.suffixes)

                        if prefix_ok and suffix_ok:
                             results.append({
                                 "seed": seed_bytes.hex(), # Store seed as hex
                                 "address": address,
                                 "private_key": b58encode(kp.secret()[:32]), # Base58 encode the private key part
                             })
                             logging.info(f"Found valid key: {address}")
                        else:
                             logging.warning(f"GPU reported match for seed {seed_bytes.hex()} but CPU check failed for address {address}.")

                    except Exception as e:
                         logging.error(f"Error processing seed {seed_bytes.hex()} on CPU: {e}")

        except cupy.cuda.runtime.CUDARuntimeError as e:
            logging.error(f"CUDA Runtime Error on GPU {gpu_id}: {e}")
            # Handle GPU errors (e.g., log, skip GPU, raise)
        except Exception as e:
            logging.error(f"Unexpected error during GPU processing {gpu_id}: {e}")
            # Handle other errors

        end_time = time.time()
        logging.info(f"GPU {gpu_id} processing took {end_time - start_time:.2f} seconds.")
        return results[:target_count] # Return only up to the target count

# Removed multi_gpu_init as the orchestration is now in generator.py
# The CudaSearcher instance will be created per process in generator.py
