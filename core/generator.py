import logging
import multiprocessing
from multiprocessing.pool import Pool
from typing import List, Dict, Any
import time
import multiprocessing as mp
import cupy
from loguru import logger
# Explicitly import Queue and Event for type hinting if needed
from multiprocessing import Queue, Event
# Import the Empty exception for queue.get timeout
from queue import Empty

from core.config import CudaHostSetting, DEFAULT_CUDA_BATCH_SIZE # Use new CUDA config
# Removed OpenCL imports
from core.searcher import CudaSearcher # Use refactored searcher and CudaSearcher
from core.utils.helpers import check_character # Keep validation

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")

# Set start method for multiprocessing (if needed, 'fork' is default on Linux)
# mp.set_start_method('spawn', force=True) # Use 'spawn' if 'fork' causes CUDA issues

def worker_process(
    gpu_id: int,
    prefixes: List[str],
    suffixes: List[str],
    case_sensitive: bool,
    target_count_per_worker: int, # How many keys this worker should aim for
    result_queue: 'Queue',         # Use string type hint
    stop_event: 'Event',           # Use string type hint
    batch_size: int
):
    """Target function for each GPU worker process."""
    logger.info(f"Starting worker process for GPU {gpu_id}")
    found_count_local = 0
    try:
        # Each process gets its own searcher instance
        searcher = CudaSearcher(prefixes, suffixes, case_sensitive)

        # generate_vanity_keys now loops internally until target_count_per_worker is potentially met
        found_keys = searcher.generate_vanity_keys(
            gpu_id=gpu_id,
            batch_size=batch_size,
            # Pass a target count, but the main process controls the overall stop
            target_count=target_count_per_worker
        )
        found_count_local = len(found_keys)

        # Put found keys onto the queue
        for key_info in found_keys:
            if not stop_event.is_set(): # Check stop event before putting
                result_queue.put(key_info)
            else:
                logger.info(f"GPU {gpu_id}: Stop event set, discarding remaining found key(s).")
                break # Stop sending results if told to stop

        logger.info(f"Worker for GPU {gpu_id} finished search, found {found_count_local} keys locally.")

    except FileNotFoundError as e:
        logger.error(f"GPU {gpu_id}: Worker failed - file not found: {e}. Check CUDA setup, kernel/lib paths in manager.py.")
        # Signal error state if needed, e.g., result_queue.put(Exception("File Error"))
    except RuntimeError as e:
         logger.error(f"GPU {gpu_id}: Worker failed - runtime error (CUDA setup/exec?): {e}")
         # Kernel compilation or loading likely failed
    except Exception as e:
        logger.exception(f"GPU {gpu_id}: Worker failed with unexpected error: {e}")
    finally:
        # Ensure the process signals it's done, even on error, to avoid hangs
        # Putting None signifies completion or error for this worker
        result_queue.put(None)
        logger.info(f"Worker process for GPU {gpu_id} exiting.")


def find_vanity_addresses(
    prefixes: List[str],
    suffixes: List[str],
    count: int = 1,
    batch_size: int = DEFAULT_CUDA_BATCH_SIZE,
    case_sensitive: bool = True
) -> List[Dict[str, str]]:
    """Finds Solana vanity addresses using available CUDA GPUs via multiprocessing."""
    start_total_time = time.time()
    all_found_results: List[Dict[str, str]] = []

    # Validate inputs before starting processes
    if not prefixes and not suffixes:
         raise ValueError("Must provide at least one prefix or suffix.")
    # Add character validation if needed (moved from old generator)
    # from core.utils.helpers import check_character
    # for p in prefixes: check_character("prefix", p)
    # for s in suffixes: check_character("suffix", s)

    try:
        num_gpus = cupy.cuda.runtime.getDeviceCount()
        if num_gpus == 0:
            logger.error("No CUDA-enabled GPUs found.")
            return []
        logger.info(f"Found {num_gpus} CUDA device(s).")
    except cupy.cuda.runtime.CUDARuntimeError as e:
        logger.error(f"CUDA runtime error during device count: {e}")
        logger.error("Ensure CUDA toolkit is installed and drivers are up to date.")
        return []
    except Exception as e:
         logger.error(f"Error detecting CUDA devices: {e}")
         return []

    # --- Multiprocessing Setup ---
    processes = []
    # Use a standard multiprocessing Queue and Event
    result_queue = mp.Queue()
    stop_event = mp.Event()
    # Don't spawn more workers than needed GPUs or target keys
    workers_to_spawn = min(num_gpus, count)
    if workers_to_spawn <= 0:
         logger.warning("Target count is 0 or less, no search needed.")
         return []

    # Distribute target count (approximately)
    # Give each worker a slightly higher target to compensate for uneven speeds
    # and ensure the main loop collects enough even if one worker is slow.
    target_per_worker = (count + workers_to_spawn - 1) // workers_to_spawn + 1 # Aim slightly higher
    logger.info(f"Spawning {workers_to_spawn} worker processes, each aiming for up to {target_per_worker} keys.")

    for gpu_id in range(workers_to_spawn):
        process = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                prefixes,
                suffixes,
                case_sensitive,
                target_per_worker, # Target for this worker
                result_queue,
                stop_event,
                batch_size
            ),
            daemon=True # Make workers daemonic so they exit if main process exits
        )
        processes.append(process)
        process.start()
        logger.info(f"Launched worker for GPU {gpu_id} (PID: {process.pid})")

    # --- Collect Results ---
    completed_workers = 0
    while completed_workers < workers_to_spawn:
        try:
            # Get result from the queue, block until available
            result = result_queue.get(timeout=1) # Use timeout to prevent indefinite block

            if result is None:
                completed_workers += 1 # Worker finished (or errored out)
                logger.debug(f"Worker signal received. Completed: {completed_workers}/{workers_to_spawn}")
            elif isinstance(result, dict):
                if len(all_found_results) < count:
                    all_found_results.append(result)
                    logger.info(f"Collected result #{len(all_found_results)}/{count}: {result.get('address')}")
                    if len(all_found_results) >= count:
                        logger.info(f"Target count ({count}) reached. Signaling workers to stop.")
                        stop_event.set() # Signal all other workers to stop
                        # Keep draining queue to allow workers to exit cleanly
                else:
                     # Already have enough results, but log if more come in
                      logger.debug(f"Received extra result: {result.get('address')} after target met.")

            else:
                 logger.warning(f"Received unexpected item from queue: {type(result)}")

        except Empty: # Use the imported Empty exception
            # Timeout occurred, check if processes are still alive
            all_dead = True
            for p in processes:
                if p.is_alive():
                    all_dead = False
                    break

            if stop_event.is_set() and all_dead:
                 logger.info("All workers terminated after stop signal.")
                 break # Exit if stop was set and workers are done
            elif not all_dead:
                 # logger.debug("Queue empty, workers still running...")
                 pass # Workers still running, continue waiting
            else:
                 logger.warning("Queue empty and all workers seem finished, but completion signals not all received.")
                 completed_workers = workers_to_spawn # Assume they are done to break loop
                 break # Avoid potential infinite loop

        except Exception as e:
            logger.exception(f"Error retrieving from result queue: {e}")
            stop_event.set() # Signal stop on unexpected queue error
            break

    # --- Cleanup ---
    logger.info("Waiting for worker processes to terminate cleanly...")
    active_workers = workers_to_spawn
    start_join_time = time.time()
    join_timeout = 30 # Max seconds to wait for join

    while active_workers > 0 and (time.time() - start_join_time) < join_timeout:
        active_workers = 0
        for i, process in enumerate(processes):
             if process.is_alive():
                  process.join(timeout=0.1) # Short non-blocking join attempt
                  if process.is_alive():
                       active_workers += 1
        if active_workers > 0:
             time.sleep(0.5) # Wait a bit before checking again

    # Force terminate any remaining processes
    for i, process in enumerate(processes):
        if process.is_alive():
            logger.warning(f"Process {process.pid} (GPU {i}) did not terminate gracefully after {join_timeout}s. Forcing termination.")
            try:
                process.terminate()
                process.join(timeout=2) # Brief wait after terminate
            except Exception as term_err:
                 logger.error(f"Error terminating process {process.pid}: {term_err}")


    end_total_time = time.time()
    logger.info(
        f"Total execution time: {end_total_time - start_total_time:.2f} seconds. "
        f"Found {len(all_found_results)} keys (requested {count})."
    )

    # Return only the requested number of keys
    return all_found_results[:count] 