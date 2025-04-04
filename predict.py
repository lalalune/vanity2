#!/usr/bin/env python
from typing import List, Dict, Any
import time
import logging
import multiprocessing

from cog import BasePredictor, Input, Path
from loguru import logger # Keep loguru if preferred for logging

# Import the core generator function
from core.generator import find_vanity_addresses
from core.config import DEFAULT_CUDA_BATCH_SIZE

class Predictor(BasePredictor):
    def setup(self):
        """
        Initialize multiprocessing context if needed.
        Potentially pre-initialize OpenCL devices here if beneficial,
        though the generator function also handles initialization.
        """
        logger.info("Setting up predictor...")
        # Ensure spawn start method is used for multiprocessing with CUDA/OpenCL
        # This might be redundant if set globally, but good to ensure here.
        try:
            multiprocessing.set_start_method("spawn", force=True)
            logger.info("Multiprocessing start method set to spawn.")
        except RuntimeError as e:
            # Handles case where it might already be set
            logger.warning(f"Could not set multiprocessing start method: {e}")
        # Optional: Add a check to ensure CuPy can see the GPU
        try:
            import cupy
            device_count = cupy.cuda.runtime.getDeviceCount()
            logger.info(f"Found {device_count} CUDA devices.")
            if device_count == 0:
                 logger.warning("No CUDA devices detected by CuPy during setup!")
        except ImportError:
            logger.error("CuPy not installed, CUDA acceleration unavailable.")
        except Exception as e:
            logger.error(f"Error during CuPy initialization: {e}")
        logger.info("Predictor setup complete.")

    def predict(
        self,
        starts_with: str = Input(description="Comma-separated list of desired prefixes (e.g., 'SOL,VAN'). Leave empty if only using ends_with.", default=""),
        ends_with: str = Input(description="Desired suffix. Leave empty if only using starts_with.", default=""),
        count: int = Input(description="Number of vanity addresses to generate.", default=1, ge=1),
        batch_size: int = Input(description="Number of keys to generate on GPU per batch.", default=DEFAULT_CUDA_BATCH_SIZE, ge=1024),
        is_case_sensitive: bool = Input(description="Whether the prefix/suffix matching should be case-sensitive.", default=True)
        # Removed base, owner, target_type as they are not used by the Python core logic
    ) -> Dict[str, List[Dict[str, Any]]]: # Define return structure
        """
        Generate Solana vanity address(es) using CUDA acceleration (GPU key generation + CPU check).
        """

        # Input validation and parsing
        if not starts_with and not ends_with:
            logger.error("Either starts_with or ends_with must be provided.")
            return {"results": []}

        # Split comma-separated prefixes into a list
        prefix_list = [p.strip() for p in starts_with.split(',') if p.strip()] if starts_with else []
        suffix_list = [ends_with.strip()] if ends_with.strip() else [] # Pass as list

        logger.info(f"Generating {count} vanity address(es) with prefixes={prefix_list}, suffixes={suffix_list}, case_sensitive={is_case_sensitive}, batch_size={batch_size}...")

        start_time = time.time()
        try:
            # Call the core logic function with correct argument names
            results = find_vanity_addresses(
                prefixes=prefix_list,         # Correct name
                suffixes=suffix_list,         # Correct name
                count=count,
                batch_size=batch_size,
                case_sensitive=is_case_sensitive, # Correct name
            )

            elapsed = time.time() - start_time
            logger.info(f"Vanity address generation completed in {elapsed:.2f} seconds. Found {len(results)} results.")

            # Format the output for Cog, returning the base58 private key string
            formatted_results = []
            for res in results:
                address = res.get('address')
                secret_key_b58 = res.get('private_key') # Get the base58 string

                if address and secret_key_b58:
                    formatted_results.append({
                        "address": address,
                        "secret_key": secret_key_b58 # Return base58 string directly
                    })
                else:
                    logger.warning(f"Skipping result with missing data: {res}")

            return {"results": formatted_results}

        except ValueError as e:
            logger.error(f"Input validation error: {e}")
            return {"results": []} # Return empty on known errors
        except RuntimeError as e:
            logger.error(f"Runtime error during generation: {e}")
            return {"results": []} # Return empty on runtime errors (e.g., OpenCL issues)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            # In a production Cog model, you might want to return an error message
            # or structure, but for now, return empty results.
            return {"results": []} 