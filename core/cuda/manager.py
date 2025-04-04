import cupy
import os
import subprocess # Keep for potential future use, but not for compilation now
from loguru import logger

# Define the path to the CUDA source file and the library
CUDA_SOURCE_DIR = os.path.join(os.path.dirname(__file__))
KERNEL_SOURCE_FILE = os.path.join(CUDA_SOURCE_DIR, "vanity_generator.cu")
CUDA_ECC_LIB_DIR = os.path.join(CUDA_SOURCE_DIR, "cuda-ecc-ed25519")
PTX_FILE = os.path.join(CUDA_SOURCE_DIR, "kernel.ptx") # Path to pre-compiled PTX

# Remove the list of library .cu source files as they don't exist/aren't needed for compilation
# CUDA_LIB_SOURCES = [
#     os.path.join(CUDA_ECC_LIB_DIR, "ed25519.cu"),
#     os.path.join(CUDA_ECC_LIB_DIR, "sha512.cu"),
#     os.path.join(CUDA_ECC_LIB_DIR, "ge.cu"),
#     os.path.join(CUDA_ECC_LIB_DIR, "fe.cu"),
#     os.path.join(CUDA_ECC_LIB_DIR, "sc.cu"),
#     os.path.join(CUDA_ECC_LIB_DIR, "keypair.cu"),
# ]

# Only compile the main kernel source file
ALL_SOURCES = [KERNEL_SOURCE_FILE]
ALL_SOURCES_STR = " ".join(ALL_SOURCES)

# Kernel names (must match those in the PTX)
KERNEL_NAME = "find_vanity_address_kernel"
INIT_KERNEL_NAME = "init_curand_states"

class CudaManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CudaManager, cls).__new__(cls)
            cls._instance.kernel = None
            cls._instance.init_kernel = None
            # Load kernel on first instantiation
            cls._instance.load_kernel()
        return cls._instance

    def load_kernel(self):
        """Loads the pre-compiled CUDA kernel from the PTX file."""
        if not os.path.exists(PTX_FILE):
            logger.error(f"Pre-compiled PTX file not found: {PTX_FILE}")
            logger.error("This file should have been generated during the 'cog build' process.")
            raise FileNotFoundError(f"PTX file not found: {PTX_FILE}. Ensure build step in cog.yaml ran successfully.")

        try:
            logger.info(f"Loading pre-compiled PTX from: {PTX_FILE}")
            # Read the pre-compiled PTX code
            with open(PTX_FILE, "r") as f:
                ptx_code = f.read()

            # Load the main kernel
            self.kernel = cupy.RawKernel(ptx_code, KERNEL_NAME)
            logger.info(f"Loaded CUDA kernel: {KERNEL_NAME}")

            # Load the init kernel
            self.init_kernel = cupy.RawKernel(ptx_code, INIT_KERNEL_NAME)
            logger.info(f"Loaded CUDA kernel: {INIT_KERNEL_NAME}")

        # Removed subprocess.CalledProcessError handling as compilation is done at build time
        except cupy.CuPyError as e:
            logger.error(f"Failed to load CUDA kernel from PTX with CuPy: {e}")
            raise RuntimeError("Failed to load CUDA kernel from PTX with CuPy.") from e
        except FileNotFoundError as e:
            # Catch potential race condition or issue reading the file
            logger.error(f"Error reading PTX file {PTX_FILE}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during kernel loading from PTX: {e}")
            raise

    def get_kernel(self):
        if self.kernel is None:
             # Should have been loaded during __new__, but reload as fallback
             logger.warning("Kernel was None, attempting to reload...")
             self.load_kernel()
        return self.kernel

    def get_init_kernel(self):
        if self.init_kernel is None:
            logger.warning("Init kernel was None, attempting to reload...")
            self.load_kernel() # Should load both kernels
        return self.init_kernel

# Singleton instance creation will now trigger load_kernel
# which expects the PTX file to exist.
cuda_manager = CudaManager() 