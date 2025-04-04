import secrets
# Removed numpy import as it's not used here

# Default batch size for CUDA kernel - adjust based on GPU memory/performance
DEFAULT_CUDA_BATCH_SIZE = 1 << 24 # Example: 16 million keys per batch

class CudaHostSetting:
    """Manages settings and state for CUDA-based key generation.
    NOTE: The global_offset feature is not currently utilized by CudaSearcher.
    """
    def __init__(self, batch_size: int = DEFAULT_CUDA_BATCH_SIZE):
        if not isinstance(batch_size, int) or batch_size <= 0:
             raise ValueError("Batch size must be a positive integer.")
        
        self.batch_size = batch_size
        # Start with a random 64-bit offset for the first batch
        # Using secrets.randbits for better randomness than just starting at 0
        self.global_offset = secrets.randbits(64)
        
        # Define typical CUDA block and grid dimensions
        # These can be tuned
        self.block_dim = (256,) # Threads per block
        # Calculate grid dimension based on batch size and block size
        self.grid_dim = ((self.batch_size + self.block_dim[0] - 1) // self.block_dim[0],)

    def advance_offset(self) -> None:
        """Advances the global offset for the next batch."""
        self.global_offset += self.batch_size
