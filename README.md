# SolVanityCL (Cog Version)

GPU-accelerated Solana vanity address generator, adapted for use with [Cog](https://github.com/replicate/cog) and [Replicate](https://replicate.com/).

This version uses Python with PyOpenCL for GPU acceleration, leveraging the core logic from the original SolVanityCL project.

## Prerequisites

*   **Docker**: Cog uses Docker to build and run the model environment. [Install Docker](https://docs.docker.com/get-docker/).
*   **Cog**: Install the Cog CLI. [Installation instructions](https://github.com/replicate/cog#install).
*   **NVIDIA GPU & Drivers**: An NVIDIA GPU with appropriate drivers installed is required for prediction. OpenCL libraries compatible with your driver are also needed (these are included in the Cog build environment defined in `cog.yaml`).

## Getting Started (Local)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Build the Cog image:**
    ```bash
    cog build -t sol-vanity-cog
    ```
    This command builds the Docker image based on `cog.yaml`, installing Python, PyOpenCL, necessary system libraries (like OpenCL headers/dev libraries), and Python dependencies from `requirements.txt`.

3.  **Run Predictions Locally:**
    ```bash
    cog predict -i starts_with="MYPREFIX" -i ends_with="suffix" -i count=2
    ```
    Replace the input parameters (`-i`) with your desired values:
    *   `starts_with`: Comma-separated list of desired prefixes (e.g., "SOL,VAN"). Case-sensitive matching depends on `is_case_sensitive`.
    *   `ends_with`: Desired suffix. Case-sensitive matching depends on `is_case_sensitive`.
    *   `count`: Number of addresses to generate (default: 1).
    *   `iteration_bits`: OpenCL iteration bits (default: 24). Higher values might be faster per iteration but require more setup time.
    *   `is_case_sensitive`: `true` or `false` (default: true) for prefix/suffix matching.

    You must provide at least one of `starts_with` or `ends_with`.

    The output will be a JSON object containing a list of results:
    ```json
    {
      "results": [
        {
          "address": "MYPREFIX...somehash...suffix",
          "secret_key": [ ... list of 64 integers ... ]
        },
        {
          "address": "MYPREFIX...anotherhash...suffix",
          "secret_key": [ ... list of 64 integers ... ]
        }
      ]
    }
    ```

## Deployment to Replicate

1.  **Log in to Replicate:**
    ```bash
    cog login
    ```

2.  **Push the model:**
    ```bash
    cog push r8.im/your-username/your-model-name
    ```
    Replace `your-username` with your Replicate username and `your-model-name` with the desired name for your model.

    Follow the URL provided by the `cog push` command to see your model on Replicate and run predictions via the API or web interface.

## (Optional) Local CLI Usage

While the primary interface is now through Cog, the original CLI script (`core/cli.py`) has been retained for potential local testing or utility use. It uses the same underlying generator logic.

1.  **Ensure Dependencies are Installed:**
    You might need to manually install dependencies if not using the Cog environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```
    *Note: Installing PyOpenCL and its system dependencies (OpenCL SDK/drivers) outside Docker can be complex.* 

2.  **Run the CLI:**
    ```bash
    python -m core.cli --help
    python -m core.cli search-pubkey --starts-with SOL --count 1 --output-dir ./found_keys
    python -m core.cli show-device # List available OpenCL devices
    ```

## FAQs

See [FAQs.md](./FAQs.md) (Note: Some FAQs might refer to the original Rust implementation and may not be fully applicable).

## Donations

If you find this project helpful, please consider making a donation:

SOLANA: `PRM3ZUA5N2PRLKVBCL3SR3JS934M9TZKUZ7XTLUS223`

EVM: `0x8108003004784434355758338583453734488488`
