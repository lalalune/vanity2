import logging
import multiprocessing
import sys
from typing import List, Tuple

import click
import cupy # For device count in show_device (optional)

from core.config import DEFAULT_CUDA_BATCH_SIZE # Use CUDA default
from core.generator import find_vanity_addresses

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option(
    "--starts-with",
    type=str,
    default=[],
    help="Public key starts with the indicated prefix. Provide multiple arguments to search for multiple prefixes.",
    multiple=True,
)
@click.option(
    "--ends-with",
    type=str,
    default="",
    help="Public key ends with the indicated suffix.",
)
@click.option("--count", type=int, default=1, help="Count of pubkeys to generate.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="./",
    help="Output directory.",
)
@click.option(
    "--batch-size", # Changed from iteration-bits
    type=int,
    default=DEFAULT_CUDA_BATCH_SIZE, # Use CUDA default
    help="Keys generated per GPU per batch.",
)
@click.option(
    "--is-case-sensitive", type=bool, default=True, help="Case sensitive search flag."
)
def search_pubkey(
    starts_with: Tuple[str, ...],
    ends_with: str,
    count: int,
    output_dir: str,
    batch_size: int, # Changed parameter
    is_case_sensitive: bool,
):
    """Search for Solana vanity pubkeys (CUDA) and save them to files."""
    try:
        results = find_vanity_addresses(
            prefixes=list(starts_with),
            suffixes=[ends_with] if ends_with else [],
            count=count,
            batch_size=batch_size,
            case_sensitive=is_case_sensitive,
        )

        if results:
            logging.info(f"Found {len(results)} result(s). Saving...")
            save_results_to_files(results, output_dir)
            logging.info(f"Results saved to {output_dir}")
        else:
            logging.warning("No vanity addresses found matching the criteria.")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Runtime Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        click.echo("An unexpected error occurred. Check logs for details.", err=True)
        sys.exit(1)


def save_results_to_files(results: List[dict], output_dir: str):
    """Saves the found addresses and keys to JSON files."""
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    for i, result in enumerate(results):
        address = result.get('address')
        secret_key = result.get('secret_key')
        if address and secret_key:
            filename = os.path.join(output_dir, f"{address}.json")
            data_to_save = {
                "address": address,
                "secret_key": secret_key
            }
            try:
                with open(filename, "w") as f:
                    json.dump(data_to_save, f, indent=4)
            except IOError as e:
                logging.error(f"Failed to save file {filename}: {e}")
        else:
            logging.warning(f"Skipping result {i+1} due to missing data: {result}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    cli()
