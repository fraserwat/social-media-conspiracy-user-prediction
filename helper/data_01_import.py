"""
Importing into 01_exploratory_data_analysis AND 02_ because I want to be able to download the data
in order to replicate it.
"""

import gzip
import os
import requests


def download_and_unzip(output_dir="data") -> None:
    """
    Helper function to download the data.
    Does not return anything, but puts everything (by default) into a `data` subfolder.
    """

    download_urls = [
        "https://cs230-reddit.s3.us-west-1.amazonaws.com/non-q-posts-v2.csv.gz",
        "https://cs230-reddit.s3.us-west-1.amazonaws.com/q-posts-v2.csv.gz",
        "https://cs230-reddit.s3.us-west-1.amazonaws.com/bert.csv.gz",
    ]

    # Ensure the data folder exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for url in download_urls:
        # Define file names based on the URL
        file_name = os.path.join(output_dir, url.split("/")[-1])
        decompressed_file_name = os.path.join(
            output_dir, url.split("/")[-1].rsplit(".", 1)[0]
        )

        # Download the .gz file
        response = requests.get(url, stream=True, timeout=1e6)
        with open(file_name, "wb") as gz_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    gz_file.write(chunk)

        # Decompress the .gz file
        with gzip.open(file_name, "rb") as f_in:
            with open(decompressed_file_name, "wb") as f_out:
                f_out.write(f_in.read())

        # Delete the original .gz file
        os.remove(file_name)
