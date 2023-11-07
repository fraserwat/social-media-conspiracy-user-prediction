import os
import logging
from typing import List
import requests


def download(output_dir="data") -> List[str]:
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

    filepaths = []

    for url in download_urls:
        logging.debug("Downloading from %s...", url)
        # Define file names based on the URL
        file_name = os.path.join(output_dir, url.split("/")[-1])

        # Download the .gz file
        response = requests.get(url, stream=True, timeout=1e6)
        with open(file_name, "wb") as gz_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    gz_file.write(chunk)
        logging.debug("Finished downloading from %s.", url)
        filepaths.append(file_name)

    return filepaths
