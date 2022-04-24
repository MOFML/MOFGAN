import multiprocessing
import os
import subprocess
import tarfile
import time
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

from util import utils

core_file_name: str = "2019-07-01-ASR-public_12020.tar"
core_download_url: str = f"https://zenodo.org/record/3370144/files/{core_file_name}?download=1"
core_file_size: int = 263168000  # Bytes
core_files_count: int = 11660

# core_download_csv_url = "https://zenodo.org/record/3370144/files/2019-07-01-ASR-public_12020.csv?download=1"


input_path = Path("_data/inputs")
output_path = Path("_data/outputs")


def download_core():
    def download_file(file_url, output_file_path):
        response = requests.get(file_url, stream=True)
        total_bytes = int(response.headers.get('Content-Length'))
        progress_bar = tqdm(total=total_bytes, unit='iB', unit_scale=True, ncols=80)
        with open(output_file_path, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):  # 1 KiB
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if progress_bar.n != total_bytes:
            print("Downloaded file size mismatch!")
            exit(1)

    download_path = Path('_data')
    download_path.mkdir(exist_ok=True)

    local_file = download_path / core_file_name
    if not local_file.exists():
        print("Downloading CoRE dataset...")
        download_file(core_download_url, str(local_file))

    if os.path.getsize(local_file) != core_file_size:
        print("CoRE dataset size mismatch!")
        exit(1)

    if not (download_path / 'structure_11660').exists():
        print("Extracting CoRE dataset...")
        with tarfile.open(local_file) as core_tar:
            core_tar.extractall(download_path)


def run(cmd: List[str], debug=False):
    if debug:
        print(f"Executing: {' '.join(cmd)}")

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # signal.alarm(1000)

    (stdout, stderr) = (x.decode('utf-8').strip() for x in p.communicate())
    if p.returncode != 0:
        print(f"ERROR CODE: {p.returncode}")
    if stderr:
        print(stderr)
        exit(1)

    if stdout:
        print(stdout)


def generate_inputs():
    if input_path.exists():
        assert len(os.listdir(input_path)) == core_files_count
        return

    input_path.mkdir()

    cif_folder = Path("_data/structure_11660")
    cif_files = list(cif_folder.iterdir())

    if len(cif_files) != core_files_count:
        print("Extracted file count mismatch!")
        exit(1)

    for p in cif_files:
        print(f"Processing: {p}")
        name = p.name[:p.name.rindex('.')]
        run(['./bin/cif2input', str(p), 'src/dataset/MOFGAN/data_ff_UFF', str(input_path / f"{name}.input")])


def generate_energy_grids():
    if output_path.exists():
        assert len(os.listdir(output_path)) == core_files_count
        return

    output_path.mkdir()

    input_files = list(input_path.iterdir())

    if len(input_files) != core_files_count:
        print("Input file count mismatch!")
        exit(1)

    process_count = utils.get_available_threads()

    print(f"Generating energy grids with {process_count} threads...")
    start = time.time()
    with multiprocessing.Pool(process_count) as pool:
        for i, p in enumerate(pool.imap_unordered(generate_energy_grid, input_files)):
            print(f"Processed: {p}  [avg time: {round((time.time() - start) / (i + 1), 2)}s]")


def generate_energy_grid(path: Path):
    name = path.name[:path.name.rindex('.')]
    run(['./bin/Vext_cpu', str(path), str(output_path / f"{name}.output")])
    return path


def main():
    # Note: First clone https://github.com/MusenZhou/MOFGAN into this directory and compile
    download_core()

    sub_repo = Path('src/dataset/MOFGAN')
    if not sub_repo.exists():
        print(f"Missing {sub_repo} subrepo")
        exit(1)

    generate_inputs()
    generate_energy_grids()

    print("DONE!")


if __name__ == '__main__':
    main()
