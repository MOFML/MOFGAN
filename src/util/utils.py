import multiprocessing
from io import BytesIO
from pathlib import Path
from typing import Union

import pysftp as pysftp

import config


class SFTPConnection:

    def __init__(self, sftp: pysftp.Connection):
        self.sftp = sftp
        self.root = Path(config.sftp.root)

    def __enter__(self):
        self.sftp.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sftp.__exit__(exc_type, exc_val, exc_tb)

    def download_bytes(self, path: Union[Path, str]) -> bytes:
        relative_path_string = path.as_posix() if isinstance(path, Path) else str(path)
        path_string = (self.root / relative_path_string).as_posix()

        print(f"Downloading {path_string}")

        flo = BytesIO()
        self.sftp.getfo(path_string, flo)
        flo.seek(0)
        return flo.read()

    def download_string(self, path: Union[Path, str]) -> str:
        return self.download_bytes(path).decode('utf-8')


def sftp_connection() -> SFTPConnection:
    print("Connecting to remote server")
    return SFTPConnection(pysftp.Connection(config.sftp.host,
                                            username=config.sftp.user,
                                            private_key=config.sftp.key,
                                            cnopts=pysftp.CnOpts()))


def get_available_threads():
    return {
        8:  4,
        16: 8,
        40: 32,
    }.get(multiprocessing.cpu_count(), 1)
