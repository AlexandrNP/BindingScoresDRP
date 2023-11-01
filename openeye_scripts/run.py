import os
import subprocess
from multiprocessing import Pool, current_process

PDB_DIR = "PDB/"
POOL_SIZE = 8


def dock(pdb_name):
    pdb = os.path.join(PDB_DIR, pdb_name)
    pid = current_process()._identity[0]
    subprocess.run(["./dock.sh", "--pdb", f"{pdb}", "--tmp", f"tmp-{pid}"])


if __name__ == "__main__":
    process_pool = Pool(POOL_SIZE)
    pdbs = os.listdir(PDB_DIR)
    process_pool.map(dock, pdbs)
