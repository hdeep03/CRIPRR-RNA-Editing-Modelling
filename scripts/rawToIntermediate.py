
import subprocess
import sys
import os

if __name__ == "__main__":
    os.chdir("/home/harsh")
    start_directory = sys.argv[1]
    end_directory = sys.argv[2]
    start_files = list()
    for file in os.listdir(start_directory):
        if file.endswith(".fasta") or file.endswith(".fasta.gz"):
            x = os.path.join(os.getcwd(), start_directory, file)
            fn = os.path.basename(x)
            if fn.endswith(".fasta.gz"):
                fn = fn[:-9]
            elif fn.endswith(".fasta"):
                fn = fn[:-6]
            target = os.path.join(os.getcwd(), end_directory, fn)
            if not os.path.exists(target+".hdf5"):
                subprocess.run(["python", "src/preprocessing/fastaToHDF.py", x, target])