
import subprocess
import sys
import os

if __name__ == "__main__":
    os.chdir("/home/harsh")
    start_directory = sys.argv[1]
    end_directory = sys.argv[2]
    start_files = list()
    for file in os.listdir(start_directory):
        if file.endswith(".txt") or file.endswith(".txt.gz"):
            x = os.path.join(os.getcwd(), start_directory, file)
            fn = os.path.basename(x)
            if fn.endswith(".txt.gz"):
                fn = fn[:-7]
            elif fn.endswith(".txt"):
                fn = fn[:-4]
            target = os.path.join(os.getcwd(), end_directory, fn)+".structure"
            if not os.path.exists(target+".hdf5"):
                subprocess.run(["python", "src/preprocessing/txtToHDF.py", x, target])