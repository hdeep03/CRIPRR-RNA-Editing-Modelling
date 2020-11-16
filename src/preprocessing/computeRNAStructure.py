import os
import h5py
import numpy as np
import subprocess
from random import randint
from multiprocessing.pool import ThreadPool
import sys

def writeFasta(sequences, location):
    i = 1
    with open(location, "w") as f:
        for seq in sequences:
            f.write('>seq {}\n'.format(i))
            i+=1
            f.write(seq+'\n')
            
def runCommand(command):
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
def list_to_str(lst):
    ''' Given a list, return the string of that list with tab separators 
    '''
    return "\t".join(lst) 

def gCRS(e_f, h_f, i_f, m_f):
    fEprofile = open(e_f)
    Eprofiles = fEprofile.readlines()

    fHprofile = open(h_f)
    Hprofiles = fHprofile.readlines()

    fIprofile = open(i_f)
    Iprofiles = fIprofile.readlines()

    fMprofile = open(m_f)
    Mprofiles = fMprofile.readlines()
    
    ret = list()
    for i in range(0, len(Eprofiles)//2):
        id = Eprofiles[i*2].split()
        H_prob =  [float(x) for x in Hprofiles[i*2+1].split()]
        I_prob =  [float(x) for x in Iprofiles[i*2+1].split()]
        M_prob =  [float(x) for x in Mprofiles[i*2+1].split()]
        E_prob =  [float(x) for x in Eprofiles[i*2+1].split()]
        P_prob = list(map((lambda a, b, c, d: 1-float(a)-float(b)-float(c)-float(d)), H_prob, I_prob, M_prob, E_prob))
        tmp = np.stack([P_prob, H_prob, I_prob, M_prob, E_prob], axis=-1)
        ret.append(tmp)
    fin = None
    if len(ret)>1:
        fin = np.stack(ret)
    else:
        fin = np.array(ret)
    return fin 
def getBatchStructureData(sequences):
    hsh = str(randint(0, 9999999999999999999))
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'tmp/rna_structure_temp-{}.fasta'.format(hsh))
    writeFasta(sequences, filename)
    
    pool = ThreadPool(processes=4)
    commands = list()
    for x in ["E", "M", "H", "I"]:
        exe_path = os.path.join(dirname, '{}_RNAplfold'.format(x))
        out_path = os.path.join(dirname, 'tmp/{}_profile-{}.txt'.format(x, hsh))
        command = "{} -W 240 -L 160 -u 1 <{} >{}".format(exe_path, filename, out_path)
        commands.append(command)
    pool.map(runCommand, commands)
    ret = gCRS(os.path.join(dirname, 'tmp/E_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/H_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/I_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/M_profile-{}.txt'.format(hsh)))
    subprocess.run("rm {}".format(filename), shell=True)
    subprocess.run("rm {} {} {} {}".format(os.path.join(dirname, 'tmp/E_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/H_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/I_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/M_profile-{}.txt'.format(hsh))), shell=True)
    return ret
def getStructureData(sequence):
    return getBatchStructureData([sequence])

from tqdm import tqdm
from multiprocessing import Pool
def main(filepath, tgt):
    vals = list()
    i = 0
    with h5py.File(filepath, "r") as hdf:
        pbar = tqdm(total=hdf['sequences'].shape[0])
        pbar.update(i)
        while i<hdf['sequences'].shape[0]:
            btc = 0
            batch = list()
            while btc < 4096 and i < hdf['sequences'].shape[0]:
                pbar.update(1)
                batch.append(hdf['sequences'][i][450:551].decode("utf-8").strip())
                btc+=1
                i+=1
            vals.append(getBatchStructureData(batch))
        pbar.close()
    fin = np.concatenate(vals, axis=0)
    with h5py.File(tgt, "w") as hdf:
        print(fin.shape)
        hdf.create_dataset('RNAdata', compression="gzip", data=fin)
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])