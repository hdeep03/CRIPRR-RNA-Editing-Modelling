#!/usr/bin/env/python

import sys
import numpy as np

def list_to_str(lst):
    ''' Given a list, return the string of that list with tab separators 
    '''
    return "\t".join(lst) 

def getCombinedRNAStructure(e_f, h_f, i_f, m_f):
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
        H_prob =  Hprofiles[i*2+1].split()
        I_prob =  Iprofiles[i*2+1].split()
        M_prob =  Mprofiles[i*2+1].split()
        E_prob =  Eprofiles[i*2+1].split()			
        P_prob = list(map((lambda a, b, c, d: 1-float(a)-float(b)-float(c)-float(d)), H_prob, I_prob, M_prob, E_prob))
        tmp = np.stack([P_prob, H_prob, I_prob, M_prob, E_prob], axis=-1)
        ret.append(tmp)
    fin = np.stack(ret)
    return fin
if __name__ == "__main__":
    getCombinedRNAStructure(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])