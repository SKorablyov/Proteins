import os
import numpy as np
from io import BytesIO
from tqdm import tqdm
import bcolz

pn_path = os.curdir + '/data/proteinnet/casp11/'
data_path = os.curdir + '/data/'
name = 'validation'
ids = []
seqs = []
evs = []
coords = []
masks = ['init', '/n']
id_next, pri_next, ev_next, ter_next, msk_next = False, False, False, False, False
with open(pn_path + name) as fp:
    for line in tqdm(iter(fp.readline, '')):
        if id_next:
            ids.append(line[:-1])
        elif pri_next:
            seqs.append(line[:-1])
        elif ev_next:
            #evs.append(line[:-1])
            #print(BytesIO(bytes(line, 'utf-8')))
            evs.append(np.genfromtxt(BytesIO(bytes(line, 'utf-8'))))
            #print(np.genfromtxt(BytesIO(bytes(line, 'utf-8')))[1])#.shape())
            #print(line[:-1])
            #coords.append(line[1:-1])

        elif ter_next:
            coords.append(np.genfromtxt(BytesIO(bytes(line, 'utf-8'))))
            #coords.append(line[1:-1])
        elif msk_next:
            masks.append(line[:-1])

        if np.core.defchararray.find(line, "[ID]", end=5) != -1:
            id_next = True
            masks.pop()
            masks.pop()
            pri_next, ev_next, ter_next, msk_next = False, False, False, False
        elif np.core.defchararray.find(line, "[PRIMARY]", end=10) != -1:
            pri_next = True
            ids.pop()
            id_next, ev_next, ter_next, msk_next = False, False, False, False
        elif np.core.defchararray.find(line, "[EVOLUTIONARY]", end=15) != -1:
            ev_next = True
            seqs.pop()
            id_next, pri_next, ter_next, msk_next = False, False, False, False
        elif np.core.defchararray.find(line, "[TERTIARY]", end=11) != -1:
            ter_next = True
            evs.pop()
            id_next, pri_next, ev_next, msk_next = False, False, False, False
        elif np.core.defchararray.find(line, "[MASK]", end=7) != -1:
            msk_next = True
            coords.pop()
            id_next, pri_next, ev_next, ter_next = False, False, False, False

print('# IDs: {}'.format(len(ids)))
print('# Seqs: {}'.format(len(seqs)))
print('# PSSMs: {}'.format(len(evs)))
print('# Coords: {}'.format(len(coords)))
print('# Masks: {}'.format(len(masks[:-1]))) #-1 because of blank line at end of file

pssm = evs
xyz = coords

# loop through each evolutionary section
for i in range(len(ids)):
    # first store the id and sequence
    id = ids[i]
    seq = seqs[i]

    # next get the PSSM matrix for the sequence
    sp = 21 * i
    ep = 21 * (i + 1)
    psi = np.array(pssm[sp:ep])
    pssmi = np.stack([p for p in psi], axis=1)

    # then get the coordinates
    sx = 3 * i
    ex = 3 * (i + 1)
    xi = np.array(xyz[sx:ex])
    xyzi = np.stack([c for c in xi], axis=1) / 100  # have to scale by 100 to match PDB

    # lastly convert the mask to indices
    msk_idx = np.where(np.array(list(masks[i])) == '+')[0]

    # bracket id or get "setting an array element with a sequence"
    zt = np.array([[id], seq, pssmi, xyzi, msk_idx])

    if i == 0:
        bc = bcolz.carray([zt], rootdir=data_path + name+'.bc', mode='w', expectedlen=len(ids))
        bc.flush()
    else:
        bc = bcolz.carray(rootdir=data_path + name+'.bc', mode='w')
        bc.append([zt])
        bc.flush()