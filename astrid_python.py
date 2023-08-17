import six
import glob, os, struct
from os.path import isfile, expanduser
import numpy as np
import h5py
import pickle
from bigfile import BigFile

def get_subfind_chunk(snap):
    subdir = '/hildafs/datasets/Asterix/subfind/subfind_%d/*' % snap
    file_list = sorted(glob.glob(subdir))
    data = [(int(ff.split('/')[-1].split('.')[0][5:]), int(ff.split('/')[-1].split('.')[1])) for ff in file_list]
    data.sort(key=lambda x: x[0])  # Sort based on the first element of each tuple
    
    chunk_list = np.array([item[0] for item in data])
    maxgroup_list = np.array([item[1] for item in data])
    return chunk_list, maxgroup_list

def getSnapshot(snapNum):
    ''' Load all snapshot information for one snapshot. '''
    
    if snapNum > 294:
        path = '/hildafs/datasets/Asterix/PIG2/PIG_%03d' % snapNum
    else:
        path = '/hildafs/datasets/Asterix/PIG_files/PIG_%03d' % snapNum
    
    return BigFile(path)

def getAttributes(pig):
    ''' Load and print attributes for a given PIG. '''
    battr = pig["Header"].attrs
    scale_fac = battr["Time"][0]
    redshift = 1./battr["Time"][0] - 1
    Lbox = battr['BoxSize']
    hh = battr['HubbleParam']
    om0 = battr['Omega0']
    omb = battr['OmegaBaryon']
    oml = battr['OmegaLambda']
    Nfof = battr['NumFOFGroupsTotal']
    sigma8 = 0.82
    print('----------PIG File Info------------')

    print('Redshift = %.2f'%redshift)
    print('Lbox = %d ckpc/h'%Lbox)
    print('NfofGroups = %d'%Nfof)

    print('------Cosmological Parameters-----')
    print('h = %.4f'%hh)
    print('Omega_m = %.4f'%om0)
    print('Omega_b = %.4f'%omb)
    print('Omega_l = %.4f'%oml)
    
    return scale_fac, redshift, Lbox, hh, om0, omb, oml, Nfof, sigma8

def getProperties(pig, blockName):
    ''' Load all properties for one block within one snapshot. '''
    
    path = blockName + '/'
    
    return pig[path].keys()

def getValues(snapNum, idxs, propName, subgroup=False):
    ''' Load values of a specified property for a list of group/subgroup indices. '''
    def group_chunk_dir(groupidx,snap):
        sub_root = '/hildafs/datasets/Asterix/subfind/subfind_%03d/'%snap
        chunk = np.nonzero(maxgroup_list-1>=groupidx)[0][0]
        subdir = sub_root + 'chunk%d.%d/output/'%(chunk,maxgroup_list[chunk])
        tabfile = subdir + 'fof_subhalo_tab_%03d.hdf5'%snap
        return subdir
    
    chunk_list, maxgroup_list = get_subfind_chunk(snapNum)
    subdir_list = [group_chunk_dir(groupidx=idx, snap=snapNum) for idx in idxs]
    tabfiles = [subdir + 'fof_subhalo_tab_%03d.hdf5' % snapNum for subdir in subdir_list]

    result = {
        idx: np.array(h5py.File(tabfile, 'r')['Subhalo'][propName]) if subgroup
        else np.array(h5py.File(tabfile, 'r')['Group'][propName])
        for idx, tabfile in zip(idxs, tabfiles)
    }

    return result

def getParticleProperties(snapNum, idx, partType, propName):
    ''' Load values of a specified particle type and property for a given group/subgroup index. '''
    
    def group_chunk_dir(groupidx,snap):
        sub_root = '/hildafs/datasets/Asterix/subfind/subfind_%03d/'%snap
        chunk = np.nonzero(maxgroup_list-1>=groupidx)[0][0]
        # print('groupidx',groupidx,'chunk',chunk,maxgroup_list[chunk-1],maxgroup_list[chunk])

        subdir = sub_root + 'chunk%d.%d/output/'%(chunk,maxgroup_list[chunk])
        tabfile = subdir + 'fof_subhalo_tab_%03d.hdf5'%snap
        return subdir
    
    chunk_list, maxgroup_list = get_subfind_chunk(snapNum)
    
    subdir = group_chunk_dir(idx, snapNum)
    grpfile = subdir + 'snap_%03d.hdf5' % snapNum
    partType = 'PartType' + str(partType)
    
    with h5py.File(grpfile, 'r') as sbgrp:
        result = np.array(sbgrp[partType][propName])
    
    return result

def getValidIndices(pig, propName, minVal, maxVal):
    ''' Return an array of group/subgroup indices which satisfy givne property conditions. '''
    
    path = 'FOFGroups/' + propName
    fofs = pig.open(path)[:]
    mask = (fofs >= minVal) & (fofs <= maxVal)
    result = np.where(mask)[0]
    
    return result

def getBHIndices(snapNum, bhid):
    ''' Return the index, group/subgroup indices for a single BH. '''
    
    pig = getSnapshot(snapNum)
    sfile = BigFile('/hildafs/datasets/Asterix/subfind/subfind-idx/subfind_%03d' % snapNum)
    BHIDs = pig.open('5/ID')[:]
    bhidx = (BHIDs==bhid).nonzero()[0][0]
    groupidx = pig.open('5/GroupID')[bhidx]-1
    subhidx = sfile.open('5/Subfind-SubGrpIndex')[bhidx]
    
    return bhidx, groupidx, subhidx

def getBHHistory(bhid):
    ''' Load the entire history of a given black hole. '''
  
    path = '/hildafs/datasets/Asterix/BH_details_dict/Read-Blackhole-Detail'
    detail = BigFile(path)
    AllIDs = detail.open('BHID')[:]
    Index = detail.open('Index')[:]
    
    idx = np.where(AllIDs == bhid)[0][0]
    chunk = Index[idx]
    
    outdir = '/hildafs/datasets/Asterix/BH_details_dict/'
    save = outdir + f'BlackholeDetails-{chunk:04d}'
    
    with open(save, 'rb') as f:
        data = pickle.load(f)
    
    bh = data[bhid]
    
    return bh

def getSubfindOBT(groupidx,snapNum):
    ''' Load ObjectByType for a given group and snapshot. '''
  
    def group_chunk_dir(groupidx,snap):
        sub_root = '/hildafs/datasets/Asterix/subfind/subfind_%03d/'%snap
        chunk = np.nonzero(maxgroup_list-1>=groupidx)[0][0]
        subdir = sub_root + 'chunk%d.%d/output/'%(chunk,maxgroup_list[chunk])
        tabfile = subdir + 'fof_subhalo_tab_%03d.hdf5'%snap
        return subdir
    
    chunk_list, maxgroup_list = get_subfind_chunk(snapNum)
    
    subdir = group_chunk_dir(groupidx,snapNum)
    tabfile = subdir + 'fof_subhalo_tab_%03d.hdf5'%snapNum
    
    with h5py.File(tabfile,'r') as sbgrp:
        zeros = np.array([[0,0,0,0,0,0]],dtype=np.uint64)

        glbt = sbgrp['Group']['GroupLenType'][:]
        gobt = np.concatenate([zeros,np.cumsum(glbt,axis=0)],axis=0).astype(int)


        first_sub = sbgrp['Group']['GroupFirstSub'][:]
        nsub = sbgrp['Group']['GroupNsubs'][:]

        # print('total subhalos in this chunk:',sum(nsub))


        slbt = sbgrp['Subhalo']['SubhaloLenType'][:]
        sobt = np.zeros_like(slbt)
        sobt = np.concatenate([zeros,sobt],axis=0).astype(int)
        for i,f in enumerate(first_sub):
            if f < 0:  # skip groups with no subhalo
                continue
            # align the first subgroup with group starting point
            sobt[f] = gobt[i]
            # assign the rest of the subgroup starting idx
            sobt[f+1:f+nsub[i]] = sobt[f] + np.cumsum(slbt[f:f+nsub[i]-1],axis=0)
            
        return gobt,sobt,first_sub

def placeBH(groupidx,bhid,snapNum,gobt,sobt,first_sub):
    ''' Load the subhalo indices for a given black hole. '''
    def group_chunk_dir(groupidx,snap):
        sub_root = '/hildafs/datasets/Asterix/subfind/subfind_%03d/'%snap
        chunk = np.nonzero(maxgroup_list-1>=groupidx)[0][0]
        subdir = sub_root + 'chunk%d.%d/output/'%(chunk,maxgroup_list[chunk])
        tabfile = subdir + 'fof_subhalo_tab_%03d.hdf5'%snap
        return subdir
    
    chunk_list, maxgroup_list = get_subfind_chunk(snapNum)
    
    subdir = group_chunk_dir(groupidx,snapNum)
    if snapNum < 294:
        grpfile = subdir + 'snap_%03d.hdf5'%snapNum
    else:
        grpfile = subdir + 'snap-groupordered_%03d.hdf5'%snapNum
    
    with h5py.File(grpfile,'r') as sbgrp:
        id5 = sbgrp['PartType5']['ParticleIDs'][:]
        bidx = (id5==bhid).nonzero()[0][0]
        
    gidx = (gobt[:,5]>bidx).nonzero()[0][0]-1
    sidx = (sobt[:,5]>bidx).nonzero()[0][0]-1

    # make sure that we get the correct BH group
    bh_group = id5[gobt[gidx][5]:gobt[gidx+1][5]]
    bh_sbgrp = id5[sobt[sidx][5]:sobt[sidx+1][5]]
    assert(bhid in bh_group)
    assert(bhid in bh_sbgrp)
    
    sbegin = first_sub[gidx]
    send = first_sub[gidx+1]

    return bidx,gidx,sidx,sbegin,send

def getSubhaloSummary(groupidx,snapNum,gidx,sidx,sbegin,send,feature_list,main_sub=False):
    ''' Load the subhalo summary for a given black hole. '''
    def group_chunk_dir(groupidx,snap):
        sub_root = '/hildafs/datasets/Asterix/subfind/subfind_%03d/'%snap
        chunk = np.nonzero(maxgroup_list-1>=groupidx)[0][0]
        subdir = sub_root + 'chunk%d.%d/output/'%(chunk,maxgroup_list[chunk])
        tabfile = subdir + 'fof_subhalo_tab_%03d.hdf5'%snap
        return subdir
    
    # Recent examples are not using group_chunk_dir; see notebook for faster implementation.
    
    chunk_list, maxgroup_list = get_subfind_chunk(snapNum)

    subdir = group_chunk_dir(groupidx,snapNum)
    tabfile = subdir + 'fof_subhalo_tab_%03d.hdf5'%snapNum
    
    if main_sub:
        print('output the main subhalo in group')
        sidx = sbegin

    
    output = {}
    with h5py.File(tabfile,'r') as sbgrp:
        if feature_list:
            for ff in feature_list:
                try: 
                    output[ff] = sbgrp['Subhalo'][ff][sidx]
                except KeyError:
                    print('skipping %s: feature does not exist!'%ff)
        else:
            for ff in sbgrp['Subhalo'].keys():
                output[ff] = sbgrp['Subhalo'][ff][sidx]
                
    return output

def getSubhalo(bhid, snap):
    ''' Load all subhalo information for a given black hole. '''
    bhidx, groupidx, subhidx = getBHIndices(snap, bhid)
    gobt, sobt, first_sub = getSubfindOBT(groupidx, snap)
    bidx, gidx, sidx, sbegin, send = placeBH(groupidx, bhid, snap, gobt, sobt, first_sub)
    subhalo_summary = getSubhaloSummary(groupidx, snap, gidx, sidx, sbegin, send, feature_list=[],main_sub=False)
    
    return subhalo_summary

pig_files = np.array([( 13., 12.00052002), ( 15., 11.14730568), ( 17., 10.00000011),( 18.,  9.51937328), ( 20.,  9.        ), ( 21.,  8.74003771),
       ( 22.,  8.30125937), ( 23.,  8.0000009 ), ( 25.,  7.85215935),
       ( 27.,  7.57949693), ( 29.,  7.22391986), ( 31.,  7.        ),
       ( 33.,  6.82728484), ( 35.,  6.7341494 ), ( 36.,  6.63963261),
       ( 39.,  6.50001875), ( 41.,  6.37395074), ( 43.,  6.25000181),
       ( 47.,  6.000007  ), ( 49.,  5.88951234), ( 51.,  5.80392007),
       ( 55.,  5.68661939), ( 58.,  5.60043255), ( 62.,  5.5000065 ),
       ( 66.,  5.40706607), ( 73.,  5.2889018 ), ( 84.,  5.1959187 ),
       (107.,  4.9999988 ), (113.,  4.90696449), (118.,  4.80103969),
       (122.,  4.70227684), (125.,  4.60801268), (126.,  4.56050098),
       (128.,  4.5000055 ), (130.,  4.4522403 ), (131.,  4.41275301),
       (133.,  4.33075808), (134.,  4.28518299), (136.,  4.25000525),
       (137.,  4.21808346), (139.,  4.16268666), (142.,  4.10562721),
       (144.,  4.05660173), (147.,  4.        ), (149.,  3.95214443),
       (151.,  3.90008125), (153.,  3.84395322), (155.,  3.80781523),
       (158.,  3.75000713), (161.,  3.69999994), (163.,  3.65787776),
       (166.,  3.59662107), (170.,  3.54071105), (174.,  3.5000045 ),
       (177.,  3.45601206), (181.,  3.39524235), (184.,  3.35324595),
       (187.,  3.2980892 ), (190.,  3.25000213), (194.,  3.19999997),
       (198.,  3.14888379), (202.,  3.10024067), (207.,  3.04872081),
       (214.,  3.        ), (226.,  2.89741198), (240.,  2.80000007),
       (252.,  2.7000037 ), (261.,  2.59999712), (272.,  2.49999995),
       (282.,  2.3780361 ), (294.,  2.3000033 ), (299., 2.26944371),
        (301., 2.25776553), (303., 2.25000325), (311., 2.2       ), (319., 2.15000315), (329., 2.09999659),
       (334., 2.06989544), (336., 2.05661653), (338., 2.04999863),
       (340., 2.03899071), (343., 2.02489603), (345., 2.01242231),
       (348., 2.00000003), (350., 1.98742113), (352., 1.97489497),
       (354., 1.96748243), (357., 1.95      ), (367., 1.9       ),
       (376., 1.85      ), (385., 1.8       ), (395., 1.75      ),
       (403., 1.7       ), (412., 1.65      ), (421., 1.6       ),
       (427., 1.56746757), (437., 1.5143181 ), (440., 1.5       ),
       (444., 1.48269844), (453., 1.43897421), (464., 1.38897236),
       (470., 1.36165484), (473., 1.35000003), (478., 1.32623316),
       (481., 1.31062135), (483., 1.29999999)],
      dtype=[('snap', '<f8'), ('redshift', '<f8')])

def querySubhalos(bhid, zi, zf):
    ''' Load subhalo information for a given black hole across a period of time. '''
    snaps = pig_files[(pig_files['redshift'] <= zi) & (pig_files['redshift'] >= zf)]
    
    result = {}
    
    for curr in snaps:
        snapNum = curr['snap']
        z = curr['redshift']
        print(z)
        result[z] = getSubhalo(bhid, snapNum)
        
    return result
