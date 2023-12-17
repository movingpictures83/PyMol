from collections import defaultdict

from dataset import PISToN_dataset
import os
import numpy as np

import pickle

import PyPluMA
import PyIO
import torch
import cv2

def get_significant_pixels(ppi, grid_dir, spatial_attn, stat_spatial_attn_dict):
    all_significant_pixels = []
    resnames = f"{grid_dir}/{ppi}_resnames.npy"
    resnames = np.load(resnames, allow_pickle=True)
    for feature_i in range(len(stat_spatial_attn_dict.keys())):
        feature = list(stat_spatial_attn_dict.keys())[feature_i]
        att_mat = spatial_attn[feature_i]
        final_attn = get_spatial_attn(att_mat)
        z_scores = (final_attn-stat_spatial_attn_dict[feature]['mean'])/stat_spatial_attn_dict[feature]['std']
        mask = z_scores>1.96#2.58
        significant_minipatches = (mask*final_attn)/final_attn.max()
        significant_minipatches = significant_minipatches.reshape((8,8))
        significant_pixels = cv2.resize(significant_minipatches, dsize=(resnames.shape[0],resnames.shape[1]))
        all_significant_pixels.append(significant_pixels)
    return all_significant_pixels

def get_spatial_attn(att_mat):
    """
    Given a multihead attention map, output normalized attention.
    """
    att_mat = torch.stack(att_mat).squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1).cpu()
    #att_mat = att_mat[-1]

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).cpu()
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    final_attn = v[0, 1:]/v[0, 1:].sum()

    # final_attn = final_attn/final_attn.max()
    # final_attn = final_attn.reshape([8,8])
    # final_attn
    return final_attn.detach().numpy()



class PyMolPlugin:
 def input(self, inputfile):
  self.parameters = PyIO.readParameters(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
  GRID_DIR = PyPluMA.prefix()+"/"+self.parameters["grid"]
  attn_native = open(PyPluMA.prefix()+"/"+self.parameters["attn_native"], "rb")
  attn_spatial = open(PyPluMA.prefix()+"/"+self.parameters["attn_spatial"], "rb")
  all_attn_native = pickle.load(attn_native)
  stat_spatial_attn_dict = pickle.load(attn_spatial)
  ppi_list = os.listdir(GRID_DIR)
  ppi_list = [x.split('.npy')[0] for x in ppi_list if 'resnames' not in x and '.ref' not in x]
  masif_test_native =  PISToN_dataset(GRID_DIR, ppi_list)
  ppi_idx = masif_test_native.ppi_list.index(self.parameters["id"])
  ppi = masif_test_native.ppi_list[ppi_idx]
  grid_dir = masif_test_native.grid_dir
  spatial_attn = all_attn_native[ppi_idx][0]
  print(f"Testing for {ppi}")
  all_significant_pixels = get_significant_pixels(ppi, grid_dir, spatial_attn, stat_spatial_attn_dict)
  resnames_path = grid_dir + '/' + ppi + '_resnames.npy'
  resnames = np.load(resnames_path, allow_pickle=True)

  for i, significant_pixels in enumerate(all_significant_pixels):
    feature = list(stat_spatial_attn_dict.keys())[i]
    all_significant_res = set()
    for i in range(resnames.shape[0]):
        for j in range(resnames.shape[1]):
            if significant_pixels[i][j]>0:
                all_significant_res.add(resnames[i][j][0])
                all_significant_res.add(resnames[i][j][1])

    all_significant_res = set([x.split('-')[0] for x in all_significant_res if x!=0])

    ## Generate pymol selection string
    chains_dict = defaultdict(list)
    for res in all_significant_res:
        #ch = res.split(':')[0]
        #atom_indx = res.split('-')[1].split(':')[0]
        ch, resi, resname = res.split(':')
        chains_dict[ch].append(resi)
        #chains_dict[ch].append(atom_indx)

    ## select
    pymol_string=f"select {feature},"
    for ch in chains_dict.keys():
        pymol_string += f" (chain {ch} and resi {','.join(chains_dict[ch])}) or"
        #pymol_string += f" (chain {ch} and index {','.join(chains_dict[ch])}) or"
    pymol_string = pymol_string.strip('or')
    print(pymol_string)
#     print(chains_dict)


