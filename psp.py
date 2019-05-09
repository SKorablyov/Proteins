"""
Protein structure prediction
"""
#Import block
import torch
import argparse
import torch.optim as optim
import requests
import visdom
import glob
import os.path
import os
import platform
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import torch.utils.data
import h5py
import Bio.PDB
import math
import pnerf.pnerf as pnerf
from datetime import datetime
import PeptideBuilder
import time
import torch.nn.utils.rnn as rnn_utils
from torch.nn.parameter import Parameter

"""
Preprocess data
"""
MAX_SEQUENCE_LENGTH = 2000


def process_raw_data(use_gpu, force_pre_processing_overwrite=True):
    print("Starting pre-processing of raw data...")
    input_files = glob.glob("data/raw/*")
    print(input_files)
    input_files_filtered = filter_input_files(input_files)
    for file_path in input_files_filtered:
        if platform.system() is 'Windows':
            filename = file_path.split('\\')[-1]
        else:
            filename = file_path.split('/')[-1]
        preprocessed_file_name = "data/preprocessed/" + filename + ".hdf5"

        # check if we should remove the any previously processed files
        if os.path.isfile(preprocessed_file_name):
            print("Preprocessed file for " + filename + " already exists.")
            if force_pre_processing_overwrite:
                print("force_pre_processing_overwrite flag set to True, overwriting old file...")
                os.remove(preprocessed_file_name)
            else:
                print("Skipping pre-processing for this file...")

        if not os.path.isfile(preprocessed_file_name):
            process_file(filename, preprocessed_file_name, use_gpu)
    print("Completed pre-processing.")


def read_protein_from_file(file_pointer):
    dict_ = {}
    _dssp_dict = {'L': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
    _mask_dict = {'-': 0, '+': 1}

    while True:
        next_line = file_pointer.readline()
        if next_line == '[ID]\n':
            id_ = file_pointer.readline()[:-1]
            dict_.update({'id': id_})
        elif next_line == '[PRIMARY]\n':
            primary = encode_primary_string(file_pointer.readline()[:-1])
            dict_.update({'primary': primary})
        elif next_line == '[EVOLUTIONARY]\n':
            evolutionary = []
            for residue in range(21): evolutionary.append(
                [float(step) for step in file_pointer.readline().split()])
            dict_.update({'evolutionary': evolutionary})
        elif next_line == '[SECONDARY]\n':
            secondary = list([_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]])
            dict_.update({'secondary': secondary})
        elif next_line == '[TERTIARY]\n':
            tertiary = []
            # 3 dimension
            for axis in range(3): tertiary.append(
                [float(coord) for coord in file_pointer.readline().split()])
            dict_.update({'tertiary': tertiary})
        elif next_line == '[MASK]\n':
            mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
            dict_.update({'mask': mask})
        elif next_line == '\n':
            return dict_
        elif next_line == '':
            return None


def process_file(input_file, output_file, use_gpu):
    print("Processing raw data file", input_file)

    # create output file
    f = h5py.File(output_file, 'w')
    current_buffer_size = 1
    current_buffer_allocation = 0
    dset1 = f.create_dataset('primary', (current_buffer_size, MAX_SEQUENCE_LENGTH),
                             maxshape=(None, MAX_SEQUENCE_LENGTH), dtype='int32')
    dset2 = f.create_dataset('tertiary', (current_buffer_size, MAX_SEQUENCE_LENGTH, 9),
                             maxshape=(None, MAX_SEQUENCE_LENGTH, 9), dtype='float')
    dset3 = f.create_dataset('mask', (current_buffer_size, MAX_SEQUENCE_LENGTH), maxshape=(None, MAX_SEQUENCE_LENGTH),
                             dtype='uint8')

    input_file_pointer = open("data/raw/" + input_file, "r")

    while True:
        # while there's more proteins to process
        next_protein = read_protein_from_file(input_file_pointer)
        if next_protein is None:
            break

        sequence_length = len(next_protein['primary'])

        if sequence_length > MAX_SEQUENCE_LENGTH:
            print("Dropping protein as length too long:", sequence_length)
            continue

        if current_buffer_allocation >= current_buffer_size:
            current_buffer_size = current_buffer_size + 1
            dset1.resize((current_buffer_size, MAX_SEQUENCE_LENGTH))
            dset2.resize((current_buffer_size, MAX_SEQUENCE_LENGTH, 9))
            dset3.resize((current_buffer_size, MAX_SEQUENCE_LENGTH))

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((9, MAX_SEQUENCE_LENGTH))
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)

        # masking and padding here happens so that the stored dataset is of the same size.
        # when the data is loaded in this padding is removed again.
        primary_padded[:sequence_length] = next_protein['primary']
        t_transposed = np.ravel(np.array(next_protein['tertiary']).T)
        t_reshaped = np.reshape(t_transposed, (sequence_length, 9)).T

        tertiary_padded[:, :sequence_length] = t_reshaped
        mask_padded[:sequence_length] = next_protein['mask']

        mask = torch.Tensor(mask_padded).type(dtype=torch.uint8)

        prim = torch.masked_select(torch.Tensor(primary_padded).type(dtype=torch.long), mask)
        pos = torch.masked_select(torch.Tensor(tertiary_padded), mask).view(9, -1).transpose(0, 1).unsqueeze(1) / 100

        if use_gpu:
            pos = pos.cuda()

        angles, batch_sizes = calculate_dihedral_angles_over_minibatch(pos, [len(prim)], use_gpu=use_gpu)

        tertiary, _ = get_backbone_positions_from_angular_prediction(angles, batch_sizes, use_gpu=use_gpu)
        tertiary = tertiary.squeeze(1)

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((MAX_SEQUENCE_LENGTH, 9))

        length_after_mask_removed = len(prim)

        primary_padded[:length_after_mask_removed] = prim.data.cpu().numpy()
        tertiary_padded[:length_after_mask_removed, :] = tertiary.data.cpu().numpy()
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        mask_padded[:length_after_mask_removed] = np.ones(length_after_mask_removed)

        dset1[current_buffer_allocation] = primary_padded
        dset2[current_buffer_allocation] = tertiary_padded
        dset3[current_buffer_allocation] = mask_padded
        current_buffer_allocation += 1

    print("Wrote output to", current_buffer_allocation, "proteins to", output_file)


def filter_input_files(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))
"""
Useful functions
"""

AA_ID_DICT = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
              'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
              'V': 18, 'W': 19,'Y': 20}

def contruct_dataloader_from_disk(filename, minibatch_size):
    return torch.utils.data.DataLoader(H5PytorchDataset(filename), batch_size=minibatch_size,
                                       shuffle=True, collate_fn=H5PytorchDataset.merge_samples_to_minibatch)


class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(H5PytorchDataset, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_proteins, self.max_sequence_len = self.h5pyfile['primary'].shape

    def __getitem__(self, index):
        mask = torch.Tensor(self.h5pyfile['mask'][index,:]).type(dtype=torch.uint8)
        prim = torch.masked_select(torch.Tensor(self.h5pyfile['primary'][index,:]).type(dtype=torch.long), mask)
        tertiary = torch.Tensor(self.h5pyfile['tertiary'][index][:int(mask.sum())]) # max length x 9
        return  prim, tertiary, mask

    def __len__(self):
        return self.num_proteins

    def merge_samples_to_minibatch(samples):
        samples_list = []
        for s in samples:
            samples_list.append(s)
        # sort according to length of aa sequence
        samples_list.sort(key=lambda x: len(x[0]), reverse=True)
        return zip(*samples_list)

def set_experiment_id(data_set_identifier, learning_rate, minibatch_size):
    output_string = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    output_string += "-" + data_set_identifier
    output_string += "-LR" + str(learning_rate).replace(".","_")
    output_string += "-MB" + str(minibatch_size)
    globals().__setitem__("experiment_id",output_string)

def write_out(*args, end='\n'):
    output_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + str.join(" ", [str(a) for a in args]) + end
    if globals().get("experiment_id") is not None:
        with open("output/"+globals().get("experiment_id")+".txt", "a+") as output_file:
            output_file.write(output_string)
            output_file.flush()
    print(output_string, end="")

def write_model_to_disk(model):
    path = "output/models/"+globals().get("experiment_id")+".model"
    torch.save(model,path)
    return path

def draw_plot(fig, plt, validation_dataset_size, sample_num, train_loss_values,
              validation_loss_values):
    def draw_with_vars():
        ax = fig.gca()
        ax2 = ax.twinx()
        plt.grid(True)
        plt.title("Training progress (" + str(validation_dataset_size) + " samples in validation set)")
        train_loss_plot, = ax.plot(sample_num, train_loss_values)
        ax.set_ylabel('Train Negative log likelihood')
        ax.yaxis.labelpad = 0
        validation_loss_plot, = ax2.plot(sample_num, validation_loss_values, color='black')
        ax2.set_ylabel('Validation loss')
        ax2.set_ylim(bottom=0)
        plt.legend([train_loss_plot, validation_loss_plot],
                   ['Train loss on last batch', 'Validation loss'])
        ax.set_xlabel('Minibatches processed (=network updates)', color='black')
    return draw_with_vars

def draw_ramachandran_plot(fig, plt, phi, psi):
    def draw_with_vars():
        ax = fig.gca()
        plt.grid(True)
        plt.title("Ramachandran plot")
        train_loss_plot, = ax.plot(phi, psi)
        ax.set_ylabel('Psi')
        ax.yaxis.labelpad = 0
        plt.legend([train_loss_plot],
                   ['Phi psi'])
        ax.set_xlabel('Phi', color='black')
    return draw_with_vars

def write_result_summary(accuracy):
    output_string = globals().get("experiment_id") + ": " + str(accuracy) + "\n"
    with open("output/result_summary.txt", "a+") as output_file:
        output_file.write(output_string)
        output_file.flush()
    print(output_string, end="")

def calculate_dihedral_angles_over_minibatch(atomic_coords_padded, batch_sizes, use_gpu):
    angles = []
    atomic_coords = atomic_coords_padded.transpose(0,1)
    for idx, _ in enumerate(batch_sizes):
        angles.append(calculate_dihedral_angles(atomic_coords[idx][:batch_sizes[idx]], use_gpu))
    return torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(angles))

def protein_id_to_str(protein_id_list):
    _aa_dict_inverse = {v: k for k, v in AA_ID_DICT.items()}
    aa_list = []
    for a in protein_id_list:
        aa_symbol = _aa_dict_inverse[int(a)]
        aa_list.append(aa_symbol)
    return aa_list

def calculate_dihedral_angles(atomic_coords, use_gpu):

    assert int(atomic_coords.shape[1]) == 9
    atomic_coords = atomic_coords.contiguous().view(-1,3)

    zero_tensor = torch.tensor(0.0)
    if use_gpu:
        zero_tensor = zero_tensor.cuda()

    dihedral_list = [zero_tensor,zero_tensor]
    dihedral_list.extend(compute_dihedral_list(atomic_coords))
    dihedral_list.append(zero_tensor)
    angles = torch.tensor(dihedral_list).view(-1,3)
    return angles

def compute_dihedral_list(atomic_coords):
    # atomic_coords is -1 x 3
    ba = atomic_coords[1:] - atomic_coords[:-1]
    ba /= ba.norm(dim=1).unsqueeze(1)
    ba_neg = -1 * ba

    n1_vec = torch.cross(ba[:-2], ba_neg[1:-1], dim=1)
    n2_vec = torch.cross(ba_neg[1:-1], ba[2:], dim=1)
    n1_vec /= n1_vec.norm(dim=1).unsqueeze(1)
    n2_vec /= n2_vec.norm(dim=1).unsqueeze(1)

    m1_vec = torch.cross(n1_vec, ba_neg[1:-1], dim=1)

    x = torch.sum(n1_vec*n2_vec,dim=1)
    y = torch.sum(m1_vec*n2_vec,dim=1)

    return torch.atan2(y,x)



def write_to_pdb(structure, prot_id):
    out = Bio.PDB.PDBIO()
    out.set_structure(structure)
    out.save("output/protein_" + str(prot_id) + ".pdb")

def calc_pairwise_distances(chain_a, chain_b, use_gpu):
    distance_matrix = torch.Tensor(chain_a.size()[0], chain_b.size()[0]).type(torch.float)
    # add small epsilon to avoid boundary issues
    epsilon = 10 ** (-4) * torch.ones(chain_a.size(0), chain_b.size(0))
    if use_gpu:
        distance_matrix = distance_matrix.cuda()
        epsilon = epsilon.cuda()

    for i, row in enumerate(chain_a.split(1)):
        distance_matrix[i] = torch.sum((row.expand_as(chain_b) - chain_b) ** 2, 1).view(1, -1)

    return torch.sqrt(distance_matrix + epsilon)

def calc_drmsd(chain_a, chain_b, use_gpu=False):
    assert len(chain_a) == len(chain_b)
    distance_matrix_a = calc_pairwise_distances(chain_a, chain_a, use_gpu)
    distance_matrix_b = calc_pairwise_distances(chain_b, chain_b, use_gpu)
    return torch.norm(distance_matrix_a - distance_matrix_b, 2) \
            / math.sqrt((len(chain_a) * (len(chain_a) - 1)))

# method for translating a point cloud to its center of mass
def transpose_atoms_to_center_of_mass(x):
    # calculate com by summing x, y and z respectively
    # and dividing by the number of points
    centerOfMass = np.matrix([[x[0, :].sum() / x.shape[1]],
                    [x[1, :].sum() / x.shape[1]],
                    [x[2, :].sum() / x.shape[1]]])
    # translate points to com and return
    return x - centerOfMass

def calc_rmsd(chain_a, chain_b):
    # move to center of mass
    a = chain_a.cpu().numpy().transpose()
    b = chain_b.cpu().numpy().transpose()
    X = transpose_atoms_to_center_of_mass(a)
    Y = transpose_atoms_to_center_of_mass(b)

    R = Y * X.transpose()
    # extract the singular values
    _, S, _ = np.linalg.svd(R)
    # compute RMSD using the formular
    E0 = sum(list(np.linalg.norm(x) ** 2 for x in X.transpose())
             + list(np.linalg.norm(x) ** 2 for x in Y.transpose()))
    TraceS = sum(S)
    RMSD = np.sqrt((1 / len(X.transpose())) * (E0 - 2 * TraceS))
    return RMSD

def calc_angular_difference(a1, a2):
    a1 = a1.transpose(0,1).contiguous()
    a2 = a2.transpose(0,1).contiguous()
    sum = 0
    for idx, _ in enumerate(a1):
        assert a1[idx].shape[1] == 3
        assert a2[idx].shape[1] == 3
        a1_element = a1[idx].view(-1, 1)
        a2_element = a2[idx].view(-1, 1)
        sum += torch.sqrt(torch.mean(
            torch.min(torch.abs(a2_element - a1_element),
                      2 * math.pi - torch.abs(a2_element - a1_element)
                      ) ** 2))
    return sum / a1.shape[0]

def structures_to_backbone_atoms_padded(structures):
    backbone_atoms_list = []
    for structure in structures:
        backbone_atoms_list.append(structure_to_backbone_atoms(structure))
    backbone_atoms_padded, batch_sizes_backbone = torch.nn.utils.rnn.pad_packed_sequence(
        torch.nn.utils.rnn.pack_sequence(backbone_atoms_list))
    return backbone_atoms_padded, batch_sizes_backbone

def structure_to_backbone_atoms(structure):
    predicted_coords = []
    for res in structure.get_residues():
        predicted_coords.append(torch.Tensor(res["N"].get_coord()))
        predicted_coords.append(torch.Tensor(res["CA"].get_coord()))
        predicted_coords.append(torch.Tensor(res["C"].get_coord()))
    return torch.stack(predicted_coords).view(-1,9)

def get_backbone_positions_from_angular_prediction(angular_emissions, batch_sizes, use_gpu):
    # angular_emissions -1 x minibatch size x 3 (omega, phi, psi)
    points = pnerf.dihedral_to_point(angular_emissions, use_gpu)
    coordinates = pnerf.point_to_coordinate(points, use_gpu) / 100 # devide by 100 to angstrom unit
    return coordinates.transpose(0,1).contiguous().view(len(batch_sizes),-1,9).transpose(0,1), batch_sizes


def calc_avg_drmsd_over_minibatch(backbone_atoms_padded, actual_coords_padded, batch_sizes):
    backbone_atoms_list = list(
        [backbone_atoms_padded[:batch_sizes[i], i] for i in range(int(backbone_atoms_padded.size(1)))])
    actual_coords_list = list(
        [actual_coords_padded[:batch_sizes[i], i] for i in range(int(actual_coords_padded.size(1)))])
    drmsd_avg = 0
    for idx, backbone_atoms in enumerate(backbone_atoms_list):
        actual_coords = actual_coords_list[idx].transpose(0, 1).contiguous().view(-1, 3)
        drmsd_avg += calc_drmsd(backbone_atoms.transpose(0, 1).contiguous().view(-1, 3), actual_coords)
    return drmsd_avg / len(backbone_atoms_list)

def encode_primary_string(primary):
    return list([AA_ID_DICT[aa] for aa in primary])

def get_structure_from_angles(aa_list_encoded, angles):
    #aa_list = protein_id_to_str(aa_list_encoded)
    omega_list = angles[1:,0]
    #print(list(angles.size()))
    phi_list = angles[1:,1]
    psi_list = angles[:-1,2]


#def get_structure_from_angles(aa_list_encoded, omega_list,phi_list,psi_list):
    aa_list = protein_id_to_str(aa_list_encoded)

    assert len(aa_list) == len(phi_list)+1 == len(psi_list)+1 == len(omega_list)+1
    structure = PeptideBuilder.make_structure(aa_list,
                                              list(map(lambda x: math.degrees(x), phi_list)),
                                              list(map(lambda x: math.degrees(x), psi_list)),
                                              list(map(lambda x: math.degrees(x), omega_list)))
    return structure
def intial_pos_from_aa_string(batch_aa_string):
    structures = []

    for aa_string in batch_aa_string:


        angles =  np.array([np.repeat([-120], len(aa_string)),
                  np.repeat([140], len(aa_string)),
                  np.repeat([-370], len(aa_string) )],np.float)
        angles = np.transpose(angles)

        #print(angles.shape)


        structure = get_structure_from_angles(aa_string,angles)
        structures.append(structure)
    return structures

def pass_messages(aa_features, message_transformation, use_gpu):
    # aa_features (#aa, #features) - each row represents the amino acid type (embedding) and the positions of the backbone atoms
    # message_transformation: (-1 * 2 * feature_size) -> (-1 * output message size)
    feature_size = aa_features.size(1)
    aa_count = aa_features.size(0)
    eye = torch.eye(aa_count,dtype=torch.uint8).view(-1).expand(2,feature_size,-1).transpose(1,2).transpose(0,1)
    eye_inverted = torch.ones(eye.size(),dtype=torch.uint8) - eye
    if use_gpu:
        eye_inverted = eye_inverted.cuda()
    features_repeated = aa_features.repeat((aa_count,1)).view((aa_count,aa_count,feature_size))
    aa_messages = torch.stack((features_repeated.transpose(0,1), features_repeated)).transpose(0,1).transpose(1,2).view(-1,2,feature_size)
    aa_msg_pairs = torch.masked_select(aa_messages,eye_inverted).view(-1,2,feature_size) # (aa_count^2 - aa_count) x 2 x aa_features     (all pairs except for reflexive connections)
    transformed = message_transformation(aa_msg_pairs).view(aa_count, aa_count - 1, -1)
    transformed_sum = transformed.sum(dim=1) # aa_count x output message size
    return transformed_sum

class Pranam(torch.nn.Module):
    def __init__(self,embed_size,out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Pranam, self).__init__()

        #self.embed_size=16
        dtype = torch.cuda.FloatTensor
        self.w1 = Parameter(2 * (torch.rand(embed_size ,embed_size )).type(dtype), requires_grad=True)
        #self.w1.to(device)
        self.w2 = Parameter((2 * torch.rand(embed_size, out).type(dtype)) / embed_size ** 0.5, requires_grad=True)

    def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """

            w_sub = (self.w1.mm(self.w2))
            w = w_sub.tanh()
            # w = nn.Tanh(w_sub)
            out = x.mm(w)
            return out

class Pranam_mod(torch.nn.Module):
    def __init__(self,inp,embed_size,out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Pranam_mod, self).__init__()

        #self.embed_size=16
        dtype = torch.cuda.FloatTensor
        self.w1 = Parameter(2  * (torch.rand(embed_size*inp ,embed_size*inp )).type(dtype), requires_grad=True)
        #self.w1.to(device)
        self.w2 = Parameter((2  * torch.rand(embed_size*inp, out).type(dtype)) / embed_size ** 0.5, requires_grad=True)

    def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            X = x.repeat(self.inp)
            w_sub = (self.w1.mm(self.w2))
            w = w_sub.tanh()
            # w = nn.Tanh(w_sub)
            out = X.mm(w)

            return out

"""
Models 
"""

class BaseModel(nn.Module):
    def __init__(self, use_gpu, embedding_size):
        super(BaseModel, self).__init__()

        # initialize model variables
        self.use_gpu = use_gpu
        self.embedding_size = embedding_size
        self.historical_rmsd_avg_values = list()
        self.historical_drmsd_avg_values = list()

    def get_embedding_size(self):
        return self.embedding_size

    def embed(self, original_aa_string):
        data, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(original_aa_string))

        # one-hot encoding
        start_compute_embed = time.time()
        prot_aa_list = data.unsqueeze(1)
        embed_tensor = torch.zeros(prot_aa_list.size(0), 21, prot_aa_list.size(2)) # 21 classes
        if self.use_gpu:
            prot_aa_list = prot_aa_list.cuda()
            embed_tensor = embed_tensor.cuda()
        input_sequences = embed_tensor.scatter_(1, prot_aa_list.data, 1).transpose(1,2)
        end = time.time()
        #write_out("Embed time:", end - start_compute_embed)
        packed_input_sequences = rnn_utils.pack_padded_sequence(input_sequences, batch_sizes)
        return packed_input_sequences

    def compute_loss(self, minibatch):
        (original_aa_string, actual_coords_list, mask) = minibatch

        emissions, backbone_atoms_padded, batch_sizes = self._get_network_emissions(original_aa_string)
        actual_coords_list_padded, batch_sizes_coords = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(actual_coords_list))
        if self.use_gpu:
            actual_coords_list_padded = actual_coords_list_padded.cuda()
        start = time.time()
        emissions_actual, batch_sizes_actual = \
            calculate_dihedral_angles_over_minibatch(actual_coords_list_padded, batch_sizes_coords, self.use_gpu)
        drmsd_avg = calc_avg_drmsd_over_minibatch(backbone_atoms_padded, actual_coords_list_padded, batch_sizes)
        #write_out("Angle calculation time:", time.time() - start)
        if self.use_gpu:
            emissions_actual = emissions_actual.cuda()
            drmsd_avg = drmsd_avg.cuda()
            emissions = emissions.cuda()
        angular_loss = calc_angular_difference(emissions, emissions_actual)

        return  drmsd_avg #+angular_loss

    def forward(self, original_aa_string):
        return self._get_network_emissions(original_aa_string)

    def evaluate_model(self, data_loader):
        loss = 0
        data_total = []
        dRMSD_list = []
        RMSD_list = []
        for i, data in enumerate(data_loader, 0):
            primary_sequence, tertiary_positions, mask = data
            #start = time.time()
            predicted_angles, backbone_atoms, batch_sizes = self(primary_sequence)
            #write_out("Apply model to validation minibatch:", time.time() - start)
            cpu_predicted_angles = predicted_angles.transpose(0, 1).cpu().detach()
            cpu_predicted_backbone_atoms = backbone_atoms.transpose(0, 1).cpu().detach()
            minibatch_data = list(zip(primary_sequence,
                                      tertiary_positions,
                                      cpu_predicted_angles,
                                      cpu_predicted_backbone_atoms))
            data_total.extend(minibatch_data)
            #start = time.time()
            for primary_sequence, tertiary_positions, predicted_pos, predicted_backbone_atoms in minibatch_data:
                actual_coords = tertiary_positions.transpose(0, 1).contiguous().view(-1, 3)
                predicted_coords = predicted_backbone_atoms[:len(primary_sequence)].transpose(0, 1).contiguous().view(
                    -1, 3).detach()
                rmsd = calc_rmsd(predicted_coords, actual_coords)
                drmsd = calc_drmsd(predicted_coords, actual_coords)
                RMSD_list.append(rmsd)
                dRMSD_list.append(drmsd)
                error = rmsd
                loss += error
                #end = time.time()
            #write_out("Calculate validation loss for minibatch took:", end - start)
        loss /= data_loader.dataset.__len__()
        self.historical_rmsd_avg_values.append(float(torch.Tensor(RMSD_list).mean()))
        self.historical_drmsd_avg_values.append(float(torch.Tensor(dRMSD_list).mean()))

        prim = data_total[0][0]
        pos = data_total[0][1]
        pos_pred = data_total[0][3]
        if self.use_gpu:
            pos = pos.cuda()
            pos_pred = pos_pred.cuda()
        angles = calculate_dihedral_angles(pos, self.use_gpu)
        angles_pred = calculate_dihedral_angles(pos_pred, self.use_gpu)
        write_to_pdb(get_structure_from_angles(prim, angles), "test")
        write_to_pdb(get_structure_from_angles(prim, angles_pred), "test_pred")

        data = {}
        data["pdb_data_pred"] = open("output/protein_test_pred.pdb", "r").read()
        data["pdb_data_true"] = open("output/protein_test.pdb", "r").read()
        data["phi_actual"] = list([math.degrees(float(v)) for v in angles[1:, 1]])
        data["psi_actual"] = list([math.degrees(float(v)) for v in angles[:-1, 2]])
        data["phi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[1:, 1]])
        data["psi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[:-1, 2]])
        data["rmsd_avg"] = self.historical_rmsd_avg_values
        data["drmsd_avg"] = self.historical_drmsd_avg_values



        return (loss, data)


torch.manual_seed(1)


class LstmModel(BaseModel):
    def __init__(self, embedding_size, minibatch_size, use_gpu):
        super(LstmModel, self).__init__(use_gpu, embedding_size)

        self.hidden_size = 25
        self.num_lstm_layers = 2
        self.mixture_size = 500
        self.bi_lstm = nn.LSTM(self.get_embedding_size(), self.hidden_size,
                               num_layers=self.num_lstm_layers, bidirectional=True, bias=True)
        self.hidden_to_labels = nn.Linear(self.hidden_size * 2, self.mixture_size, bias=True) # * 2 for bidirectional
        self.init_hidden(minibatch_size)
        self.softmax_to_angle = soft_to_angle(self.mixture_size)
        self.soft = nn.LogSoftmax(2)
        self.bn = nn.BatchNorm1d(self.mixture_size)

    def init_hidden(self, minibatch_size):
        # number of layers (* 2 since bidirectional), minibatch_size, hidden size
        initial_hidden_state = torch.zeros(self.num_lstm_layers * 2, minibatch_size, self.hidden_size)
        initial_cell_state = torch.zeros(self.num_lstm_layers * 2, minibatch_size, self.hidden_size)
        if self.use_gpu:
            initial_hidden_state = initial_hidden_state.cuda()
            initial_cell_state = initial_cell_state.cuda()
        self.hidden_layer = (autograd.Variable(initial_hidden_state),
                             autograd.Variable(initial_cell_state))

    def _get_network_emissions(self, original_aa_string):
        packed_input_sequences = self.embed(original_aa_string)
        minibatch_size = int(packed_input_sequences[1][0])
        self.init_hidden(minibatch_size)
        (data, bi_lstm_batches), self.hidden_layer = self.bi_lstm(packed_input_sequences, self.hidden_layer)

        emissions_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(self.hidden_to_labels(data), bi_lstm_batches))
        #print(bi_lstm_batches==torch.ones(201,dtype=torch.long))
        x = emissions_padded.transpose(0,1).transpose(1,2) # minibatch_size, self.mixture_size, -1
        x = self.bn(x)
        x = x.transpose(1,2) #(minibatch_size, -1, self.mixture_size)
        p = torch.exp(self.soft(x))
        output_angles = self.softmax_to_angle(p).transpose(0,1) # max size, minibatch size, 3 (angels)
        #print(batch_sizes.size())
        backbone_atoms_padded, batch_sizes_backbone = get_backbone_positions_from_angular_prediction(output_angles, batch_sizes, self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes
"""
class ExampleCnnModel(BaseModel):
    def __init__(self, embedding_size, minibatch_size, use_gpu):
        super(ExampleCnnModel, self).__init__(use_gpu, embedding_size)

        self.hidden_size = 25
        self.num_lstm_layers = 2
        self.mixture_size = 500

        self.convolution = nn.Conv1d(21,500, 3,padding=1)
        self.convolution_11 = nn.Conv1d(21, 200, 3, padding=1)
        self.convolution_12 = nn.Conv1d(200, 500, 3, padding=1)

        self.softmax_to_angle = soft_to_angle(self.mixture_size)
        self.soft = nn.LogSoftmax(2)
        self.bn = nn.BatchNorm1d(self.mixture_size)

    def _get_network_emissions(self, original_aa_string):
        packed_input_sequences = self.embed(original_aa_string)
        #print(packed_input_sequences.data.size())
        
        a = (packed_input_sequences.data)
        #print((a.size()))
        minibatch_size = int(packed_input_sequences[1][0])
        batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(torch.ones(201,dtype=torch.long)))
        #batch_sizes=torch.ones(201,dtype=torch.long)
        a= a.unsqueeze(2)

        # 40200 x 50 x 1 -->  201 500
        x = self.convolution_11(a)
        x = self.convolution_12(x)
        #print(x.size())
        x = x.view(1, 201, 500)
        #x = x.reshape(201,21)
        #print(x.size())
        p = x
        output_angles = self.softmax_to_angle(p).transpose(0,1) # max size, minibatch size, 3 (angels)
        backbone_atoms_padded, batch_sizes_backbone = get_backbone_positions_from_angular_prediction(output_angles, batch_sizes, self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes
        
        emissions_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(self.hidden_to_labels(data), bi_lstm_batches))
        # print(bi_lstm_batches==torch.ones(201,dtype=torch.long))
        x = emissions_padded.transpose(0, 1).transpose(1, 2)  # minibatch_size, self.mixture_size, -1
        x = self.bn(x)
        x = x.transpose(1, 2)  # (minibatch_size, -1, self.mixture_size)
        p = torch.exp(self.soft(x))
        output_angles = self.softmax_to_angle(p).transpose(0, 1)  # max size, minibatch size, 3 (angels)
        # print(batch_sizes.size())
        backbone_atoms_padded, batch_sizes_backbone = get_backbone_positions_from_angular_prediction(output_angles,
                                                                                                     batch_sizes,
                                                                                                     self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes

"""


class RNNModel(BaseModel):
    def __init__(self, embedding_size, minibatch_size, use_gpu):
        super(RNNModel, self).__init__(use_gpu, embedding_size)
        self.recurrent_steps = 2
        self.hidden_size = 50
        self.msg_output_size = 50
        self.output_size = 9 # 3 dimensions * 3 coordinates for each aa
        #print(embedding_size * 2 + 9)
        self.f_to_hid = nn.Linear((embedding_size*2 + 9), self.hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.hid_to_pos = nn.Linear(self.hidden_size, self.msg_output_size, bias=True)
        self.g = nn.Linear(embedding_size + 9 + self.msg_output_size, 9, bias=True) # (last state + orginal state)
        self.use_gpu = use_gpu





    def f(self, aa_features):
        # aa_features: msg_count * 2 * feature_count
        min_distance = torch.tensor(0.000001)
        if self.use_gpu:
            min_distance = min_distance.cuda()
        aa_features_transformed = torch.cat(
            (
                aa_features[:,0,0:21],
                aa_features[:,1,0:21],
                aa_features[:,0,21:30] - aa_features[:,1,21:30]
            ), dim=1)
        out = self.hid_to_pos(self.relu(self.f_to_hid(aa_features_transformed)))


        return out  # msg_count * outputsize

    def _get_network_emissions(self, original_aa_string):
        initial_aa_pos = intial_pos_from_aa_string(original_aa_string)
        packed_input_sequences = self.embed(original_aa_string)
        backbone_atoms_padded, batch_sizes_backbone = structures_to_backbone_atoms_padded(initial_aa_pos)
        if self.use_gpu:
            backbone_atoms_padded = backbone_atoms_padded.cuda()
        embedding_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(packed_input_sequences))
        for i in range(self.recurrent_steps):
            combined_features = torch.cat((embedding_padded,backbone_atoms_padded),dim=2)
            for idx, aa_features in enumerate(combined_features.transpose(0,1)):
                msg = pass_messages(aa_features, self.f, self.use_gpu) # aa_count * output size
                backbone_atoms_padded[:,idx] = self.g(torch.cat((aa_features, msg), dim=1))

        output_angles, batch_sizes = calculate_dihedral_angles_over_minibatch(backbone_atoms_padded, batch_sizes_backbone, self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes


class CNN3Model(BaseModel):
    #drmsd 14.7
    def __init__(self, embedding_size, minibatch_size, use_gpu):
        super(CNN3Model, self).__init__(use_gpu, embedding_size)
        self.recurrent_steps = 2
        self.hidden_size = 50
        self.msg_output_size = 50
        self.output_size = 9 # 3 dimensions * 3 coordinates for each aa
        self.convolution = nn.Conv1d(51,51,3,padding=1)
        self.convolution2 = nn.Conv1d(51, 50, 3, padding=1)
        self.f_to_hid = nn.Linear((embedding_size*2 + 9), self.hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.hid_to_pos = nn.Linear(self.hidden_size, self.msg_output_size, bias=True)
        self.g = nn.Linear(embedding_size + 9 + self.msg_output_size, 9, bias=True) # (last state + orginal state)
        self.use_gpu = use_gpu



    def f(self, aa_features):
        # aa_features: msg_count * 2 * feature_count
        min_distance = torch.tensor(0.000001)
        #print(aa_features.size())
        if self.use_gpu:
            min_distance = min_distance.cuda()
        aa_features_transformed = torch.cat(
            (
                aa_features[:,0,0:21],
                aa_features[:,1,0:21],
                aa_features[:,0,21:30] - aa_features[:,1,21:30]
            ), dim=1)

        aa_features_transformed_1=aa_features_transformed.unsqueeze(2)
        sizes = aa_features_transformed.size()
        # 40200 x 50 x 1 -->  201 500
        x = self.convolution(aa_features_transformed_1)
        x = self.convolution2(x)
        #print(x.size())
        x = x.reshape(sizes[0],sizes[1]-1)
        #print(x.size())
        #output = self.hid_to_pos(self.relu(x))

        #print(output.size())
        #print(self.hid_to_pos(x).size())
        return self.hid_to_pos(self.relu(x)) # msg_count * outputsize

    def _get_network_emissions(self, original_aa_string):
        initial_aa_pos = intial_pos_from_aa_string(original_aa_string)
        packed_input_sequences = self.embed(original_aa_string)
        backbone_atoms_padded, batch_sizes_backbone = structures_to_backbone_atoms_padded(initial_aa_pos)
        if self.use_gpu:
            backbone_atoms_padded = backbone_atoms_padded.cuda()
        embedding_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(packed_input_sequences))
        for i in range(self.recurrent_steps):
            combined_features = torch.cat((embedding_padded,backbone_atoms_padded),dim=2)
            for idx, aa_features in enumerate(combined_features.transpose(0,1)):
                msg = pass_messages(aa_features, self.f, self.use_gpu) # aa_count * output size
                backbone_atoms_padded[:,idx] = self.g(torch.cat((aa_features, msg), dim=1))

        output_angles, batch_sizes = calculate_dihedral_angles_over_minibatch(backbone_atoms_padded, batch_sizes_backbone, self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes

class CNN5Model(BaseModel):
    #drmsd 14.7
    def __init__(self, embedding_size, minibatch_size, use_gpu):
        super(CNN5Model, self).__init__(use_gpu, embedding_size)
        self.recurrent_steps = 2
        self.hidden_size = 50
        self.msg_output_size = 50
        self.output_size = 9 # 3 dimensions * 3 coordinates for each aa
        self.convolution = nn.Conv1d(51,102,5,padding=2)
        self.convolution2 = nn.Conv1d(102, 204, 5, padding=2)
        self.avg_pool = nn.AvgPool1d(5)
        self.maxpool = nn.MaxPool1d(5)
        self.relu = nn.ReLU()
        self.hid_to_pos = nn.Linear(204, self.msg_output_size, bias=True)
        self.g = nn.Linear(embedding_size + 9 + self.msg_output_size, 9, bias=True) # (last state + orginal state)
        self.use_gpu = use_gpu



    def f(self, aa_features):
        # aa_features: msg_count * 2 * feature_count
        min_distance = torch.tensor(0.000001)
        #print(aa_features.size())
        if self.use_gpu:
            min_distance = min_distance.cuda()
        aa_features_transformed = torch.cat(
            (
                aa_features[:,0,0:21],
                aa_features[:,1,0:21],
                aa_features[:,0,21:30] - aa_features[:,1,21:30]
            ), dim=1)

        aa_features_transformed_1=aa_features_transformed.unsqueeze(2)

        # 40200 x 51 x 1 -->  40200 x 50
        x = self.convolution(aa_features_transformed_1)
        sizes = aa_features_transformed.size()
        # 40200 x 50 x 1 -->  201 500
        x = self.convolution(aa_features_transformed_1)
        x = self.convolution2(x)
        # print(x.size())
        x = x.reshape(sizes[0], sizes[1] - 1)
        #print(x.size())
        #output = self.hid_to_pos(self.relu(x))

        #print(output.size())
        #print(self.hid_to_pos(x).size())
        return self.hid_to_pos(x) # msg_count * outputsize

    def _get_network_emissions(self, original_aa_string):
        initial_aa_pos = intial_pos_from_aa_string(original_aa_string)
        packed_input_sequences = self.embed(original_aa_string)
        backbone_atoms_padded, batch_sizes_backbone = structures_to_backbone_atoms_padded(initial_aa_pos)
        if self.use_gpu:
            backbone_atoms_padded = backbone_atoms_padded.cuda()
        embedding_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(packed_input_sequences))
        for i in range(self.recurrent_steps):
            combined_features = torch.cat((embedding_padded,backbone_atoms_padded),dim=2)
            for idx, aa_features in enumerate(combined_features.transpose(0,1)):
                msg = pass_messages(aa_features, self.f, self.use_gpu) # aa_count * output size
                backbone_atoms_padded[:,idx] = self.g(torch.cat((aa_features, msg), dim=1))

        output_angles, batch_sizes = calculate_dihedral_angles_over_minibatch(backbone_atoms_padded, batch_sizes_backbone, self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes

class CNN7Model(BaseModel):
    # drmsd 14.6
    def __init__(self, embedding_size, minibatch_size, use_gpu):
        super(CNN7Model, self).__init__(use_gpu, embedding_size)
        self.recurrent_steps = 2
        self.hidden_size = 50
        self.msg_output_size = 50
        self.output_size = 9 # 3 dimensions * 3 coordinates for each aa
        self.convolution = nn.Conv1d(51,51,7,padding=3)
        self.convolution2 = nn.Conv1d(51, 50, 7, padding=3)
        self.f_to_hid = nn.Linear((embedding_size*2 + 9), self.hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.hid_to_pos = nn.Linear(self.hidden_size, self.msg_output_size, bias=True)
        self.g = nn.Linear(embedding_size + 9 + self.msg_output_size, 9, bias=True) # (last state + orginal state)
        self.use_gpu = use_gpu



    def f(self, aa_features):
        # aa_features: msg_count * 2 * feature_count
        min_distance = torch.tensor(0.000001)
        #print(aa_features.size())
        if self.use_gpu:
            min_distance = min_distance.cuda()
        aa_features_transformed = torch.cat(
            (
                aa_features[:,0,0:21],
                aa_features[:,1,0:21],
                aa_features[:,0,21:30] - aa_features[:,1,21:30]
            ), dim=1)

        aa_features_transformed_1=aa_features_transformed.unsqueeze(2)
        sizes = aa_features_transformed.size()
        # 40200 x 50 x 1 -->  201 500
        x = self.convolution(aa_features_transformed_1)
        x = self.convolution2(x)
        # print(x.size())
        x = x.reshape(sizes[0], sizes[1] - 1)
        #print(x.size())
        #output = self.hid_to_pos(self.relu(x))

        #print(output.size())
        #print(self.hid_to_pos(x).size())
        return self.hid_to_pos(x) # msg_count * outputsize

    def _get_network_emissions(self, original_aa_string):
        initial_aa_pos = intial_pos_from_aa_string(original_aa_string)
        packed_input_sequences = self.embed(original_aa_string)
        backbone_atoms_padded, batch_sizes_backbone = structures_to_backbone_atoms_padded(initial_aa_pos)
        if self.use_gpu:
            backbone_atoms_padded = backbone_atoms_padded.cuda()
        embedding_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(packed_input_sequences))
        for i in range(self.recurrent_steps):
            combined_features = torch.cat((embedding_padded,backbone_atoms_padded),dim=2)
            for idx, aa_features in enumerate(combined_features.transpose(0,1)):
                msg = pass_messages(aa_features, self.f, self.use_gpu) # aa_count * output size
                backbone_atoms_padded[:,idx] = self.g(torch.cat((aa_features, msg), dim=1))

        output_angles, batch_sizes = calculate_dihedral_angles_over_minibatch(backbone_atoms_padded, batch_sizes_backbone, self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes
class CNN_Model(BaseModel):
    def __init__(self, embedding_size, minibatch_size, use_gpu):
        super(CNN_Model, self).__init__(use_gpu, embedding_size)
        self.steps = 1

        self.output_size = 9 # 3 dimensions * 3 coordinates for each aa

        self.convolution = nn.Conv1d(200,400, kernel_size=11)
        self.convolution2 = nn.Conv1d(400, 800, kernel_size=11)
        self.convolution3 = nn.Conv1d(800,1600,kernel_size=4)

        self.max_pool = nn.MaxPool1d(2)
        self.avg_pool = nn.AvgPool1d(2)

        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(1600,160)
        self.linear2 = nn.Linear(160,16)
        self.linear3 = nn.Linear(16,9)

        self.use_gpu = use_gpu

    def f(self, aa_features):
        # aa_features: msg_count * 2 * feature_count
        min_distance = torch.tensor(0.000001)
        # print(aa_features.size())
        if self.use_gpu:
            min_distance = min_distance.cuda()
        print(aa_features.size())
        aa_features_transformed = torch.cat(
            (
                aa_features[:, 0, 0:21],
                aa_features[:, 1, 0:21],
                aa_features[:, 0, 21:30] - aa_features[:, 1, 21:30]
            ), dim=1)
        sizes = aa_features_transformed.size()
        x = aa_features_transformed.view(sizes)

        return

    def _get_network_emissions(self, original_aa_string):
        initial_aa_pos = intial_pos_from_aa_string(original_aa_string)
        packed_input_sequences = self.embed(original_aa_string)
        backbone_atoms_padded, batch_sizes_backbone = structures_to_backbone_atoms_padded(initial_aa_pos)
        if self.use_gpu:
            backbone_atoms_padded = backbone_atoms_padded.cuda()
        embedding_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(packed_input_sequences))
        for i in range(self.recurrent_steps):
            combined_features = torch.cat((embedding_padded,backbone_atoms_padded),dim=2)
            for idx, aa_features in enumerate(combined_features.transpose(0,1)):
                msg = pass_messages(aa_features, self.f, self.use_gpu) # aa_count * output size
                ou = self.g(torch.cat((aa_features, msg), dim=1))
                print(ou.size())
                backbone_atoms_padded[:,idx] = self.g(torch.cat((aa_features, msg), dim=1))

        output_angles, batch_sizes = calculate_dihedral_angles_over_minibatch(backbone_atoms_padded, batch_sizes_backbone, self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes

class CNN1Model(BaseModel):
    # drmsd 14.4
    def __init__(self, embedding_size, minibatch_size, use_gpu):
        super(CNN1Model, self).__init__(use_gpu, embedding_size)
        self.recurrent_steps = 1
        self.hidden_size = 50
        self.msg_output_size = 50
        self.output_size = 9 # 3 dimensions * 3 coordinates for each aa
        self.convolution = nn.Conv1d(51,50,1)
        self.convolution2 = nn.Conv1d(51, 50, 7, padding=3)
        self.f_to_hid = nn.Linear((embedding_size*2 + 9), self.hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.hid_to_pos = nn.Linear(self.hidden_size, self.msg_output_size, bias=True)
        print(embedding_size + 9 + self.msg_output_size,)
        self.g = nn.Linear(embedding_size + 9 + self.msg_output_size, 9, bias=True) # (last state + orginal state)
        self.use_gpu = use_gpu



    def f(self, aa_features):
        # aa_features: msg_count * 2 * feature_count
        min_distance = torch.tensor(0.000001)
        #print(aa_features.size())
        if self.use_gpu:
            min_distance = min_distance.cuda()
        print(aa_features.size())
        aa_features_transformed = torch.cat(
            (
                aa_features[:,0,0:21],
                aa_features[:,1,0:21],
                aa_features[:,0,21:30] - aa_features[:,1,21:30]
            ), dim=1)

        aa_features_transformed_1=aa_features_transformed.unsqueeze(2)
        sizes = aa_features_transformed.size()
        # 40200 x 50 x 1 -->  201 500
        x = self.convolution(aa_features_transformed_1)
        x = self.convolution2(x)
        # print(x.size())
        x = x.reshape(sizes[0], sizes[1] - 1)
        #print(x.size())
        #output = self.hid_to_pos(self.relu(x))

        #print(output.size())
        #print(self.hid_to_pos(x).size())
        return self.hid_to_pos(x) # msg_count * outputsize

    def _get_network_emissions(self, original_aa_string):
        initial_aa_pos = intial_pos_from_aa_string(original_aa_string)
        packed_input_sequences = self.embed(original_aa_string)
        backbone_atoms_padded, batch_sizes_backbone = structures_to_backbone_atoms_padded(initial_aa_pos)
        if self.use_gpu:
            backbone_atoms_padded = backbone_atoms_padded.cuda()
        embedding_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(packed_input_sequences))
        for i in range(self.recurrent_steps):
            combined_features = torch.cat((embedding_padded,backbone_atoms_padded),dim=2)
            for idx, aa_features in enumerate(combined_features.transpose(0,1)):
                msg = pass_messages(aa_features, self.f, self.use_gpu) # aa_count * output size
                ou = self.g(torch.cat((aa_features, msg), dim=1))
                print(ou.size())
                backbone_atoms_padded[:,idx] = self.g(torch.cat((aa_features, msg), dim=1))

        output_angles, batch_sizes = calculate_dihedral_angles_over_minibatch(backbone_atoms_padded, batch_sizes_backbone, self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes


class soft_to_angle(nn.Module):
    def __init__(self, mixture_size):
        super(soft_to_angle, self).__init__()
        # Omega intializer
        omega_components1 = np.random.uniform(0, 1, int(mixture_size * 0.1))  # Initialize omega 90/10 pos/neg
        omega_components2 = np.random.uniform(2, math.pi, int(mixture_size * 0.9))
        omega_components = np.concatenate((omega_components1, omega_components2))
        np.random.shuffle(omega_components)

        phi_components = np.genfromtxt("data/mixture_model_pfam_"+str(mixture_size)+".txt")[:, 1]
        psi_components = np.genfromtxt("data/mixture_model_pfam_"+str(mixture_size)+".txt")[:, 2]

        self.phi_components = nn.Parameter(torch.from_numpy(phi_components).contiguous().view(-1, 1).float())
        self.psi_components = nn.Parameter(torch.from_numpy(psi_components).contiguous().view(-1, 1).float())
        self.omega_components = nn.Parameter(torch.from_numpy(omega_components).view(-1, 1).float())

    def forward(self, x):
        phi_input_sin = torch.matmul(x, torch.sin(self.phi_components))
        phi_input_cos = torch.matmul(x, torch.cos(self.phi_components))
        psi_input_sin = torch.matmul(x, torch.sin(self.psi_components))
        psi_input_cos = torch.matmul(x, torch.cos(self.psi_components))
        omega_input_sin = torch.matmul(x, torch.sin(self.omega_components))
        omega_input_cos = torch.matmul(x, torch.cos(self.omega_components))

        eps = 10 ** (-4)
        phi = torch.atan2(phi_input_sin, phi_input_cos + eps)
        psi = torch.atan2(psi_input_sin, psi_input_cos + eps)
        omega = torch.atan2(omega_input_sin, omega_input_cos + eps)

        return torch.cat((phi, psi, omega), 2)


"""
Training model
"""

def train_model(data_set_identifier, model, train_loader, validation_loader, learning_rate, minibatch_size=64, eval_interval=50, hide_ui=False, use_gpu=False, minimum_updates=1000):
    set_experiment_id(data_set_identifier, learning_rate, minibatch_size)

    validation_dataset_size = validation_loader.dataset.__len__()

    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sample_num = list()
    train_loss_values = list()
    validation_loss_values = list()

    best_model_loss = 1e20
    best_model_minibatch_time = None
    best_model_path = None
    stopping_condition_met = False
    minibatches_proccesed = 0

    viz = visdom.Visdom()
    loss_box = []
    timestamp = []
    i = 0.0
    win = viz.line(
        X=np.column_stack(np.arange(0, 10)),
        Y=np.column_stack(np.linspace(5, 10, 10)),
    )

    while not stopping_condition_met:
        optimizer.zero_grad()
        model.zero_grad()
        loss_tracker = np.zeros(0)

        for minibatch_id, training_minibatch in enumerate(train_loader, 0):
            minibatches_proccesed += 1
            start_compute_loss = time.time()
            loss = model.compute_loss(training_minibatch)
            loss_box.append(float(loss))
            timestamp.append(i)
            i+=1.0
            #if len(loss_box)==50:
            viz.line(
                    #X=timestamp,
                    #Y=loss_box,
                    X=[i],
                    Y=[float(loss)],
                    win=win,
                    update='append'
                )
            write_out("Train loss:", float(loss))
            start_compute_grad = time.time()
            loss.backward()
            loss_tracker = np.append(loss_tracker, float(loss))
            end = time.time()
            write_out("Loss time:", start_compute_grad-start_compute_loss, "Grad time:", end-start_compute_grad)
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

            # for every eval_interval samples, plot performance on the validation set
            if minibatches_proccesed % eval_interval == 0:

                write_out("Testing model on validation set...")

                train_loss = loss_tracker.mean()
                loss_tracker = np.zeros(0)
                validation_loss, json_data = model.evaluate_model(validation_loader)

                if validation_loss < best_model_loss:
                    best_model_loss = validation_loss
                    best_model_minibatch_time = minibatches_proccesed
                    best_model_path = write_model_to_disk(model)

                write_out("Validation loss:", validation_loss, "Train loss:", train_loss)
                write_out("Best model so far (validation loss): ", validation_loss, "at time", best_model_minibatch_time)
                write_out("Best model stored at " + best_model_path)
                write_out("Minibatches processed:",minibatches_proccesed)
                sample_num.append(minibatches_proccesed)
                train_loss_values.append(train_loss)
                validation_loss_values.append(validation_loss)

                if not hide_ui:
                    json_data["validation_dataset_size"] = validation_dataset_size
                    json_data["sample_num"] = sample_num
                    json_data["train_loss_values"] = train_loss_values
                    json_data["validation_loss_values"] = validation_loss_values
                    res = requests.post('http://localhost:5000/graph', json=json_data)
                    if res.ok:
                        print(res.json())

                if minibatches_proccesed > minimum_updates and minibatches_proccesed > best_model_minibatch_time * 2:
                    stopping_condition_met = True
                    break
    write_result_summary(best_model_loss)
    return best_model_path

"""
Main part
"""

parser = argparse.ArgumentParser()
parser.add_argument('--silent', dest='silent', action='store_true',
                    help='Dont print verbose debug statements.')
parser.add_argument('--hide-ui', dest = 'hide_ui', action = 'store_true',
                    default=False, help='Hide loss graph and visualization UI while training goes on.')
parser.add_argument('--evaluate-on-test', dest = 'evaluate_on_test', action = 'store_true',
                    default=False, help='Run model of test data.')
parser.add_argument('--eval-interval', dest = 'eval_interval', type=int,
                    default=10, help='Evaluate model on validation set every n minibatches.')
parser.add_argument('--min-updates', dest = 'minimum_updates', type=int,
                    default=10000, help='Minimum number of minibatch iterations.')
parser.add_argument('--minibatch-size', dest = 'minibatch_size', type=int,
                    default=1, help='Size of each minibatch.')
parser.add_argument('--learning-rate', dest = 'learning_rate', type=float,
                    default=0.01, help='Learning rate to use during training.')
args, unknown = parser.parse_known_args()


args.hide_ui=True


use_gpu = False
if torch.cuda.is_available():
    write_out("CUDA is available, using GPU")
    use_gpu = True


process_raw_data(use_gpu, force_pre_processing_overwrite=False)

training_file = "data/preprocessed/sample.txt.hdf5"
validation_file = "data/preprocessed/sample.txt.hdf5"
testing_file = "data/preprocessed/testing.hdf5"
print(args.minibatch_size)

model = CNN1Model(21, args.minibatch_size, use_gpu=use_gpu)  # embed size = 21

train_loader = contruct_dataloader_from_disk(training_file, args.minibatch_size)
validation_loader = contruct_dataloader_from_disk(validation_file, args.minibatch_size)

train_model_path = train_model(data_set_identifier="TRAIN",
                               model=model,
                               train_loader=train_loader,
                               validation_loader=validation_loader,
                               learning_rate=args.learning_rate,
                               minibatch_size=args.minibatch_size,
                               eval_interval=args.eval_interval,
                               hide_ui=args.hide_ui,
                               use_gpu=use_gpu,
                               minimum_updates=args.minimum_updates)

print(train_model_path)

