import os
import csv
import numpy as np
import scipy.io as sio


from nilearn import connectome


# Input data variables
root_folder = '/data/abide/'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')
phenotype = "/data/abide/tool/Phenotypic_V1_0b_preprocessed1.csv"  


def get_label(subject_list):
    label_dict = {}
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                label = int(row['DX_GROUP'])
                if label == 2:
                    label_dict[row['SUB_ID']] = 0
                else:
                    label_dict[row['SUB_ID']] = 1

    return label_dict


#get all connectivity
#referring to the code https://github.com/sk1712/gcn_metric_learning/blob/master/lib/abide_utils.py
def load_connectivity(subject_list, kind, atlas_name = 'cc400'):
    """

        subject_list : the subject short IDs list

        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation

        atlas_name   : name of the atlas used
    returns:

        all_networks : list of connectivity matrices (regions x regions)
    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject, subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)['connectivity']
        if atlas_name == 'ho':
            matrix = np.delete(matrix, 82, axis=0)
            matrix = np.delete(matrix, 82, axis=1)
        all_networks.append(matrix)
    all_networks=np.array(all_networks)
    return all_networks
                          
def getconn_vector(subject_name0, kind, atlas, label_dict):
    subject_name = np.array(subject_name0)
    data_x = []
    data_y = []
    conn_array = load_connectivity(subject_name, kind, atlas)
    # Get upper diagonal indices
    idx = np.triu_indices_from(conn_array[0], 1)
    # Get vectorised matrices
    vec_networks = [mat[idx] for mat in conn_array]
    # Each subject should be a row of the matrix
    data_x = np.array(vec_networks)
    
    for subname in subject_name:
        data_y.append(int(label_dict[subname]))
    
    data_y = np.array(data_y)
    print("conn vector generator finished")
    return data_x, data_y





