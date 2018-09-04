import numpy as np
import librosa
from Utils.utils import *
from config import *


def preprocess(dataset_A, dataset_B, sr = sr, n_features=n_features, frame_period = frame_period ) :
    
    print("Constructing MCEPs....")
    dataset_A = load_wavs(dataset_A, sr = sr)
    dataset_B = load_wavs(dataset_B, sr = sr)
    
    f0s_A, _, _, _, coded_sps_A = world_encode_data(wavs = dataset_A, fs = sr, frame_period = frame_period, coded_dim = n_features)
    f0s_B, _, _, _, coded_sps_B = world_encode_data(wavs = dataset_B, fs = sr, frame_period = frame_period, coded_dim = n_features)

    coded_sps_A_T = transpose_in_list(coded_sps_A)
    coded_sps_B_T = transpose_in_list(coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(coded_sps_A_T)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(coded_sps_B_T)
        
    if not os.path.exists(os.path.join("./data","mcep.npz")) :
        np.savez(os.path.join("./data",'mcep.npz'),
                 A_mean = coded_sps_A_mean, A_std = coded_sps_A_std,
                 B_mean = coded_sps_B_mean, B_std = coded_sps_B_std)

    print("Constructing Log_f0s....")
    if not os.path.exists(os.path.join("./data","logf0s.npz")) :
        logf0s_A_mean, logf0s_A_std = logf0_statistics(f0s_A)
        logf0s_B_mean, logf0s_B_std = logf0_statistics(f0s_B)
        np.savez(os.path.join("./data","logf0s.npz"),
                 A_mean = logf0s_A_mean, A_std = logf0s_A_std, 
                 B_mean = logf0s_B_mean, B_std = logf0s_B_std)
    print("Preprocessing Done!!!")    

    return coded_sps_A_norm, coded_sps_B_norm
        
