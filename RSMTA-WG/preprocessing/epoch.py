import numpy as np
import os
import mne

task_period = 10

eeg_sample_rate = 200
eeg_pre_onset = 5
eeg_post_onset = task_period

fnirs_sample_rate = 10
fnirs_pre_onset = 5
fnirs_post_onset = task_period + 12

fnirs_chn_names = ['AF7','AFF5','AFp7','AF5h','AFp3','AFF3h','AF1','AFFz','AFpz','AF2','AFp4','FCC3','C3h','C5h','CCP3','CPP3','P3h','P5h','PPO3','AFF4h','AF6h','AFF6','AFp8','AF8','FCC4','C6h','C4h','CCP4','CPP4','P6h','P4h','PPO4','PPOz','PO1','PO2','POOz']
fnirs_info = mne.create_info(ch_names=fnirs_chn_names, sfreq=10, ch_types='eeg')
fnirs_info.set_montage('standard_1005')

subject_path = r'F:\学生文件夹\刘强\DataSet\WG\preprocessed'
subject_list = os.listdir(subject_path)

for subject in subject_list:
    with np.load(os.path.join(subject_path,subject)) as data:
        eeg = data['eeg']
        eeg_time = data['eeg_time']
        hbo = data['hbo']
        hbr = data['hbr']
        fnirs_time = data['fnirs_time']
        label = data['label']

    #epoch
    eeg_epoch = np.ones((60, 28, eeg_sample_rate*(eeg_pre_onset+eeg_post_onset)), dtype=np.float64)
    hbo_epoch = np.ones((60, 36, fnirs_sample_rate*(fnirs_pre_onset+fnirs_post_onset)), dtype=np.float64)
    hbr_epoch = np.ones((60, 36, fnirs_sample_rate*(fnirs_pre_onset+fnirs_post_onset)), dtype=np.float64)

    for t in range(60):
        #eeg
        eeg_start_indice = int((eeg_time[0, t]/1000.-eeg_pre_onset)*eeg_sample_rate)
        eeg_end_indice = int(eeg_start_indice + (eeg_pre_onset+eeg_post_onset)*eeg_sample_rate)

        eeg_one_epoch = eeg[:, eeg_start_indice:eeg_end_indice]
        eeg_epoch[t,] = eeg_one_epoch

        #fnirs
        fnirs_start_indice = int((fnirs_time[0, t]/1000.-fnirs_pre_onset)*fnirs_sample_rate)
        fnirs_end_indice = int(fnirs_start_indice + (fnirs_pre_onset+fnirs_post_onset)*fnirs_sample_rate)

        hbo_one_epoch = hbo[:, fnirs_start_indice:fnirs_end_indice]
        hbr_one_epoch = hbr[:, fnirs_start_indice:fnirs_end_indice]
        hbo_epoch[t,] = hbo_one_epoch
        hbr_epoch[t,] = hbr_one_epoch

    #fnirs baseline correction
    hbo_raw_bc = mne.EpochsArray(data=hbo_epoch, info=fnirs_info, baseline=(None, 3.))
    hbr_raw_bc = mne.EpochsArray(data=hbr_epoch, info=fnirs_info, baseline=(None, 3.))  
    hbo_epoch_bc = hbo_raw_bc.get_data()
    hbr_epoch_bc = hbr_raw_bc.get_data()

    print(eeg_epoch.shape)
    print(hbo_epoch_bc.shape)
    print(hbr_epoch_bc.shape)
    print(label.shape)

    save_dict = {
        'eeg':eeg_epoch,
        'hbo':hbo_epoch_bc,
        'hbr':hbr_epoch_bc,
        'label':label
    }

    save_dir = r'F:\学生文件夹\刘强\DataSet\WG\epoch'
    save_name = subject

    np.savez(os.path.join(save_dir,save_name),**save_dict)
    print('\n==============save {} success=============\n'.format(save_name))


