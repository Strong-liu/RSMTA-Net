import numpy as np
import os

fnirs_lag_length = 11 # with t-self


subject_path = r'F:\学生文件夹\刘强\DataSet\WG\window'
subject_list = os.listdir(subject_path)

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        hbo = data['hbo']
        hbr = data['hbr']
        label = data['label']

    # eeg
    eeg_session_dataset = np.expand_dims(eeg, axis=-1)
    eeg_input = eeg_session_dataset.reshape(600, 16, 16, 600, 1)

    # fnirs
    fnirs_session_dataset = np.ones((60, 10, fnirs_lag_length, 16, 16, 30, 2))

    for e in range(60):
        for w in range(10):
            # first 10 windows has same time interval
            hbo_sample = hbo[e, w:(w+fnirs_lag_length),]
            hbr_sample = hbr[e, w:(w+fnirs_lag_length),]

            hbo_sample = np.expand_dims(hbo_sample, axis=-1)
            hbr_sample = np.expand_dims(hbr_sample, axis=-1)

            fnirs_sample = np.concatenate((hbo_sample, hbr_sample), axis=-1)

            fnirs_session_dataset[e, w,] = fnirs_sample

    fnirs_input = fnirs_session_dataset.reshape(600, 11, 16, 16, 30, 2)

    # label
    label_session_dataset = label.T
    label_input = np.repeat(label_session_dataset, repeats=10, axis=0)

    print(eeg_input.shape)
    print(fnirs_input.shape)
    print(label_input.shape)

    save_dict = {
    'eeg':eeg_input,
    'fnirs':fnirs_input,
    'label':label_input
    }

    save_dir = r'F:\学生文件夹\刘强\DataSet\WG\model_input'
    save_name = subject

    np.savez(os.path.join(save_dir,save_name),**save_dict)
    print('\n==============save {} success=============\n'.format(save_name))