import numpy as np
import os


win_step = 1
win_length = 3

eeg_segments_number = 10
fnirs_segments_number = 22

eeg_srate = 200
fnirs_srate = 10

subject_path = r'F:\学生文件夹\刘强\DataSet\WG\d3'
subject_list = os.listdir(subject_path)

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        hbo = data['hbo']
        hbr = data['hbr']
        label = data['label']

    eeg_window = np.ones((60, eeg_segments_number, 16, 16, win_length*eeg_srate))
    hbo_window = np.ones((60, fnirs_segments_number, 16, 16, win_length*fnirs_srate))
    hbr_window = np.ones((60, fnirs_segments_number, 16, 16, win_length*fnirs_srate))

    for e in range(60):
        # first 10 windows has same time interval
        for w in range(eeg_segments_number):
            eeg_start_indice = (3+w)*eeg_srate
            eeg_end_indice = eeg_start_indice + win_length*eeg_srate

            eeg_segment = eeg[e, :, :, eeg_start_indice:eeg_end_indice]

            eeg_window[e, w, :, :, :] = eeg_segment

        for fw in range(fnirs_segments_number):
            fnirs_start_indice = (3+fw)*fnirs_srate
            fnirs_end_indice = fnirs_start_indice + win_length*fnirs_srate

            hbo_segment = hbo[e, :, :, fnirs_start_indice:fnirs_end_indice]
            hbr_segment = hbr[e, :, :, fnirs_start_indice:fnirs_end_indice]

            hbo_window[e, fw, :, :, :] = hbo_segment
            hbr_window[e, fw, :, :, :] = hbr_segment

    print(eeg_window.shape)
    print(hbo_window.shape)
    print(hbr_window.shape)
    print(label.shape)

    save_dict = {
    'eeg':eeg_window,
    'hbo':hbo_window,
    'hbr':hbr_window,
    'label':label
    }

    save_dir = r'F:\学生文件夹\刘强\DataSet\WG\window'
    save_name = subject

    np.savez(os.path.join(save_dir,save_name),**save_dict)
    print('\n==============save {} success=============\n'.format(save_name))
