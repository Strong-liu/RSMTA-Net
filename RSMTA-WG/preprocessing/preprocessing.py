import numpy as np
import os
import mne
import matplotlib.pyplot as plt

# eeg info
eeg_chn_names = [
    'Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'Cz',
    'CP5', 'CP1', 'P7', 'P3', 'Pz', 'POz', 'O1', 'Fp2',
    'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 'CP2', 'CP6',
    'P4', 'P8', 'O2'
]

eeg_info = mne.create_info(
    ch_names=eeg_chn_names,
    sfreq=200,
    ch_types='eeg'
)
eeg_info.set_montage('standard_1005')

# fnirs info
fnirs_chn_names = [
    'AF7', 'AFF5', 'AFp7', 'AF5h', 'AFp3', 'AFF3h', 'AF1', 'AFFz',
    'AFpz', 'AF2', 'AFp4', 'FCC3', 'C3h', 'C5h', 'CCP3', 'CPP3',
    'P3h', 'P5h', 'PPO3', 'AFF4h', 'AF6h', 'AFF6', 'AFp8', 'AF8',
    'FCC4', 'C6h', 'C4h', 'CCP4', 'CPP4', 'P6h', 'P4h', 'PPO4',
    'PPOz', 'PO1', 'PO2', 'POOz'
]

fnirs_info = mne.create_info(
    ch_names=fnirs_chn_names,
    sfreq=10,
    ch_types='eeg'
)
fnirs_info.set_montage('standard_1005')

# begin preprocess
subject_no = 'VP001'

with np.load(
    r'F:\学生文件夹\刘强\DataSet\WG\mat2array\{}.npz'.format(subject_no)
) as data:
    eeg = data['eeg']
    eeg_time = data['eeg_time']
    hbo = data['hbo']
    hbr = data['hbr']
    fnirs_time = data['fnirs_time']
    label = data['label']


# =========================
# EEG preprocessing
# =========================

raw = mne.io.RawArray(
    data=eeg[:-2, :],
    info=eeg_info
)

# notch filter
raw_notch = raw.notch_filter(
    freqs=np.arange(50, 100, 50)
)

# band-pass filter
raw_filtered = raw_notch.filter(
    l_freq=0.5,
    h_freq=50.,
    method='iir',
    iir_params=dict(order=6, ftype='butter')
)

# average reference
raw_avg_ref = raw_filtered.set_eeg_reference(
    ref_channels="average"
)

raw_avg_ref.load_data()

# filtering just for ICA
filt_ica_raw = raw_avg_ref.copy().filter(
    l_freq=1.,
    h_freq=None
)

# ICA
ica = mne.preprocessing.ICA(
    n_components=20,
    random_state=97,
    max_iter='auto'
)

ica.fit(filt_ica_raw)

# ======================================================
# 手动点击选择需要删除的 ICA 成分
#
# 第一步：
#   弹出 ICA components 空间拓扑图
#   点击你认为是伪迹的成分，成分会变灰 / 标记为 exclude
#
# 第二步：
#   弹出 ICA sources 时间序列图
#   可以继续根据时间序列点击成分名称进行标记
#
# 关闭窗口后，代码会自动读取 ica.exclude
# 不需要再手动输入数字
# ======================================================

# 查看 ICA 空间拓扑图
ica.plot_components()
plt.show(block=True)

# 查看 ICA 时间序列
ica.plot_sources(raw_avg_ref)
plt.show(block=True)

print("Selected ICA components to exclude:", ica.exclude)

# apply ICA
raw_icaed = ica.apply(raw_avg_ref.copy())

eeg_processed = raw_icaed.get_data()


# =========================
# fNIRS preprocessing
# =========================

hbo_raw = mne.io.RawArray(
    data=hbo,
    info=fnirs_info
)

hbr_raw = mne.io.RawArray(
    data=hbr,
    info=fnirs_info
)

hbo_filtered = hbo_raw.filter(
    l_freq=0.01,
    h_freq=0.1,
    method='iir',
    iir_params=dict(order=6, ftype='butter')
)

hbr_filtered = hbr_raw.filter(
    l_freq=0.01,
    h_freq=0.1,
    method='iir',
    iir_params=dict(order=6, ftype='butter')
)

hbo_processed = hbo_filtered.get_data()
hbr_processed = hbr_filtered.get_data()


# =========================
# Save data
# =========================

save_dict = {
    'eeg': eeg_processed,
    'eeg_time': eeg_time,
    'hbo': hbo_processed,
    'hbr': hbr_processed,
    'fnirs_time': fnirs_time,
    'label': label
}

save_dir = r'F:\学生文件夹\刘强\DataSet\WG\preprocessed'

save_name = subject_no

np.savez(
    os.path.join(save_dir, save_name),
    **save_dict
)

print('\n============== save {} success =============\n'.format(save_name))