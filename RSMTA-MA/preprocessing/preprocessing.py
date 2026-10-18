import numpy as np
import os
import mne
import scipy.io as io
from mne_icalabel import label_components
import matplotlib.pyplot as plt

# =========================
# 任务配置
# =========================
# 可选: "MI" 或 "MA"
TASK = "MA"

RAW_ROOT = r'F:\学生文件夹\刘强\DataSet\MA+MI'

if TASK == "MI":
    data_root = os.path.join(RAW_ROOT, 'MI')
elif TASK == "MA":
    data_root = os.path.join(RAW_ROOT, 'MA')
else:
    raise ValueError("TASK 只能设置为 'MI' 或 'MA'")

mat2array_dir = os.path.join(data_root, 'mat2array')
save_dir = os.path.join(data_root, 'preprocessed')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# =========================
# EEG / fNIRS 通道名称
# =========================
# 30 个真正用于模型训练的 EEG 通道
eeg_chn_names = [
    'F7', 'AFF5h', 'F3', 'AFp1', 'AFp2', 'AFF6h', 'F4', 'F8',
    'AFF1h', 'AFF2h', 'Cz', 'Pz', 'FCC5h', 'FCC3h', 'CCP5h',
    'CCP3h', 'T7', 'P7', 'P3', 'PPO1h', 'POO1', 'POO2',
    'PPO2h', 'P4', 'FCC4h', 'FCC6h', 'CCP4h', 'CCP6h', 'P8', 'T8'
]

# 36 个真实 fNIRS 解剖学通道名称
fnirs_chn_names = [
    'AF7Fp1', 'AF3Fp1', 'AF3AFz', 'FpzFp1', 'FpzAFz', 'FpzFp2',
    'AF4AFz', 'AF4Fp2', 'AF8Fp2', 'OzPOz', 'OzO1', 'OzO2',
    'C5CP5', 'C5FC5', 'C5C3', 'FC3FC5', 'FC3C3', 'FC3FC1',
    'CP3CP5', 'CP3C3', 'CP3CP1', 'C1C3', 'C1FC1', 'C1CP1',
    'C2FC2', 'C2CP2', 'C2C4', 'FC4FC2', 'FC4C4', 'FC4FC6',
    'CP4CP6', 'CP4CP2', 'CP4C4', 'C6CP6', 'C6C4', 'C6FC6'
]


# =========================
# 通用工具函数
# =========================
def matlab_cellstr_to_list(cell_array):
    result = []
    flat = np.ravel(cell_array)
    for item in flat:
        if isinstance(item, np.ndarray):
            if item.size == 0:
                result.append("")
            else:
                value = item.squeeze()
                if isinstance(value, np.ndarray):
                    value = value.flat[0]
                result.append(str(value))
        else:
            result.append(str(item))
    return result


def convert_pos_to_meter(pos):
    pos = np.asarray(pos, dtype=float)
    max_abs = np.max(np.abs(pos))
    if max_abs > 10:
        pos = pos / 1000.0
    elif max_abs > 1:
        pos = pos / 100.0
    return pos


def load_nirs_mnt_info(mnt_path):
    if not os.path.exists(mnt_path):
        raise FileNotFoundError(f"找不到 NIRS mnt.mat: {mnt_path}")
    mnt = io.loadmat(mnt_path)
    mnt_struct = mnt['mnt'][0, 0]
    anat_names = matlab_cellstr_to_list(mnt_struct['clab'])
    sd = np.asarray(mnt_struct['sd']).astype(int)
    source_struct = mnt_struct['source'][0, 0]
    detector_struct = mnt_struct['detector'][0, 0]
    source_pos = np.asarray(source_struct['pos_3d']).T
    detector_pos = np.asarray(detector_struct['pos_3d']).T
    source_pos = convert_pos_to_meter(source_pos)
    detector_pos = convert_pos_to_meter(detector_pos)

    if len(anat_names) != 36:
        raise ValueError(f"{mnt_path}: NIRS 解剖学通道数应为 36，但得到 {len(anat_names)}。")
    if sd.shape != (36, 2):
        raise ValueError(f"{mnt_path}: sd 应为 shape=(36, 2)，但得到 {sd.shape}")

    return anat_names, sd, source_pos, detector_pos


def build_fnirs_cw_names_from_mnt(mnt_path):
    anat_names, sd, _, _ = load_nirs_mnt_info(mnt_path)
    fnirs_cw_names = []
    for s_id, d_id in sd:
        fnirs_cw_names.append(f"S{s_id}_D{d_id} 760")
        fnirs_cw_names.append(f"S{s_id}_D{d_id} 850")

    if len(fnirs_cw_names) != 72:
        raise ValueError(f"{mnt_path}: 自动生成的 MNE-NIRS 通道数应为 72，但得到 {len(fnirs_cw_names)}")
    return fnirs_cw_names, anat_names, sd


def create_fnirs_montage_from_mnt(mnt_path):
    _, _, source_pos, detector_pos = load_nirs_mnt_info(mnt_path)
    ch_pos = {}
    for i, pos in enumerate(source_pos):
        ch_pos[f"S{i + 1}"] = pos
    for i, pos in enumerate(detector_pos):
        ch_pos[f"D{i + 1}"] = pos

    nasion = np.array([0.4e-03, 85.9e-03, -47.6e-03])
    lpa = np.array([-83.8e-03, -18.6e-03, -57.2e-03])
    rpa = np.array([83.9e-03, -16.6e-03, -56.7e-03])

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa)
    return montage


# =========================
# ICA 工作流
# =========================
def auto_clean_ica_workflow(raw, n_components=20, prob_threshold=0.80):
    print("1. 开始运行 ICA...")

    filt_ica = raw.copy().filter(l_freq=1.0, h_freq=None)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        max_iter='auto',
        random_state=97
    )

    # [核心修改 1]: 加入 reject_by_annotation=True，让 ICA 避开你手动框选的坏段
    ica.fit(filt_ica, reject_by_annotation=True)

    print("\n2. 调用 mne-icalabel 自动打分模型...")
    ic_labels = label_components(filt_ica, ica, method='iclabel')
    predicted_labels = ic_labels["labels"]
    predicted_probs = ic_labels["y_pred_proba"]

    print("\n3. 分析成分概率...")
    artifact_classes = ['eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise']
    auto_exclude_idx = []
    manual_components_idx = []
    manual_probs = []
    manual_labels = []

    print("-" * 70)
    for comp_idx, (pred_label, prob) in enumerate(zip(predicted_labels, predicted_probs)):
        pred_label = str(pred_label).lower()
        prob = float(prob)
        print(f"成分 {comp_idx:02d}: {pred_label:18s} 概率: {prob:.4f}")

        is_auto_exclude = False
        if pred_label in artifact_classes and prob >= prob_threshold:
            auto_exclude_idx.append(comp_idx)
            is_auto_exclude = True

        if pred_label != 'brain' and not is_auto_exclude:
            manual_components_idx.append(comp_idx)
            manual_probs.append(prob)
            manual_labels.append(pred_label)
    print("-" * 70)

    ica.exclude = auto_exclude_idx

    print(f"\n=> 自动建议剔除概率 >= {prob_threshold * 100:.1f}% 的明确伪影成分:")
    print(f"=> {auto_exclude_idx}")
    print("\n[提醒] other 不会被自动剔除，需要人工判断。")

    print("\n4. 准备人工把关疑似成分...")
    if not manual_components_idx:
        print("未检测到需要人工把关的非 brain 成分。")
    else:
        print("\n以下成分建议人工核查：")
        for idx, prob, lbl in zip(manual_components_idx, manual_probs, manual_labels):
            print(f" -> 成分 {idx:02d}: {lbl}，概率={prob:.4f}")

        fig_topo = ica.plot_components(picks=manual_components_idx, show=False)
        if hasattr(fig_topo, '__iter__'):
            for fig in fig_topo:
                fig.suptitle('Manual Check: Suspected Components Topology')
        else:
            fig_topo.suptitle('Manual Check: Suspected Components Topology')

        fig_sources = ica.plot_sources(raw, picks=manual_components_idx, show=False)
        fig_sources.canvas.manager.set_window_title('Manual Check: Suspected Components Sources')
        plt.show()

    print("\n5. ICA 人工把关结束。")
    print("最终 ICA exclude 列表:", ica.exclude)

    return ica


# =========================
# 数据读取与检查
# =========================
def load_subject_npz(raw_data_path):
    with np.load(raw_data_path, allow_pickle=True) as data:
        eeg = np.asarray(data['eeg'])
        eeg_time = np.asarray(data['eeg_time']).squeeze()
        fnirs_time = np.asarray(data['fnirs_time']).squeeze()
        label = np.asarray(data['label']).squeeze().astype(np.int64)

        if 'trial_session' in data.files:
            trial_session = np.asarray(data['trial_session']).squeeze().astype(np.int64)
        else:
            trial_session = np.repeat(np.arange(3), 20).astype(np.int64)

        if 'low_wl' in data.files and 'high_wl' in data.files:
            low_wl = np.asarray(data['low_wl'])
            high_wl = np.asarray(data['high_wl'])
        else:
            low_wl = np.asarray(data['hbo'])
            high_wl = np.asarray(data['hbr'])

    return eeg, low_wl, high_wl, eeg_time, fnirs_time, label, trial_session


def check_loaded_data(subject_no, eeg, low_wl, high_wl, eeg_time, fnirs_time, label, trial_session):
    if eeg.shape[0] != 32:
        raise ValueError(f"{subject_no}: EEG 应为 32 通道，包括 30 EEG + VEOG + HEOG，但得到 {eeg.shape[0]}")
    if low_wl.shape[0] != 36 or high_wl.shape[0] != 36:
        raise ValueError(f"{subject_no}: NIRS low/high 应为 36 通道，但得到 {low_wl.shape}, {high_wl.shape}")
    if not np.all(np.isin(label, [0, 1])):
        raise ValueError(f"{subject_no}: label 必须是 0/1，但得到 {np.unique(label)}")
    if len(eeg_time) != len(label):
        raise ValueError(f"{subject_no}: eeg_time 数量和 label 不一致")
    if len(fnirs_time) != len(label):
        raise ValueError(f"{subject_no}: fnirs_time 数量和 label 不一致")
    if len(trial_session) != len(label):
        raise ValueError(f"{subject_no}: trial_session 数量和 label 不一致")


def make_events(time_ms, label):
    events = np.zeros((len(time_ms), 3), dtype=int)
    events[:, 0] = np.round(time_ms / 1000.0 * 200).astype(int)
    events[:, 2] = label.astype(int) + 1
    return events


def make_fnirs_events(time_ms, label):
    events = np.zeros((len(time_ms), 3), dtype=int)
    events[:, 0] = np.round(time_ms / 1000.0 * 10).astype(int)
    events[:, 2] = label.astype(int) + 1
    return events


# =========================
# fNIRS 处理
# =========================
def create_fnirs_raw(low_wl_data, high_wl_data, mnt_path):
    fnirs_cw_names, anat_names_from_mnt, sd = build_fnirs_cw_names_from_mnt(mnt_path)
    if list(anat_names_from_mnt) != list(fnirs_chn_names):
        print("[Warning] 当前 mnt.clab 与代码中的 fnirs_chn_names 不完全一致。后续将优先使用 mnt.clab。")
    if low_wl_data.shape[0] != 36 or high_wl_data.shape[0] != 36:
        raise ValueError("low_wl/high_wl 应为 36 通道")
    if low_wl_data.shape[1] != high_wl_data.shape[1]:
        raise ValueError("low_wl/high_wl 时间点不一致")

    fnirs_data_interleaved = np.zeros((72, low_wl_data.shape[1]), dtype=low_wl_data.dtype)
    fnirs_data_interleaved[0::2, :] = low_wl_data
    fnirs_data_interleaved[1::2, :] = high_wl_data

    fnirs_info = mne.create_info(ch_names=fnirs_cw_names, sfreq=10.0, ch_types='fnirs_cw_amplitude')
    for ch_idx, one_chn in enumerate(fnirs_info["chs"]):
        if ch_idx % 2 == 0:
            one_chn["loc"][9] = 760.0
        else:
            one_chn["loc"][9] = 850.0

    fnirs_montage = create_fnirs_montage_from_mnt(mnt_path)
    fnirs_info.set_montage(fnirs_montage)
    fnirs_raw = mne.io.RawArray(data=fnirs_data_interleaved, info=fnirs_info)
    return fnirs_raw, fnirs_cw_names, anat_names_from_mnt


def preprocess_fnirs(fnirs_raw, fnirs_cw_names, fnirs_anat_names):
    fnirs_raw_od = mne.preprocessing.nirs.optical_density(fnirs_raw)
    fnirs_raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(fnirs_raw_od)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(fnirs_raw_od, ppf=6.0)
    raw_haemo.filter(l_freq=0.01, h_freq=0.1, method='iir', iir_params=dict(order=6, ftype='butter'))

    raw_hbo = raw_haemo.copy().pick(picks='hbo')
    raw_hbr = raw_haemo.copy().pick(picks='hbr')

    std_nirs_names = [ch.replace(' 760', '') for ch in fnirs_cw_names[::2]]
    rename_hbo_map = {f"{std} hbo": anat for std, anat in zip(std_nirs_names, fnirs_anat_names)}
    rename_hbr_map = {f"{std} hbr": anat for std, anat in zip(std_nirs_names, fnirs_anat_names)}

    raw_hbo.rename_channels(rename_hbo_map)
    raw_hbr.rename_channels(rename_hbr_map)
    raw_hbo.set_channel_types({ch: 'hbo' for ch in raw_hbo.ch_names})
    raw_hbr.set_channel_types({ch: 'hbr' for ch in raw_hbr.ch_names})

    return raw_hbo, raw_hbr


# =========================
# 主循环
# =========================
sub_ids = [16]  # 单独处理被试
# sub_ids = range(1,30)

for sub_id in sub_ids:
    subject_no = f'subject {sub_id:02d}'

    print(f"\n{'=' * 60}")
    print(f"正在处理: {subject_no} | TASK={TASK}")
    print(f"{'=' * 60}")

    raw_data_path = os.path.join(mat2array_dir, f'{subject_no}.npz')
    if not os.path.exists(raw_data_path):
        print(f"找不到 {subject_no} 的 mat2array 文件，已跳过。")
        continue

    fnirs_mnt_path = os.path.join(RAW_ROOT, 'NIRS_01-29', subject_no, 'mnt.mat')
    if not os.path.exists(fnirs_mnt_path):
        print(f"找不到 {subject_no} 的 NIRS mnt.mat，已跳过。")
        continue

    eeg, low_wl, high_wl, eeg_time, fnirs_time, label, trial_session = load_subject_npz(raw_data_path)

    check_loaded_data(
        subject_no=subject_no, eeg=eeg, low_wl=low_wl, high_wl=high_wl,
        eeg_time=eeg_time, fnirs_time=fnirs_time, label=label, trial_session=trial_session
    )

    # =========================
    # EEG preprocessing
    # =========================
    eeg_data_30 = eeg[:30, :]

    eeg_info = mne.create_info(ch_names=eeg_chn_names, sfreq=200, ch_types='eeg')
    eeg_info.set_montage('standard_1005')
    raw_eeg = mne.io.RawArray(data=eeg_data_30, info=eeg_info)

    events = make_events(eeg_time, label)
    fnirs_events = make_fnirs_events(fnirs_time, label)

    # 1. 基础滤波
    raw_eeg.notch_filter(freqs=50)
    raw_eeg.filter(l_freq=0.5, h_freq=50.0, method='iir', iir_params=dict(order=6, ftype='butter'))

    # [核心修改 2]: 将坏段标记、坏通道标记与插值，提前到平均参考和 ICA 之前
    print(f"\n[交互提示] {subject_no}：准备进行坏通道与坏段排查。")
    print("操作指南：")
    print(" - 点击通道名称：将其标记为坏通道 (名称变灰)。")
    print(" - 按 'a' 键进入 Annotation 模式：鼠标拖拽框选极其离谱的突发伪影段。")
    print(" - 关闭画图窗口后，程序将自动插值并继续。")

    raw_eeg.plot(block=True, scalings='auto', title=f"{subject_no} - Mark Bad Channels & Segments")

    # [核心修改 3]: 执行坏通道插值
    if raw_eeg.info['bads']:
        print(f"\n检测到坏通道被标记: {raw_eeg.info['bads']}，正在进行球面插值修复...")
        raw_eeg.interpolate_bads(reset_bads=True)
    else:
        print("\n未标记坏通道，跳过插值。")

    # [核心修改 4]: 在插值修复后，再进行平均参考，防止噪声扩散
    print("正在进行平均参考...")
    raw_eeg.set_eeg_reference(ref_channels='average')

    print(f"\n[交互提示] {subject_no} 正在执行 ICA + ICLabel...")

    # 为避免 ICLabel 对特殊通道名解析异常，临时清理通道名
    rename_map = {}
    restore_map = {}
    for ch in raw_eeg.ch_names:
        clean_ch = ch.replace('p', '').replace('h', '')
        if clean_ch != ch:
            rename_map[ch] = clean_ch
            restore_map[clean_ch] = ch

    if rename_map:
        raw_eeg.rename_channels(rename_map)

    # 运行 ICA (此时会自动带着 reject_by_annotation 跑，避开你刚画的红框区域)
    ica = auto_clean_ica_workflow(
        raw=raw_eeg,
        n_components=20,
        prob_threshold=0.80
    )
    ica.apply(raw_eeg)

    if restore_map:
        raw_eeg.rename_channels(restore_map)

    # 保存清洗后的 EEG 连续数据
    raw_eeg.save(os.path.join(save_dir, f'{subject_no}_eeg_raw.fif'), overwrite=True)

    # =========================
    # fNIRS preprocessing
    # =========================
    print(f"\n[fNIRS] {subject_no} 正在执行 OD -> TDDR -> MBLL -> HbO/HbR...")
    fnirs_raw, fnirs_cw_names, fnirs_anat_names = create_fnirs_raw(
        low_wl_data=low_wl, high_wl_data=high_wl, mnt_path=fnirs_mnt_path
    )

    raw_hbo, raw_hbr = preprocess_fnirs(
        fnirs_raw=fnirs_raw, fnirs_cw_names=fnirs_cw_names, fnirs_anat_names=fnirs_anat_names
    )

    raw_hbo.save(os.path.join(save_dir, f'{subject_no}_hbo_raw.fif'), overwrite=True)
    raw_hbr.save(os.path.join(save_dir, f'{subject_no}_hbr_raw.fif'), overwrite=True)

    # =========================
    # 保存事件和 session 信息
    # =========================
    np.save(os.path.join(save_dir, f'{subject_no}_events.npy'), events)
    np.save(os.path.join(save_dir, f'{subject_no}_fnirs_events.npy'), fnirs_events)
    np.save(os.path.join(save_dir, f'{subject_no}_trial_session.npy'), trial_session)

    print(f"\n============== {subject_no} preprocessing save success ==============\n")