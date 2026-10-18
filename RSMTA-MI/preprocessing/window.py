import numpy as np
import os


# =========================
# 任务配置
# =========================
# 可选: "MI" 或 "MA"
TASK = "MI"

RAW_ROOT = r'F:\学生文件夹\刘强\DataSet\MA+MI'

if TASK == "MI":
    data_root = os.path.join(RAW_ROOT, 'MI')
elif TASK == "MA":
    data_root = os.path.join(RAW_ROOT, 'MA')
else:
    raise ValueError("TASK 只能设置为 'MI' 或 'MA'")


# =========================
# 窗口参数
# =========================
win_step = 1
win_length = 3

# EEG 从 -2s 到 +10s，用 3s 窗口、1s 步长，可以得到 10 个窗口：
# -2~1, -1~2, 0~3, ..., 7~10
eeg_segments_number = 10

# fNIRS 需要为每个 EEG 窗口提供当前窗口 + 后续 10 个延迟窗口。
# EEG 最后一个窗口 w=9 时，需要 fNIRS[9:20]，所以 fNIRS 至少需要 20 个窗口。
# 20 个窗口对应：
# -2~1, -1~2, ..., 17~20
fnirs_segments_number = 20

eeg_srate = 200
fnirs_srate = 10

subject_path = os.path.join(data_root, 'd3')
save_dir = os.path.join(data_root, 'window')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def check_d3_input(subject, eeg, hbo, hbr, label, trial_session):
    """
    检查 d3 阶段输入是否符合 window 切分要求。

    d3 输入应该是：
        eeg: (trials, 16, 16, eeg_timepoints)
        hbo: (trials, 16, 16, fnirs_timepoints)
        hbr: (trials, 16, 16, fnirs_timepoints)
        label: (trials,)
        trial_session: (trials,)
    """
    print(f"\n[{subject}] d3 输入检查:")
    print("EEG:", eeg.shape)
    print("HbO:", hbo.shape)
    print("HbR:", hbr.shape)
    print("Label:", label.shape, "unique:", np.unique(label))
    print("trial_session:", trial_session.shape, "unique:", np.unique(trial_session))

    if eeg.ndim != 4:
        raise ValueError(
            f"{subject}: EEG 应为 4 维 (trials, 16, 16, timepoints)，但得到 {eeg.shape}"
        )

    if hbo.ndim != 4 or hbr.ndim != 4:
        raise ValueError(
            f"{subject}: HbO/HbR 应为 4 维 (trials, 16, 16, timepoints)，"
            f"但得到 HbO={hbo.shape}, HbR={hbr.shape}"
        )

    if eeg.shape[0] != hbo.shape[0] or eeg.shape[0] != hbr.shape[0]:
        raise ValueError(
            f"{subject}: EEG/HbO/HbR trial 数不一致，"
            f"EEG={eeg.shape[0]}, HbO={hbo.shape[0]}, HbR={hbr.shape[0]}"
        )

    if len(label) != eeg.shape[0]:
        raise ValueError(
            f"{subject}: label 数量和 trial 数不一致，"
            f"label={len(label)}, trials={eeg.shape[0]}"
        )

    if len(trial_session) != eeg.shape[0]:
        raise ValueError(
            f"{subject}: trial_session 数量和 trial 数不一致，"
            f"trial_session={len(trial_session)}, trials={eeg.shape[0]}"
        )

    if eeg.shape[1] != 16 or eeg.shape[2] != 16:
        raise ValueError(
            f"{subject}: EEG 空间维度应为 16×16，但得到 {eeg.shape[1:3]}"
        )

    if hbo.shape[1] != 16 or hbo.shape[2] != 16:
        raise ValueError(
            f"{subject}: HbO 空间维度应为 16×16，但得到 {hbo.shape[1:3]}"
        )

    if hbr.shape[1] != 16 or hbr.shape[2] != 16:
        raise ValueError(
            f"{subject}: HbR 空间维度应为 16×16，但得到 {hbr.shape[1:3]}"
        )

    if not np.all(np.isin(label, [0, 1])):
        raise ValueError(
            f"{subject}: label 必须是 0/1，但得到 {np.unique(label)}"
        )


def check_window_boundary(subject, eeg, hbo, hbr):
    """
    检查 EEG/fNIRS 的时间长度是否足够切窗口。

    epoch 阶段：
        EEG: -5s 到 +10s，长度理论上 15s × 200 = 3000 点
        fNIRS: -5s 到 +22s，长度理论上 27s × 10 = 270 点

    window 阶段从真实时间 -2s 开始。
    因为 epoch 起点是 -5s，所以 -2s 对应 epoch 内第 3s：
        start = 3 * sampling_rate

    EEG 最后一个窗口：
        start = (3 + 9) * 200 = 2400
        end = 2400 + 600 = 3000

    fNIRS 最后一个窗口：
        start = (3 + 19) * 10 = 220
        end = 220 + 30 = 250
    """
    eeg_required_end = (3 + (eeg_segments_number - 1)) * eeg_srate + win_length * eeg_srate
    fnirs_required_end = (3 + (fnirs_segments_number - 1)) * fnirs_srate + win_length * fnirs_srate

    eeg_timepoints = eeg.shape[-1]
    hbo_timepoints = hbo.shape[-1]
    hbr_timepoints = hbr.shape[-1]

    print("\n[窗口边界检查]")
    print("EEG required end:", eeg_required_end, "EEG timepoints:", eeg_timepoints)
    print("fNIRS required end:", fnirs_required_end, "HbO timepoints:", hbo_timepoints, "HbR timepoints:", hbr_timepoints)

    if eeg_timepoints < eeg_required_end:
        raise ValueError(
            f"{subject}: EEG 时间点不足，至少需要 {eeg_required_end}，但只有 {eeg_timepoints}"
        )

    if hbo_timepoints < fnirs_required_end:
        raise ValueError(
            f"{subject}: HbO 时间点不足，至少需要 {fnirs_required_end}，但只有 {hbo_timepoints}"
        )

    if hbr_timepoints < fnirs_required_end:
        raise ValueError(
            f"{subject}: HbR 时间点不足，至少需要 {fnirs_required_end}，但只有 {hbr_timepoints}"
        )


# =========================
# 主流程
# =========================
subject_list = sorted([f for f in os.listdir(subject_path) if f.endswith('.npz')])

print("\n==============================")
print(f"当前任务: {TASK}")
print(f"d3 路径: {subject_path}")
print(f"window 保存路径: {save_dir}")
print("==============================\n")

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject), allow_pickle=True) as data:
        eeg = data['eeg']
        hbo = data['hbo']
        hbr = data['hbr']
        label = data['label']

        if 'trial_session' in data.files:
            trial_session = data['trial_session']
        else:
            raise KeyError(
                f"{subject} 中没有 trial_session。"
                f"请先使用新版 to3d.py 保存 trial_session。"
            )

        if 'common_selection' in data.files:
            common_selection = data['common_selection']
        else:
            common_selection = None

    label = np.asarray(label).squeeze().astype(np.int64)
    trial_session = np.asarray(trial_session).squeeze().astype(np.int64)

    check_d3_input(
        subject=subject,
        eeg=eeg,
        hbo=hbo,
        hbr=hbr,
        label=label,
        trial_session=trial_session
    )

    check_window_boundary(
        subject=subject,
        eeg=eeg,
        hbo=hbo,
        hbr=hbr
    )

    # 动态获取清洗后剩余的有效 trial 数
    n_trials = eeg.shape[0]

    eeg_window = np.zeros(
        (
            n_trials,
            eeg_segments_number,
            16,
            16,
            win_length * eeg_srate
        ),
        dtype=eeg.dtype
    )

    hbo_window = np.zeros(
        (
            n_trials,
            fnirs_segments_number,
            16,
            16,
            win_length * fnirs_srate
        ),
        dtype=hbo.dtype
    )

    hbr_window = np.zeros(
        (
            n_trials,
            fnirs_segments_number,
            16,
            16,
            win_length * fnirs_srate
        ),
        dtype=hbr.dtype
    )

    for trial_idx in range(n_trials):
        # =========================
        # EEG windows
        # =========================
        for w in range(eeg_segments_number):
            # epoch 起点是 -5s，目标窗口从 -2s 开始
            # 所以 -2s 对应 epoch 内 3s 处
            eeg_start_index = (3 + w * win_step) * eeg_srate
            eeg_end_index = eeg_start_index + win_length * eeg_srate

            eeg_segment = eeg[
                trial_idx,
                :,
                :,
                eeg_start_index:eeg_end_index
            ]

            eeg_window[trial_idx, w, :, :, :] = eeg_segment

        # =========================
        # fNIRS windows
        # =========================
        for fw in range(fnirs_segments_number):
            # fNIRS 同样从真实时间 -2s 开始切
            fnirs_start_index = (3 + fw * win_step) * fnirs_srate
            fnirs_end_index = fnirs_start_index + win_length * fnirs_srate

            hbo_segment = hbo[
                trial_idx,
                :,
                :,
                fnirs_start_index:fnirs_end_index
            ]

            hbr_segment = hbr[
                trial_idx,
                :,
                :,
                fnirs_start_index:fnirs_end_index
            ]

            hbo_window[trial_idx, fw, :, :, :] = hbo_segment
            hbr_window[trial_idx, fw, :, :, :] = hbr_segment

    print(f"\n{subject} window 输出形状:")
    print("EEG:", eeg_window.shape)
    print("HbO:", hbo_window.shape)
    print("HbR:", hbr_window.shape)
    print("Label:", label.shape, "unique:", np.unique(label), "count:", np.bincount(label))
    print("trial_session:", trial_session.shape, "unique:", np.unique(trial_session))

    for sess in [0, 1, 2]:
        sess_count = np.sum(trial_session == sess)
        print(f"session {sess} trial 数: {sess_count}")

    save_dict = {
        'eeg': eeg_window,
        'hbo': hbo_window,
        'hbr': hbr_window,
        'label': label,
        'trial_session': trial_session
    }

    if common_selection is not None:
        save_dict['common_selection'] = common_selection

    save_path = os.path.join(save_dir, subject)

    np.savez(
        save_path,
        **save_dict
    )

    print(f"============== save {save_path} success ==============\n")