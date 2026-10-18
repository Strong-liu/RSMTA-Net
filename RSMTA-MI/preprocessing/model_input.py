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
# 参数配置
# =========================
fnirs_lag_length = 11  # 当前 fNIRS 窗口 + 后续 10 个延迟窗口

eeg_segments_number = 10
fnirs_required_segments = eeg_segments_number + fnirs_lag_length - 1
# 10 + 11 - 1 = 20
# 因为最后一个 EEG 窗口 w=9 时，需要 fNIRS[9:20]

subject_path = os.path.join(data_root, 'window')
save_dir = os.path.join(data_root, 'model_input')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def one_hot_encode_binary(label):
    """
    将 0/1 标签转成 one-hot。

    输入:
        label shape = (N,)

    输出:
        one_hot shape = (N, 2)
    """
    label = np.asarray(label).squeeze().astype(np.int64)

    if not np.all(np.isin(label, [0, 1])):
        raise ValueError(f"label 必须是 0/1，但得到 {np.unique(label)}")

    one_hot = np.zeros((label.shape[0], 2), dtype=np.float32)
    one_hot[np.arange(label.shape[0]), label] = 1.0

    return one_hot


def check_window_input(subject, eeg, hbo, hbr, label, trial_session):
    """
    检查 window 阶段输出是否符合 model_input 构造要求。

    期望:
        eeg: (trials, 10, 16, 16, 600)
        hbo: (trials, 20, 16, 16, 30)
        hbr: (trials, 20, 16, 16, 30)
        label: (trials,)
        trial_session: (trials,)
    """
    print(f"\n[{subject}] window 输入检查:")
    print("EEG:", eeg.shape)
    print("HbO:", hbo.shape)
    print("HbR:", hbr.shape)
    print("label:", label.shape, "unique:", np.unique(label))
    print("trial_session:", trial_session.shape, "unique:", np.unique(trial_session))

    if eeg.ndim != 5:
        raise ValueError(
            f"{subject}: EEG 应为 5 维 "
            f"(trials, 10, 16, 16, 600)，但得到 {eeg.shape}"
        )

    if hbo.ndim != 5 or hbr.ndim != 5:
        raise ValueError(
            f"{subject}: HbO/HbR 应为 5 维 "
            f"(trials, 20, 16, 16, 30)，但得到 HbO={hbo.shape}, HbR={hbr.shape}"
        )

    n_trials = eeg.shape[0]

    if hbo.shape[0] != n_trials or hbr.shape[0] != n_trials:
        raise ValueError(
            f"{subject}: EEG/HbO/HbR trial 数不一致，"
            f"EEG={n_trials}, HbO={hbo.shape[0]}, HbR={hbr.shape[0]}"
        )

    if len(label) != n_trials:
        raise ValueError(
            f"{subject}: label 数量和 trial 数不一致，"
            f"label={len(label)}, trials={n_trials}"
        )

    if len(trial_session) != n_trials:
        raise ValueError(
            f"{subject}: trial_session 数量和 trial 数不一致，"
            f"trial_session={len(trial_session)}, trials={n_trials}"
        )

    if eeg.shape[1] != eeg_segments_number:
        raise ValueError(
            f"{subject}: EEG 窗口数应为 {eeg_segments_number}，但得到 {eeg.shape[1]}"
        )

    if hbo.shape[1] < fnirs_required_segments or hbr.shape[1] < fnirs_required_segments:
        raise ValueError(
            f"{subject}: fNIRS 窗口数不足。"
            f"至少需要 {fnirs_required_segments} 个窗口，"
            f"但 HbO={hbo.shape[1]}, HbR={hbr.shape[1]}"
        )

    if eeg.shape[2:] != (16, 16, 600):
        raise ValueError(
            f"{subject}: EEG 单窗口 shape 应为 (16, 16, 600)，"
            f"但得到 {eeg.shape[2:]}"
        )

    if hbo.shape[2:] != (16, 16, 30):
        raise ValueError(
            f"{subject}: HbO 单窗口 shape 应为 (16, 16, 30)，"
            f"但得到 {hbo.shape[2:]}"
        )

    if hbr.shape[2:] != (16, 16, 30):
        raise ValueError(
            f"{subject}: HbR 单窗口 shape 应为 (16, 16, 30)，"
            f"但得到 {hbr.shape[2:]}"
        )

    if not np.all(np.isin(label, [0, 1])):
        raise ValueError(
            f"{subject}: label 必须是 0/1，但得到 {np.unique(label)}"
        )

    if not np.all(np.isin(trial_session, [0, 1, 2])):
        raise ValueError(
            f"{subject}: trial_session 应为 0/1/2，但得到 {np.unique(trial_session)}"
        )


def build_fnirs_lag_input(hbo, hbr):
    """
    构造 STA-Net 需要的 fNIRS 输入。

    输入:
        hbo shape = (trials, 20, 16, 16, 30)
        hbr shape = (trials, 20, 16, 16, 30)

    输出:
        fnirs_input shape = (trials * 10, 11, 16, 16, 30, 2)

    逻辑:
        对于每个 EEG 窗口 w:
            取 fNIRS[w : w + 11]
        即:
            当前 fNIRS 窗口 + 后续 10 个延迟窗口
    """
    n_trials = hbo.shape[0]

    fnirs_session_dataset = np.zeros(
        (
            n_trials,
            eeg_segments_number,
            fnirs_lag_length,
            16,
            16,
            30,
            2
        ),
        dtype=hbo.dtype
    )

    for trial_idx in range(n_trials):
        for w in range(eeg_segments_number):
            hbo_sample = hbo[
                trial_idx,
                w:w + fnirs_lag_length,
                ...  # shape = (11, 16, 16, 30)
            ]

            hbr_sample = hbr[
                trial_idx,
                w:w + fnirs_lag_length,
                ...  # shape = (11, 16, 16, 30)
            ]

            if hbo_sample.shape[0] != fnirs_lag_length:
                raise ValueError(
                    f"trial={trial_idx}, window={w}: "
                    f"HbO lag 长度不足，得到 {hbo_sample.shape[0]}"
                )

            if hbr_sample.shape[0] != fnirs_lag_length:
                raise ValueError(
                    f"trial={trial_idx}, window={w}: "
                    f"HbR lag 长度不足，得到 {hbr_sample.shape[0]}"
                )

            hbo_sample = np.expand_dims(hbo_sample, axis=-1)
            hbr_sample = np.expand_dims(hbr_sample, axis=-1)

            fnirs_sample = np.concatenate(
                [hbo_sample, hbr_sample],
                axis=-1
            )

            fnirs_session_dataset[trial_idx, w, ...] = fnirs_sample

    fnirs_input = fnirs_session_dataset.reshape(
        n_trials * eeg_segments_number,
        fnirs_lag_length,
        16,
        16,
        30,
        2
    )

    return fnirs_input


# =========================
# 主流程
# =========================
subject_list = sorted([f for f in os.listdir(subject_path) if f.endswith('.npz')])

print("\n==============================")
print(f"当前任务: {TASK}")
print(f"window 路径: {subject_path}")
print(f"model_input 保存路径: {save_dir}")
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
                f"请先使用新版 window.py 保存 trial_session。"
            )

        if 'common_selection' in data.files:
            common_selection = data['common_selection']
        else:
            common_selection = None

    label = np.asarray(label).squeeze().astype(np.int64)
    trial_session = np.asarray(trial_session).squeeze().astype(np.int64)

    check_window_input(
        subject=subject,
        eeg=eeg,
        hbo=hbo,
        hbr=hbr,
        label=label,
        trial_session=trial_session
    )

    n_trials = eeg.shape[0]
    total_samples = n_trials * eeg_segments_number

    # =========================
    # EEG input
    # =========================
    # eeg shape:
    #     (trials, 10, 16, 16, 600)
    #
    # expand 后:
    #     (trials, 10, 16, 16, 600, 1)
    #
    # reshape 后:
    #     (trials * 10, 16, 16, 600, 1)
    eeg_session_dataset = np.expand_dims(eeg, axis=-1)

    eeg_input = eeg_session_dataset.reshape(
        total_samples,
        16,
        16,
        600,
        1
    )

    # =========================
    # fNIRS input
    # =========================
    fnirs_input = build_fnirs_lag_input(
        hbo=hbo,
        hbr=hbr
    )

    # =========================
    # Label input
    # =========================
    # 每个 trial 有 10 个 EEG window，所以 label 重复 10 次
    label_repeated = np.repeat(
        label,
        repeats=eeg_segments_number,
        axis=0
    )

    label_input = one_hot_encode_binary(
        label_repeated
    )

    # =========================
    # sample_session
    # =========================
    # trial_session 是 trial 级别:
    #     shape = (trials,)
    #
    # sample_session 是 sample/window 级别:
    #     shape = (trials * 10,)
    #
    # 后续 run_sta_net.py 用 sample_session 做三折 session-wise 划分。
    sample_session = np.repeat(
        trial_session,
        repeats=eeg_segments_number,
        axis=0
    ).astype(np.int64)

    # =========================
    # 最终检查
    # =========================
    print(f"\n[{subject}] 最终输入维度:")
    print("EEG Input:", eeg_input.shape)
    print("fNIRS Input:", fnirs_input.shape)
    print("Label Input:", label_input.shape)
    print("sample_session:", sample_session.shape, "unique:", np.unique(sample_session))

    if eeg_input.shape[0] != fnirs_input.shape[0]:
        raise ValueError(
            f"{subject}: EEG 和 fNIRS 样本数不一致，"
            f"EEG={eeg_input.shape[0]}, fNIRS={fnirs_input.shape[0]}"
        )

    if eeg_input.shape[0] != label_input.shape[0]:
        raise ValueError(
            f"{subject}: EEG 和 label 样本数不一致，"
            f"EEG={eeg_input.shape[0]}, label={label_input.shape[0]}"
        )

    if eeg_input.shape[0] != sample_session.shape[0]:
        raise ValueError(
            f"{subject}: EEG 和 sample_session 数量不一致，"
            f"EEG={eeg_input.shape[0]}, sample_session={sample_session.shape[0]}"
        )

    for sess in [0, 1, 2]:
        sess_count = np.sum(sample_session == sess)
        print(f"session {sess} sample 数: {sess_count}")

    save_dict = {
        'eeg': eeg_input,
        'fnirs': fnirs_input,
        'label': label_input,
        'sample_session': sample_session,
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