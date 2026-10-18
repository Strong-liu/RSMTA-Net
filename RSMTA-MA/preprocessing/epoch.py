import numpy as np
import mne
import os


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

load_dir = os.path.join(data_root, 'preprocessed')
save_dir = os.path.join(data_root, 'epoch')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# =========================
# Epoch 参数
# =========================
task_period = 10

eeg_sample_rate = 200
eeg_pre_onset = 5
eeg_post_onset = task_period

fnirs_sample_rate = 10
fnirs_pre_onset = 5
fnirs_post_onset = task_period + 12


def load_trial_session(load_dir, subject_no, n_events):
    """
    读取 preprocessing 阶段保存的 trial_session。

    trial_session:
        shape = (60,)
        取值为 0 / 1 / 2
        表示每个 trial 属于第几个 session

    如果文件不存在，则使用默认结构：
        前 20 个 trial -> session 0
        中间 20 个 trial -> session 1
        后 20 个 trial -> session 2

    注意：
        默认结构只适合每个 subject 完整保留 60 个 trial 的情况。
    """
    trial_session_path = os.path.join(load_dir, f'{subject_no}_trial_session.npy')

    if os.path.exists(trial_session_path):
        trial_session = np.load(trial_session_path).squeeze().astype(np.int64)
    else:
        print(
            f"[Warning] 未找到 {subject_no}_trial_session.npy，"
            f"将根据事件数量自动生成默认 session 标记。"
        )

        if n_events == 60:
            trial_session = np.repeat(np.arange(3), 20).astype(np.int64)
        else:
            raise ValueError(
                f"{subject_no}: 没有 trial_session 文件，且 n_events={n_events}，"
                f"无法安全推断 session 边界。"
            )

    if len(trial_session) != n_events:
        raise ValueError(
            f"{subject_no}: trial_session 数量和 events 数量不一致，"
            f"trial_session={len(trial_session)}, events={n_events}"
        )

    return trial_session


def check_epoch_boundary(subject_no, events, raw_eeg, events_fnirs, raw_hbo):
    """
    简单检查 EEG / fNIRS epoch 是否有足够的前后时间。
    这里只打印 warning，不强制中断，因为 MNE Epochs 自己也会处理边界问题。
    """
    eeg_first = events[0, 0]
    eeg_last = events[-1, 0]
    fnirs_first = events_fnirs[0, 0]
    fnirs_last = events_fnirs[-1, 0]

    eeg_first_ok = eeg_first - eeg_pre_onset * eeg_sample_rate >= 0
    eeg_last_ok = eeg_last + eeg_post_onset * eeg_sample_rate <= raw_eeg.n_times

    fnirs_first_ok = fnirs_first - fnirs_pre_onset * fnirs_sample_rate >= 0
    fnirs_last_ok = fnirs_last + fnirs_post_onset * fnirs_sample_rate <= raw_hbo.n_times

    print("\n[Epoch 边界检查]")
    print("EEG first event sample:", eeg_first)
    print("EEG last event sample:", eeg_last)
    print("EEG total samples:", raw_eeg.n_times)
    print("EEG first epoch enough pre-baseline:", eeg_first_ok)
    print("EEG last epoch enough post-task:", eeg_last_ok)

    print("fNIRS first event sample:", fnirs_first)
    print("fNIRS last event sample:", fnirs_last)
    print("fNIRS total samples:", raw_hbo.n_times)
    print("fNIRS first epoch enough pre-baseline:", fnirs_first_ok)
    print("fNIRS last epoch enough post-delay:", fnirs_last_ok)

    if not eeg_first_ok or not eeg_last_ok:
        print(f"[Warning] {subject_no}: EEG 某些 epoch 可能越界，会被 MNE 自动丢弃。")

    if not fnirs_first_ok or not fnirs_last_ok:
        print(f"[Warning] {subject_no}: fNIRS 某些 epoch 可能越界，会被 MNE 自动丢弃。")


# =========================
# 主循环
# =========================
for sub_id in range(1, 30):
    subject_no = f'subject {sub_id:02d}'

    print(f"\n{'=' * 60}")
    print(f"正在切分 Epoch: {subject_no} | TASK={TASK}")
    print(f"{'=' * 60}")

    eeg_raw_path = os.path.join(load_dir, f'{subject_no}_eeg_raw.fif')
    hbo_raw_path = os.path.join(load_dir, f'{subject_no}_hbo_raw.fif')
    hbr_raw_path = os.path.join(load_dir, f'{subject_no}_hbr_raw.fif')
    events_path = os.path.join(load_dir, f'{subject_no}_events.npy')
    fnirs_events_path = os.path.join(load_dir, f'{subject_no}_fnirs_events.npy')

    if not os.path.exists(eeg_raw_path):
        print(f"找不到 {subject_no} 的 EEG 预处理文件，已跳过。")
        continue

    if not os.path.exists(hbo_raw_path) or not os.path.exists(hbr_raw_path):
        print(f"找不到 {subject_no} 的 fNIRS 预处理文件，已跳过。")
        continue

    if not os.path.exists(events_path):
        print(f"找不到 {subject_no} 的 EEG events 文件，已跳过。")
        continue

    # =========================
    # 1. 载入连续信号和事件
    # =========================
    raw_eeg = mne.io.read_raw_fif(eeg_raw_path, preload=True)
    raw_hbo = mne.io.read_raw_fif(hbo_raw_path, preload=True)
    raw_hbr = mne.io.read_raw_fif(hbr_raw_path, preload=True)

    events = np.load(events_path)

    if os.path.exists(fnirs_events_path):
        events_fnirs = np.load(fnirs_events_path)
    else:
        print(
            f"[Warning] 未找到 {subject_no}_fnirs_events.npy，"
            f"将从 EEG events 换算 fNIRS events。"
        )
        events_fnirs = events.copy()
        events_fnirs[:, 0] = np.round(events[:, 0] / 200.0 * 10.0).astype(int)

    trial_session = load_trial_session(
        load_dir=load_dir,
        subject_no=subject_no,
        n_events=len(events)
    )

    print("\n[载入检查]")
    print("raw_eeg:", raw_eeg.get_data().shape)
    print("raw_hbo:", raw_hbo.get_data().shape)
    print("raw_hbr:", raw_hbr.get_data().shape)
    print("events:", events.shape, "event ids:", np.unique(events[:, 2]))
    print("events_fnirs:", events_fnirs.shape, "event ids:", np.unique(events_fnirs[:, 2]))
    print("trial_session:", trial_session.shape, "unique:", np.unique(trial_session))

    check_epoch_boundary(
        subject_no=subject_no,
        events=events,
        raw_eeg=raw_eeg,
        events_fnirs=events_fnirs,
        raw_hbo=raw_hbo
    )

    # =========================
    # 2. EEG Epoch 切分
    # =========================
    # EEG 提取 -5s 到 +10s
    # baseline=None：这里不做 EEG baseline correction，保持和前面滤波/ICA流程一致
    # reject_by_annotation=True：如果你在 raw_eeg.plot 中标注了坏段，重叠 trial 会被删除
    epochs_eeg = mne.Epochs(
        raw_eeg,
        events,
        tmin=-float(eeg_pre_onset),
        tmax=float(eeg_post_onset) - 1.0 / eeg_sample_rate,
        baseline=None,
        reject_by_annotation=True,
        preload=True,
    )

    # =========================
    # 3. fNIRS Epoch 切分
    # =========================
    # fNIRS 提取 -5s 到 +22s
    # 原因：
    # 后续 window.py 会从 -2s 开始切 20 个 3s 窗口：
    # -2~1, -1~2, ..., 17~20
    # 但为了稳妥保留到 +22s，这里与前面设计一致。
    # baseline=(-5, -2)：对 HbO/HbR 做 baseline correction
    epochs_hbo = mne.Epochs(
        raw_hbo,
        events_fnirs,
        tmin=-float(fnirs_pre_onset),
        tmax=float(fnirs_post_onset) - 1.0 / fnirs_sample_rate,
        baseline=(-5.0, -2.0),
        reject_by_annotation=False,
        preload=True,
    )

    epochs_hbr = mne.Epochs(
        raw_hbr,
        events_fnirs,
        tmin=-float(fnirs_pre_onset),
        tmax=float(fnirs_post_onset) - 1.0 / fnirs_sample_rate,
        baseline=(-5.0, -2.0),
        reject_by_annotation=False,
        preload=True,
    )

    # =========================
    # 4. EEG / HbO / HbR / session 信息对齐
    # =========================
    # epochs.selection 保存的是：当前保留下来的 epoch 对应原始 events 的下标
    # 因为 EEG 可能由于 annotation 删除 trial，fNIRS 也可能因为边界等原因删除 trial，
    # 所以这里取三者共同保留的原始 trial 下标。
    common_selection = np.intersect1d(epochs_eeg.selection, epochs_hbo.selection)
    common_selection = np.intersect1d(common_selection, epochs_hbr.selection)

    if len(common_selection) == 0:
        print(f"[Warning] {subject_no}: EEG / HbO / HbR 没有共同保留的 trial，已跳过。")
        continue

    eeg_idx = np.where(np.isin(epochs_eeg.selection, common_selection))[0]
    hbo_idx = np.where(np.isin(epochs_hbo.selection, common_selection))[0]
    hbr_idx = np.where(np.isin(epochs_hbr.selection, common_selection))[0]

    epochs_eeg = epochs_eeg[eeg_idx]
    epochs_hbo = epochs_hbo[hbo_idx]
    epochs_hbr = epochs_hbr[hbr_idx]

    # 根据 common_selection 同步筛选 trial_session
    trial_session_final = trial_session[common_selection]

    # =========================
    # 5. 提取最终矩阵
    # =========================
    eeg_final = epochs_eeg.get_data()
    hbo_final = epochs_hbo.get_data()
    hbr_final = epochs_hbr.get_data()

    # 注意：
    # preprocessing.py 中 events 第三列是 label + 1，即 1/2
    # 这里必须转回 0/1，方便后面 to_categorical(num_classes=2)
    labels_final = epochs_eeg.events[:, 2].astype(np.int64) - 1

    # =========================
    # 6. 最终一致性检查
    # =========================
    n_trials = len(labels_final)

    if not (
        eeg_final.shape[0]
        == hbo_final.shape[0]
        == hbr_final.shape[0]
        == len(trial_session_final)
        == n_trials
    ):
        raise ValueError(
            f"{subject_no}: EEG/HbO/HbR/label/trial_session 数量不一致："
            f"eeg={eeg_final.shape[0]}, "
            f"hbo={hbo_final.shape[0]}, "
            f"hbr={hbr_final.shape[0]}, "
            f"label={n_trials}, "
            f"trial_session={len(trial_session_final)}"
        )

    if not np.all(np.isin(labels_final, [0, 1])):
        raise ValueError(
            f"{subject_no}: labels_final 必须是 0/1，但得到 {np.unique(labels_final)}"
        )

    print(f"\n[处理结果] {subject_no}")
    print("原始 trial 数:", len(events))
    print("保留 trial 数:", n_trials)
    print("EEG:", eeg_final.shape)
    print("HbO:", hbo_final.shape)
    print("HbR:", hbr_final.shape)
    print("label:", labels_final.shape, "unique:", np.unique(labels_final), "count:", np.bincount(labels_final))
    print("trial_session:", trial_session_final.shape, "unique:", np.unique(trial_session_final))

    # 打印每个 session 剩余 trial 数，后面判断 session-wise 是否平衡很有用
    for sess in [0, 1, 2]:
        sess_count = np.sum(trial_session_final == sess)
        print(f"session {sess} 剩余 trial 数: {sess_count}")

    # =========================
    # 7. 保存
    # =========================
    save_path = os.path.join(save_dir, f'{subject_no}.npz')

    np.savez(
        save_path,
        eeg=eeg_final,
        hbo=hbo_final,
        hbr=hbr_final,
        label=labels_final,
        trial_session=trial_session_final,
        common_selection=common_selection
    )

    print(f"\n============== save {save_path} success ==============\n")