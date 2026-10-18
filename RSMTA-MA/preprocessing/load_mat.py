import scipy.io as io
import numpy as np
import os


# =========================
# 任务选择
# =========================
# 可选: "MI" 或 "MA"
# MI: session 1, 3, 5 -> Python index [0, 2, 4]
# MA: session 2, 4, 6 -> Python index [1, 3, 5]
TASK = "MA"

RAW_ROOT = r'F:\学生文件夹\刘强\DataSet\MA+MI'
EEG_ROOT = os.path.join(RAW_ROOT, 'EEG_01-29')
NIRS_ROOT = os.path.join(RAW_ROOT, 'NIRS_01-29')

if TASK == "MI":
    session_indices = [0, 2, 4]
    save_dir = os.path.join(RAW_ROOT, 'MI', 'mat2array')
elif TASK == "MA":
    session_indices = [1, 3, 5]
    save_dir = os.path.join(RAW_ROOT, 'MA', 'mat2array')
else:
    raise ValueError("TASK 只能是 'MI' 或 'MA'")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# =========================
# 工具函数
# =========================
def get_subject_list(folder_path):
    """
    获取 subject 文件夹列表，例如:
    subject 01, subject 02, ..., subject 29
    """
    subject_list = []

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)

        if os.path.isdir(full_path) and filename.lower().startswith("subject"):
            subject_list.append(filename)

    subject_list = sorted(subject_list)

    return subject_list


def load_mat_file(path):
    """
    加载 mat 文件，并检查文件是否存在。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    return io.loadmat(path)


def extract_time(mrk_struct):
    """
    从 mrk{1,i} 中提取 time。

    你当前数据里 time 是每个 trial 的 onset 时间。
    通常单位是 ms。
    输出统一为 shape = (20,)。
    """
    time = np.asarray(mrk_struct['time'][0, 0]).squeeze().astype(float)

    return time


def extract_event_desc(mrk_struct):
    """
    从 mrk{1,i} 中提取 event.desc。

    你已经确认 event.desc 是 20×1 数组，里面包含 16 和 32。
    输出统一为 shape = (20,)。
    """
    event = mrk_struct['event'][0, 0]
    desc = np.asarray(event['desc'][0, 0]).squeeze().astype(int)

    return desc


def desc_to_binary_label(desc, subject_name, session_idx):
    """
    使用 event.desc 里的 16 / 32 生成 0 / 1 标签。

    MI:
        16 -> 0 -> left hand imagery
        32 -> 1 -> right hand imagery

    MA:
        16 -> 0 -> mental arithmetic
        32 -> 1 -> baseline
    """
    desc = np.asarray(desc).squeeze().astype(int)

    unique_desc = np.unique(desc)

    if not np.all(np.isin(desc, [16, 32])):
        raise ValueError(
            f"{subject_name}, session index {session_idx}: "
            f"event.desc 中出现非 16/32 的 marker: {unique_desc}"
        )

    label = np.zeros_like(desc, dtype=np.int64)
    label[desc == 16] = 0
    label[desc == 32] = 1

    return label


def check_session_basic_info(
    subject_name,
    session_idx,
    eeg,
    eeg_time,
    label,
    low_wl,
    high_wl,
    fnirs_time
):
    """
    检查当前 session 的基本维度。
    """
    if eeg.ndim != 2:
        raise ValueError(
            f"{subject_name}, session {session_idx}: EEG 不是二维数组，shape={eeg.shape}"
        )

    if low_wl.ndim != 2 or high_wl.ndim != 2:
        raise ValueError(
            f"{subject_name}, session {session_idx}: NIRS low/high 不是二维数组，"
            f"low={low_wl.shape}, high={high_wl.shape}"
        )

    # EEG 应该是 32 通道：30 EEG + VEOG + HEOG
    if eeg.shape[0] != 32:
        print(
            f"[Warning] {subject_name}, session {session_idx}: "
            f"EEG 通道数不是 32，而是 {eeg.shape[0]}。"
        )

    # NIRS 应该是 36 低波长 + 36 高波长
    if low_wl.shape[0] != 36 or high_wl.shape[0] != 36:
        raise ValueError(
            f"{subject_name}, session {session_idx}: "
            f"NIRS low/high 通道数异常，low={low_wl.shape}, high={high_wl.shape}"
        )

    if len(eeg_time) != 20:
        print(
            f"[Warning] {subject_name}, session {session_idx}: "
            f"EEG marker 数量不是 20，而是 {len(eeg_time)}。"
        )

    if len(fnirs_time) != 20:
        print(
            f"[Warning] {subject_name}, session {session_idx}: "
            f"fNIRS marker 数量不是 20，而是 {len(fnirs_time)}。"
        )

    if len(label) != len(eeg_time):
        raise ValueError(
            f"{subject_name}, session {session_idx}: "
            f"label 数量和 EEG marker 数量不一致，"
            f"label={len(label)}, eeg_time={len(eeg_time)}"
        )


# =========================
# 主流程
# =========================
subject_list = get_subject_list(EEG_ROOT)

print("\n==============================")
print(f"当前任务: {TASK}")
print(f"处理 session index: {session_indices}")
print(f"被试数量: {len(subject_list)}")
print(f"保存目录: {save_dir}")
print("==============================\n")


for name in subject_list:
    print(f"\n{'=' * 60}")
    print(f"正在处理: {name}")
    print(f"{'=' * 60}")

    # -------------------------------------------------
    # 1. 加载 EEG / NIRS 数据
    # -------------------------------------------------
    # 你的本地 EEG 正式任务数据位于 with occular artifact 文件夹内
    eeg_cnt_path = os.path.join(
        EEG_ROOT,
        name,
        'with occular artifact',
        'cnt.mat'
    )
    eeg_mrk_path = os.path.join(
        EEG_ROOT,
        name,
        'with occular artifact',
        'mrk.mat'
    )
    eeg_mnt_path = os.path.join(
        EEG_ROOT,
        name,
        'with occular artifact',
        'mnt.mat'
    )

    # NIRS 正式任务数据在 NIRS_01-29/subject xx/ 下
    fnirs_cnt_path = os.path.join(
        NIRS_ROOT,
        name,
        'cnt.mat'
    )
    fnirs_mrk_path = os.path.join(
        NIRS_ROOT,
        name,
        'mrk.mat'
    )
    fnirs_mnt_path = os.path.join(
        NIRS_ROOT,
        name,
        'mnt.mat'
    )

    eeg_data = load_mat_file(eeg_cnt_path)
    eeg_mrk_data = load_mat_file(eeg_mrk_path)
    eeg_mnt_data = load_mat_file(eeg_mnt_path)

    fnirs_data = load_mat_file(fnirs_cnt_path)
    fnirs_mrk_data = load_mat_file(fnirs_mrk_path)
    fnirs_mnt_data = load_mat_file(fnirs_mnt_path)

    eeg_list = []
    eeg_time_list = []
    label_list = []

    low_wl_list = []
    high_wl_list = []
    fnirs_time_list = []

    trial_session_list = []

    eeg_offset_ms = 0.0
    fnirs_offset_ms = 0.0

    # -------------------------------------------------
    # 2. 按任务选择 3 个 session
    # -------------------------------------------------
    for local_session_id, i in enumerate(session_indices):
        print(
            f"\n--- {name} | TASK={TASK} | "
            f"原始 session index={i} | local_session={local_session_id} ---"
        )

        # =========================
        # EEG
        # =========================
        # cnt{1,i}.x 通常是 time × channel
        # 转置后变成 channel × time
        temp_eeg = eeg_data['cnt'][0, i]['x'][0, 0].T

        eeg_mrk_struct = eeg_mrk_data['mrk'][0, i]

        # EEG marker onset，单位 ms
        temp_eeg_time = extract_time(eeg_mrk_struct)

        # 用 event.desc 的 16/32 生成 0/1 标签
        temp_desc = extract_event_desc(eeg_mrk_struct)
        temp_label = desc_to_binary_label(temp_desc, name, i)

        # 给 EEG marker 时间加 session 拼接 offset
        temp_eeg_time_offset = temp_eeg_time + eeg_offset_ms

        eeg_list.append(temp_eeg)
        eeg_time_list.append(temp_eeg_time_offset)
        label_list.append(temp_label)

        # 保存 trial 属于哪个 local session，后续如果剔除 trial，可以用于 session-wise 切分
        trial_session_list.append(
            np.full(
                shape=temp_label.shape,
                fill_value=local_session_id,
                dtype=np.int64
            )
        )

        # 更新 EEG offset
        # temp_eeg.shape[1] 是当前 session 的采样点数
        eeg_offset_ms += (temp_eeg.shape[1] / 200.0) * 1000.0

        # =========================
        # fNIRS
        # =========================
        # cnt{1,i}.x 通常是 time × 72
        # 转置后变成 72 × time
        temp_nirs = fnirs_data['cnt'][0, i]['x'][0, 0].T

        # 前 36 个是低波长，后 36 个是高波长
        # 注意：这里还不是 HbO/HbR，只是 raw light intensity
        temp_low_wl = temp_nirs[:36, :]
        temp_high_wl = temp_nirs[36:72, :]

        fnirs_mrk_struct = fnirs_mrk_data['mrk'][0, i]

        # fNIRS marker onset，单位 ms
        temp_fnirs_time = extract_time(fnirs_mrk_struct)
        temp_fnirs_time_offset = temp_fnirs_time + fnirs_offset_ms

        check_session_basic_info(
            subject_name=name,
            session_idx=i,
            eeg=temp_eeg,
            eeg_time=temp_eeg_time,
            label=temp_label,
            low_wl=temp_low_wl,
            high_wl=temp_high_wl,
            fnirs_time=temp_fnirs_time
        )

        low_wl_list.append(temp_low_wl)
        high_wl_list.append(temp_high_wl)
        fnirs_time_list.append(temp_fnirs_time_offset)

        # 更新 fNIRS offset
        fnirs_offset_ms += (temp_low_wl.shape[1] / 10.0) * 1000.0

        print("EEG:", temp_eeg.shape)
        print("EEG time:", temp_eeg_time.shape)
        print("event.desc unique:", np.unique(temp_desc))
        print("Label:", temp_label.shape, "unique:", np.unique(temp_label))
        print("NIRS low wavelength:", temp_low_wl.shape)
        print("NIRS high wavelength:", temp_high_wl.shape)
        print("fNIRS time:", temp_fnirs_time.shape)

    # -------------------------------------------------
    # 3. 拼接 3 个 session
    # -------------------------------------------------
    eeg = np.concatenate(eeg_list, axis=1)
    eeg_time = np.concatenate(eeg_time_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    low_wl = np.concatenate(low_wl_list, axis=1)
    high_wl = np.concatenate(high_wl_list, axis=1)
    fnirs_time = np.concatenate(fnirs_time_list, axis=0)

    trial_session = np.concatenate(trial_session_list, axis=0)

    # -------------------------------------------------
    # 4. 最终检查
    # -------------------------------------------------
    print(f"\n{name} 拼接后结果:")
    print("eeg:", eeg.shape)
    print("eeg_time:", eeg_time.shape)
    print("low_wl:", low_wl.shape)
    print("high_wl:", high_wl.shape)
    print("fnirs_time:", fnirs_time.shape)
    print("label:", label.shape, "unique:", np.unique(label), "count:", np.bincount(label))
    print("trial_session:", trial_session.shape, "unique:", np.unique(trial_session))

    if label.shape[0] != 60:
        print(
            f"[Warning] {name}: label 数量不是 60，而是 {label.shape[0]}。"
            f"如果每个 session 不是 20 trial，请确认数据是否完整。"
        )

    if eeg_time.shape[0] != label.shape[0]:
        raise ValueError(
            f"{name}: eeg_time 和 label 数量不一致，"
            f"eeg_time={eeg_time.shape[0]}, label={label.shape[0]}"
        )

    if fnirs_time.shape[0] != label.shape[0]:
        raise ValueError(
            f"{name}: fnirs_time 和 label 数量不一致，"
            f"fnirs_time={fnirs_time.shape[0]}, label={label.shape[0]}"
        )

    # -------------------------------------------------
    # 5. 保存
    # -------------------------------------------------
    # 为了兼容你后面的 preprocessing.py：
    # 仍然保存 hbo/hbr 这两个 key，但这里它们实际代表 low_wl/high_wl 原始光强。
    # 同时额外保存 low_wl/high_wl，避免以后语义混淆。
    save_dict = {
        'eeg': eeg,
        'eeg_time': eeg_time,

        # 兼容旧代码：注意这里还不是真 HbO/HbR
        'hbo': low_wl,
        'hbr': high_wl,

        # 更准确的新命名
        'low_wl': low_wl,
        'high_wl': high_wl,

        'fnirs_time': fnirs_time,
        'label': label,
        'trial_session': trial_session
    }

    save_path = os.path.join(save_dir, f'{name}.npz')

    np.savez(save_path, **save_dict)

    print(f"\n============== save {save_path} success ==============\n")