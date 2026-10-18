import numpy as np
from scipy.interpolate import griddata
import mne
import os
import scipy.io as io
from tqdm import tqdm


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

epoch_path = os.path.join(data_root, 'epoch')
save_dir = os.path.join(data_root, 'd3')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# =========================
# EEG / fNIRS 通道名称
# =========================
target_eeg_ch_names = [
    'F7', 'AFF5h', 'F3', 'AFp1', 'AFp2', 'AFF6h', 'F4', 'F8',
    'AFF1h', 'AFF2h', 'Cz', 'Pz', 'FCC5h', 'FCC3h', 'CCP5h',
    'CCP3h', 'T7', 'P7', 'P3', 'PPO1h', 'POO1', 'POO2',
    'PPO2h', 'P4', 'FCC4h', 'FCC6h', 'CCP4h', 'CCP6h',
    'P8', 'T8'
]

fnirs_chn_names = [
    'AF7Fp1', 'AF3Fp1', 'AF3AFz', 'FpzFp1', 'FpzAFz', 'FpzFp2',
    'AF4AFz', 'AF4Fp2', 'AF8Fp2', 'OzPOz', 'OzO1', 'OzO2',
    'C5CP5', 'C5FC5', 'C5C3', 'FC3FC5', 'FC3C3', 'FC3FC1',
    'CP3CP5', 'CP3C3', 'CP3CP1', 'C1C3', 'C1FC1', 'C1CP1',
    'C2FC2', 'C2CP2', 'C2C4', 'FC4FC2', 'FC4C4', 'FC4FC6',
    'CP4CP6', 'CP4CP2', 'CP4C4', 'C6CP6', 'C6C4', 'C6FC6'
]


def matlab_cellstr_to_list(cell_array):
    """
    将 MATLAB cell string 转成 Python list[str]。

    兼容：
        1×N cell
        N×1 cell
        单个 cell 内部是 ndarray 字符串
        单个 cell 内部是普通 str
    """
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
    """
    自动判断坐标单位并转换为米。

    判断规则：
        max_abs > 10       大概率是 mm
        1 < max_abs <= 10  大概率是 cm
        max_abs <= 1       大概率已经是 m

    注意：
        对 to3D 投影来说，单位比例不会影响投影结果；
        但统一转成米更规范。
    """
    pos = np.asarray(pos, dtype=float)

    max_abs = np.max(np.abs(pos))

    if max_abs > 10:
        pos = pos / 1000.0
    elif max_abs > 1:
        pos = pos / 100.0

    return pos


def project_3d_to_16x16(pos_3d):
    """
    将 3D 坐标通过方位角等距投影映射到 16×16 浮点网格。

    输入:
        pos_3d shape = (channels, 3)

    输出:
        coords shape = (channels, 2)
        每一行为 [row, col]
    """
    pos_3d = np.asarray(pos_3d, dtype=float)

    if pos_3d.ndim != 2 or pos_3d.shape[1] != 3:
        raise ValueError(
            f"pos_3d 应为 shape=(channels, 3)，但得到 {pos_3d.shape}"
        )

    norms = np.linalg.norm(pos_3d, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("pos_3d 中存在零向量坐标，无法进行球面归一化。")

    pos_3d_norm = pos_3d / norms

    x = pos_3d_norm[:, 0]
    y = pos_3d_norm[:, 1]
    z = pos_3d_norm[:, 2]

    # 防止浮点误差导致 arccos 输入略微超过 [-1, 1]
    z = np.clip(z, -1.0, 1.0)

    # 方位角等距投影
    r = np.arccos(z) / (np.pi / 2)
    theta = np.arctan2(y, x)

    x_2d = r * np.cos(theta)
    y_2d = r * np.sin(theta)

    x_range = x_2d.max() - x_2d.min()
    y_range = y_2d.max() - y_2d.min()

    if x_range == 0 or y_range == 0:
        raise ValueError("投影后的 x 或 y 范围为 0，无法映射到 16×16 网格。")

    # 映射到 0~15 网格
    x_norm = (x_2d - x_2d.min()) / x_range
    y_norm = (y_2d - y_2d.min()) / y_range

    x_grid = x_norm * 15.0
    y_grid = y_norm * 15.0

    # 输出坐标格式为 [row, col]
    # 15 - y_grid 是为了让图像上方对应鼻子方向
    coords = np.column_stack((15.0 - y_grid, x_grid))

    return coords


def get_eeg_coords_16x16(ch_names):
    """
    根据 EEG 通道名称从 standard_1005 中读取 3D 坐标，
    然后投影到 16×16 网格。
    """
    montage = mne.channels.make_standard_montage('standard_1005')
    pos_dict = montage.get_positions()['ch_pos']

    pos_3d = []

    for ch in ch_names:
        if ch not in pos_dict:
            raise ValueError(
                f"EEG 通道 {ch} 不在 standard_1005 montage 中，请检查通道名。"
            )

        pos_3d.append(pos_dict[ch])

    pos_3d = np.asarray(pos_3d, dtype=float)

    coords = project_3d_to_16x16(pos_3d)

    return coords


def get_fnirs_coords_from_mnt_16x16(mnt_path):
    """
    从 NIRS mnt.mat 中读取 36 个 fNIRS 通道的真实 pos_3d，
    并映射到 16×16 网格坐标。

    mnt.pos_3d:
        shape = 3 × 36

    输出:
        coords shape = 36 × 2
        每行为 [row, col]

    关键修正：
        不再使用 standard_1005 拆 AF7Fp1 中点；
        直接使用数据集官方 mnt.pos_3d。
    """
    if not os.path.exists(mnt_path):
        raise FileNotFoundError(f"找不到 NIRS mnt.mat: {mnt_path}")

    mnt = io.loadmat(mnt_path)
    mnt_struct = mnt['mnt'][0, 0]

    anat_names = matlab_cellstr_to_list(mnt_struct['clab'])

    if len(anat_names) != 36:
        raise ValueError(
            f"{mnt_path}: mnt.clab 应为 36 个 fNIRS 通道，但得到 {len(anat_names)}。"
            f"当前读取结果: {anat_names}"
        )

    if list(anat_names) != list(fnirs_chn_names):
        print("\n[Warning] 当前 mnt.clab 与代码中的 fnirs_chn_names 不完全一致。")
        print("mnt.clab:", anat_names)
        print("code fnirs_chn_names:", fnirs_chn_names)
        print("后续仍使用 mnt.pos_3d 的原始顺序进行 fNIRS 插值。")

    # mnt.pos_3d shape = 3 × 36
    # 转成 36 × 3
    pos_3d = np.asarray(mnt_struct['pos_3d']).T

    if pos_3d.shape != (36, 3):
        raise ValueError(
            f"{mnt_path}: mnt.pos_3d 转置后应为 shape=(36, 3)，但得到 {pos_3d.shape}"
        )

    pos_3d = convert_pos_to_meter(pos_3d)

    coords = project_3d_to_16x16(pos_3d)

    print("\n[fNIRS mnt.pos_3d 投影检查]")
    print("mnt path:", mnt_path)
    print("anat_names first 5:", anat_names[:5])
    print("pos_3d shape:", pos_3d.shape)
    print("coords shape:", coords.shape)
    print("coords first 5:")
    print(coords[:5])

    return coords


def interpolate_3d(data, known_coords, name="data"):
    """
    执行 16×16 空间插值。

    输入:
        data shape = (trials, channels, timepoints)

    输出:
        out_tensor shape = (trials, 16, 16, timepoints)

    插值策略:
        1. 先用 cubic 插值，得到更平滑的空间拓扑图；
        2. cubic 在凸包外可能产生 NaN；
        3. 对 NaN 位置用 nearest 插值补全；
        4. 不再直接 fill_value=0，避免人为制造黑边。
    """
    if data.ndim != 3:
        raise ValueError(
            f"{name} 输入必须是 3 维: (trials, channels, timepoints)，"
            f"但当前 shape={data.shape}"
        )

    n_trials, n_channels, n_time = data.shape

    if known_coords.shape[0] != n_channels:
        raise ValueError(
            f"{name} 的通道数和坐标数不一致: "
            f"data channels={n_channels}, coords={known_coords.shape[0]}"
        )

    grid_y, grid_x = np.mgrid[0:16, 0:16]

    # 将 trial 和 time 展平，一次性对所有时间点做空间插值
    # data_reshaped shape = (channels, trials * timepoints)
    data_reshaped = data.transpose(1, 0, 2).reshape(n_channels, -1)

    # cubic 插值
    interp_cubic = griddata(
        points=known_coords,
        values=data_reshaped,
        xi=(grid_y, grid_x),
        method='cubic'
    )

    # nearest 插值，用于补全 cubic 无法覆盖的凸包外区域
    interp_nearest = griddata(
        points=known_coords,
        values=data_reshaped,
        xi=(grid_y, grid_x),
        method='nearest'
    )

    # cubic 可能产生 NaN，用 nearest 替换 NaN
    interp_res = np.where(
        np.isnan(interp_cubic),
        interp_nearest,
        interp_cubic
    )

    # interp_res shape = (16, 16, trials * timepoints)
    # 还原成 (trials, 16, 16, timepoints)
    out_tensor = interp_res.reshape(
        16,
        16,
        n_trials,
        n_time
    ).transpose(2, 0, 1, 3)

    return out_tensor


def check_epoch_data(subject, eeg, hbo, hbr, label, trial_session):
    """
    检查 epoch 阶段输出的数据是否符合 to3d 输入要求。
    """
    print(f"\n[{subject}] epoch 输入检查:")
    print("EEG:", eeg.shape)
    print("HbO:", hbo.shape)
    print("HbR:", hbr.shape)
    print("label:", label.shape, "unique:", np.unique(label))
    print("trial_session:", trial_session.shape, "unique:", np.unique(trial_session))

    if eeg.ndim != 3:
        raise ValueError(
            f"{subject}: EEG 应为 (trials, channels, timepoints)，但得到 {eeg.shape}"
        )

    if hbo.ndim != 3 or hbr.ndim != 3:
        raise ValueError(
            f"{subject}: HbO/HbR 应为 3 维，但得到 HbO={hbo.shape}, HbR={hbr.shape}"
        )

    if eeg.shape[0] != hbo.shape[0] or eeg.shape[0] != hbr.shape[0]:
        raise ValueError(
            f"{subject}: EEG/HbO/HbR trial 数不一致，"
            f"EEG={eeg.shape[0]}, HbO={hbo.shape[0]}, HbR={hbr.shape[0]}"
        )

    if eeg.shape[1] != len(target_eeg_ch_names):
        raise ValueError(
            f"{subject}: EEG 通道数应为 {len(target_eeg_ch_names)}，"
            f"但得到 {eeg.shape[1]}"
        )

    if hbo.shape[1] != len(fnirs_chn_names) or hbr.shape[1] != len(fnirs_chn_names):
        raise ValueError(
            f"{subject}: fNIRS 通道数应为 {len(fnirs_chn_names)}，"
            f"但得到 HbO={hbo.shape[1]}, HbR={hbr.shape[1]}"
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

    if not np.all(np.isin(label, [0, 1])):
        raise ValueError(
            f"{subject}: label 必须是 0/1，但得到 {np.unique(label)}"
        )


if __name__ == '__main__':
    print("\n==============================")
    print(f"当前任务: {TASK}")
    print(f"epoch 路径: {epoch_path}")
    print(f"d3 保存路径: {save_dir}")
    print("==============================\n")

    print("正在计算 EEG 空间投影坐标...")

    eeg_coords = get_eeg_coords_16x16(
        target_eeg_ch_names
    )

    print("EEG coords shape:", eeg_coords.shape)

    subject_list = sorted(
        [
            f for f in os.listdir(epoch_path)
            if f.endswith('.npz')
        ]
    )

    for subject in tqdm(subject_list, desc="正在进行 3D 拓扑插值"):
        subject_file_path = os.path.join(epoch_path, subject)

        with np.load(subject_file_path, allow_pickle=True) as data:
            eeg = data['eeg']
            hbo = data['hbo']
            hbr = data['hbr']
            label = data['label']

            if 'trial_session' in data.files:
                trial_session = data['trial_session']
            else:
                raise KeyError(
                    f"{subject} 中没有 trial_session。"
                    f"请先使用新版 epoch.py 保存 trial_session。"
                )

            if 'common_selection' in data.files:
                common_selection = data['common_selection']
            else:
                common_selection = None

        check_epoch_data(
            subject=subject,
            eeg=eeg,
            hbo=hbo,
            hbr=hbr,
            label=label,
            trial_session=trial_session
        )

        # =========================
        # 当前 subject 的 NIRS mnt.mat
        # =========================
        subject_no = subject.replace('.npz', '')

        fnirs_mnt_path = os.path.join(
            RAW_ROOT,
            'NIRS_01-29',
            subject_no,
            'mnt.mat'
        )

        fnirs_coords = get_fnirs_coords_from_mnt_16x16(
            fnirs_mnt_path
        )

        print("fNIRS coords shape:", fnirs_coords.shape)

        # =========================
        # EEG/HbO/HbR 插值成 16×16 空间图
        # =========================
        eeg_3dtensor = interpolate_3d(
            data=eeg,
            known_coords=eeg_coords,
            name=f"{subject} EEG"
        )

        hbo_3dtensor = interpolate_3d(
            data=hbo,
            known_coords=fnirs_coords,
            name=f"{subject} HbO"
        )

        hbr_3dtensor = interpolate_3d(
            data=hbr,
            known_coords=fnirs_coords,
            name=f"{subject} HbR"
        )

        print(f"\n[{subject}] 3D 输出检查:")
        print("EEG 3D:", eeg_3dtensor.shape)
        print("HbO 3D:", hbo_3dtensor.shape)
        print("HbR 3D:", hbr_3dtensor.shape)
        print("label:", label.shape)
        print("trial_session:", trial_session.shape)

        save_dict = {
            'eeg': eeg_3dtensor,
            'hbo': hbo_3dtensor,
            'hbr': hbr_3dtensor,
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