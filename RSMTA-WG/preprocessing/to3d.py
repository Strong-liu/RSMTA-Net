import numpy as np
from scipy.interpolate import griddata
import os
from tqdm import tqdm  # 引入进度条库

# ==========================================
# 1. 坐标定义 (保持不变)
# ==========================================
# all points
x = np.arange(16)
y = np.arange(16)
xx, yy = np.meshgrid(x, y)
all_points = np.column_stack((xx.ravel(), yy.ravel()))

# eeg interpolate [x, y] -> 注意：你的注释写着[x,y]，但代码逻辑实际用作了[row, col]即[y, x]
known_eeg_point_coordinates = np.array([
    [0., 6.], [2., 5.], [2., 8.], [3., 7.], [5., 2.], [5., 6.], [7., 1.], [7., 4.],
    [7., 8.], [9., 2.], [9., 6.], [11., 2.], [11., 5.], [11., 8.], [13., 8.], [14., 6.],
    [0., 10.], [2., 11.], [3., 9.], [5., 10.], [5., 14.], [7., 12.], [7., 15.], [9., 10.],
    [9., 14.], [11., 11.], [11., 14.], [14., 10.]
])

unknown_eeg_point_coordinates = np.array(
    [coord for coord in all_points if coord.tolist() not in known_eeg_point_coordinates.tolist()]).astype(float)

# fnirs interpolate
known_fnirs_point_coordinates = np.array([
    [2., 4.], [3., 4.], [1., 5.], [2., 5.], [1., 7.], [3., 6.], [2., 7.], [3., 8.],
    [1., 8.], [2., 9.], [1., 9.], [6., 4.], [7., 5.], [7., 3.], [8., 4.], [10., 5.],
    [11., 6.], [11., 4.], [12., 5.], [3., 10.], [2., 11.], [3., 12.], [1., 11.], [2., 12.],
    [6., 12.], [7., 13.], [7., 11.], [8., 12.], [10., 11.], [11., 12.], [11., 10.], [12., 11.],
    [12., 8.], [13., 7.], [13., 9.], [14., 8.]
])

unknown_fnirs_point_coordinates = np.array(
    [coord for coord in all_points if coord.tolist() not in known_fnirs_point_coordinates.tolist()]).astype(float)

# ==========================================
# 2. 预处理数据提取与 3D 张量构建
# ==========================================
n_epoch = 60
subject_path = r'F:\学生文件夹\刘强\DataSet\WG\epoch'
save_dir = r'F:\学生文件夹\刘强\DataSet\WG\d3'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

subject_list = os.listdir(subject_path)

# 最外层受试者循环进度条
for subject in tqdm(subject_list, desc="Processing Subjects"):
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        hbo = data['hbo']
        hbr = data['hbr']
        label = data['label']

    eeg_3dtensor = np.zeros((eeg.shape[0], 16, 16, eeg.shape[-1]))
    fnirs_hbo_3dtensor = np.zeros((hbo.shape[0], 16, 16, hbo.shape[-1]))
    fnirs_hbr_3dtensor = np.zeros((hbr.shape[0], 16, 16, hbr.shape[-1]))

    assert eeg.shape[0] == hbo.shape[0] == hbr.shape[0] == n_epoch

    # 提取常量时间维度大小，方便构建矩阵
    T_eeg = eeg.shape[-1]
    T_fnirs = hbo.shape[-1]

    # Epoch 循环进度条
    for e in tqdm(range(n_epoch), desc=f"Epochs for {subject}", leave=False):

        # --------------------------------------------------
        # [EEG 向量化插值] 彻底消灭时间点 for 循环！
        # --------------------------------------------------
        # 1. 直接取出当前 epoch 所有通道、所有时间点的数据。形状: (28, T_eeg)
        known_eeg_vals = eeg[e, :, :]

        # 2. 一次性进行三次样条插值。输出形状自动变为: (228, T_eeg)
        interp_eeg_vals = griddata(points=known_eeg_point_coordinates,
                                   values=known_eeg_vals,
                                   xi=unknown_eeg_point_coordinates,
                                   method='cubic')

        # 3. 构建临时 3D 序列收纳盒
        eeg_seq = np.full((16, 16, T_eeg), np.nan)

        # 4. 批量填充已知点 (高级索引)
        known_y = known_eeg_point_coordinates[:, 0].astype(int)
        known_x = known_eeg_point_coordinates[:, 1].astype(int)
        eeg_seq[known_y, known_x, :] = known_eeg_vals

        # 5. 批量填充三次样条插值出的未知点
        unk_y = unknown_eeg_point_coordinates[:, 0].astype(int)
        unk_x = unknown_eeg_point_coordinates[:, 1].astype(int)
        eeg_seq[unk_y, unk_x, :] = interp_eeg_vals

        # 6. 最近邻插值填补 NaN (凸包外侧的边缘点)
        # 空间拓扑是不变的，所以只需要取第0个时间点的二维切片找 NaN 坐标即可
        nan_mask = np.isnan(eeg_seq[:, :, 0])
        if np.any(nan_mask):
            valid_mask = ~nan_mask
            valid_coords = np.argwhere(valid_mask)
            nan_coords = np.argwhere(nan_mask)

            # 取出有效数据点，形状变为 (N_valid, T_eeg)
            valid_vals = eeg_seq[valid_mask]

            # 对所有的 NaN 点一次性进行最近邻插值
            nearest_vals = griddata(points=valid_coords,
                                    values=valid_vals,
                                    xi=nan_coords,
                                    method='nearest')

            # 批量填回 NaN 位置
            eeg_seq[nan_mask] = nearest_vals

        eeg_3dtensor[e] = eeg_seq

        # --------------------------------------------------
        # [fNIRS 向量化插值] 同理处理 HbO 和 HbR
        # --------------------------------------------------
        # 形状: (36, T_fnirs)
        known_hbo_vals = hbo[e, :, :]
        known_hbr_vals = hbr[e, :, :]

        # 一次性批量插值
        interp_hbo_vals = griddata(known_fnirs_point_coordinates, known_hbo_vals, unknown_fnirs_point_coordinates,
                                   method='cubic')
        interp_hbr_vals = griddata(known_fnirs_point_coordinates, known_hbr_vals, unknown_fnirs_point_coordinates,
                                   method='cubic')

        hbo_seq = np.full((16, 16, T_fnirs), np.nan)
        hbr_seq = np.full((16, 16, T_fnirs), np.nan)

        # 批量填充已知点
        fk_y = known_fnirs_point_coordinates[:, 0].astype(int)
        fk_x = known_fnirs_point_coordinates[:, 1].astype(int)
        hbo_seq[fk_y, fk_x, :] = known_hbo_vals
        hbr_seq[fk_y, fk_x, :] = known_hbr_vals

        # 批量填充插值点
        fu_y = unknown_fnirs_point_coordinates[:, 0].astype(int)
        fu_x = unknown_fnirs_point_coordinates[:, 1].astype(int)
        hbo_seq[fu_y, fu_x, :] = interp_hbo_vals
        hbr_seq[fu_y, fu_x, :] = interp_hbr_vals

        # 最近邻填补 NaN (HbO 和 HbR 坐标结构相同，共用同一个空间掩码)
        nan_mask_f = np.isnan(hbo_seq[:, :, 0])
        if np.any(nan_mask_f):
            valid_mask_f = ~nan_mask_f
            valid_coords_f = np.argwhere(valid_mask_f)
            nan_coords_f = np.argwhere(nan_mask_f)

            valid_hbo_vals = hbo_seq[valid_mask_f]
            valid_hbr_vals = hbr_seq[valid_mask_f]

            near_hbo_vals = griddata(valid_coords_f, valid_hbo_vals, nan_coords_f, method='nearest')
            near_hbr_vals = griddata(valid_coords_f, valid_hbr_vals, nan_coords_f, method='nearest')

            hbo_seq[nan_mask_f] = near_hbo_vals
            hbr_seq[nan_mask_f] = near_hbr_vals

        fnirs_hbo_3dtensor[e] = hbo_seq
        fnirs_hbr_3dtensor[e] = hbr_seq

    # 保存单个受试者数据
    save_dict = {
        'eeg': eeg_3dtensor,
        'hbo': fnirs_hbo_3dtensor,
        'hbr': fnirs_hbr_3dtensor,
        'label': label
    }

    save_name = subject
    np.savez(os.path.join(save_dir, save_name), **save_dict)

    # 因为外层有了 tqdm，你可以用 tqdm.write 替代 print 避免打乱进度条显示
    tqdm.write(f'============== saved {save_name} successfully =============')

print('\n全部数据处理完毕！')