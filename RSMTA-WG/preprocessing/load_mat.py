import scipy.io as io
import numpy as np
import os

# ===================== 核心修改：替换为你的实际路径 =====================
# 1. 根路径（EEG和NIRS的上级文件夹）
root_path = r'F:\学生文件夹\刘强\DataSet\WG'
# 2. EEG和NIRS子文件夹路径
eeg_folder = os.path.join(root_path, 'EEG_01-26_MATLAB')
fnirs_folder = os.path.join(root_path, 'NIRS_01-26_MATLAB')
# 3. 保存路径（自动创建，避免文件夹不存在报错）
save_dir = r'F:\学生文件夹\刘强\DataSet\WG\mat2array'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ===================== 加载数据核心逻辑 =====================
subject_list = []

# 遍历EEG文件夹，提取被试编号（如01、02...26）
for filename in os.listdir(eeg_folder):
    # 过滤非文件夹（仅保留 {编号}-EEG 格式的文件夹）
    if os.path.isdir(os.path.join(eeg_folder, filename)) and '-EEG' in filename:
        subject_no = filename.split('-')[0]
        subject_list.append(subject_no)

# 去重 + 排序（确保被试编号按01-26顺序）
subject_list = sorted(list(set(subject_list)))
print(f"✅ 检测到被试列表：{subject_list}")

# 逐被试加载数据
for name in subject_list:
    try:
        # ---------------------- 1. 拼接EEG文件路径 ----------------------
        eeg_cnt_path = os.path.join(eeg_folder, f'{name}-EEG', 'cnt_wg.mat')
        eeg_mrk_path = os.path.join(eeg_folder, f'{name}-EEG', 'mrk_wg.mat')

        # ---------------------- 2. 拼接NIRS文件路径 ----------------------
        fnirs_cnt_path = os.path.join(fnirs_folder, f'{name}-NIRS', 'cnt_wg.mat')
        fnirs_mrk_path = os.path.join(fnirs_folder, f'{name}-NIRS', 'mrk_wg.mat')

        # ---------------------- 3. 加载MAT文件 ----------------------
        # 加载EEG数据
        eeg_data = io.loadmat(eeg_cnt_path)
        eeg_mrk_data = io.loadmat(eeg_mrk_path)
        # 加载fNIRS数据
        fnirs_data = io.loadmat(fnirs_cnt_path)
        fnirs_mrk_data = io.loadmat(fnirs_mrk_path)

        # ---------------------- 4. 提取核心数据（保持原索引逻辑） ----------------------
        # EEG信号 + 时间戳
        eeg = eeg_data['cnt_wg'][0, 0][3].T
        eeg_time = eeg_mrk_data['mrk_wg'][0, 0][0]

        # fNIRS的HbO/HbR + 时间戳
        hbo = fnirs_data['cnt_wg']['oxy'][0, 0][0, 0][5].T
        hbr = fnirs_data['cnt_wg']['deoxy'][0, 0][0, 0][5].T
        fnirs_time = fnirs_mrk_data['mrk_wg'][0, 0][0]

        # 标签（与EEG时间戳对应）
        label = eeg_mrk_data['mrk_wg'][0, 0][1]

        # ---------------------- 5. 打印维度验证 ----------------------
        print(f"\n📌 被试 {name} 数据维度：")
        print(f"EEG信号：{eeg.shape} | EEG时间戳：{eeg_time.shape}")
        print(f"HbO：{hbo.shape} | HbR：{hbr.shape} | fNIRS时间戳：{fnirs_time.shape}")
        print(f"标签：{label.shape}")

        # ---------------------- 6. 保存为npz文件 ----------------------
        save_dict = {
            'eeg': eeg,
            'eeg_time': eeg_time,
            'hbo': hbo,
            'hbr': hbr,
            'fnirs_time': fnirs_time,
            'label': label
        }
        save_path = os.path.join(save_dir, f'{name}.npz')
        np.savez(save_path, **save_dict)
        print(f"✅ 被试 {name} 数据已保存至：{save_path}")

    except FileNotFoundError as e:
        print(f"\n❌ 被试 {name} 缺失文件：{e}，跳过该被试")
    except IndexError as e:
        print(f"\n❌ 被试 {name} 数据索引错误：{e}，检查MAT文件结构")
    except Exception as e:
        print(f"\n❌ 被试 {name} 加载失败：{e}，跳过该被试")

print("\n==================== 所有被试处理完成 ====================")