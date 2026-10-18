import os
import sys
import random
import csv

# =========================
# 固定随机种子
# =========================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

# 如果训练时报 deterministic ops 相关错误，可以注释掉这一行
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)

# =========================
# 核心配置区：强制让 TensorFlow 找到 conda 虚拟环境里装好的 CUDA DLL 库
# 注意：这部分必须放在 import tensorflow 之前
# =========================
conda_env_path = os.path.dirname(sys.executable)
conda_dll_path = os.path.join(conda_env_path, 'Library', 'bin')

if os.path.exists(conda_dll_path):
    os.environ['PATH'] = conda_dll_path + os.pathsep + os.environ['PATH']

    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(conda_dll_path)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from sta import sta_net

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================
# 任务配置
# =========================
# 可选: "MI" 或 "MA"
# =========================
TASK = "MA"

RAW_ROOT = r'F:\学生文件夹\刘强\DataSet\MA+MI'

if TASK == "MI":
    data_root = os.path.join(RAW_ROOT, 'MI')
elif TASK == "MA":
    data_root = os.path.join(RAW_ROOT, 'MA')
else:
    raise ValueError("TASK 只能设置为 'MI' 或 'MA'")

subject_path = os.path.join(data_root, 'model_input')

# 建议换一个文件名，避免覆盖你之前 window-level validation 的结果
csv_file_path = os.path.join(
    data_root,
    'Compare_Results-RFB+mamba-like-modify1.2-trialval.csv'
)

cm_save_dir = os.path.join(
    data_root,
    'ConfusionMatrices-RFB+mamba-like-modify1.2-trialval'
)

if not os.path.exists(cm_save_dir):
    os.makedirs(cm_save_dir)


# =========================
# GPU 配置
# =========================
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print(f"\n[INFO] 成功检测到 {len(gpus)} 个 GPU（已配置显存按需增长），将使用 GPU 进行训练。")

    except RuntimeError as e:
        print(e)
else:
    print("\n[INFO] 未检测到 GPU，将使用 CPU 进行训练。")


# =========================
# Callback
# =========================
class targetacccallback(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super().__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        class_loss = logs.get('class_output_loss')

        if class_loss is not None and class_loss <= self.target_acc:
            print("\nReached target loss value {} so cancelling training!\n".format(self.target_acc))
            self.model.stop_training = True


# =========================
# 工具函数
# =========================
def check_model_input(subject, eeg, fnirs, label, sample_session):
    """
    检查 model_input 文件中的数据是否符合训练要求。
    """
    print(f"\n[{subject}] model_input 检查:")
    print("EEG:", eeg.shape)
    print("fNIRS:", fnirs.shape)
    print("label:", label.shape)
    print("sample_session:", sample_session.shape, "unique:", np.unique(sample_session))

    if eeg.ndim != 5:
        raise ValueError(
            f"{subject}: EEG 应为 5 维 "
            f"(samples, 16, 16, 600, 1)，但得到 {eeg.shape}"
        )

    if fnirs.ndim != 6:
        raise ValueError(
            f"{subject}: fNIRS 应为 6 维 "
            f"(samples, 11, 16, 16, 30, 2)，但得到 {fnirs.shape}"
        )

    if label.ndim != 2 or label.shape[1] != 2:
        raise ValueError(
            f"{subject}: label 应为 one-hot，shape=(samples, 2)，但得到 {label.shape}"
        )

    if eeg.shape[0] != fnirs.shape[0]:
        raise ValueError(
            f"{subject}: EEG 和 fNIRS 样本数不一致，"
            f"EEG={eeg.shape[0]}, fNIRS={fnirs.shape[0]}"
        )

    if eeg.shape[0] != label.shape[0]:
        raise ValueError(
            f"{subject}: EEG 和 label 样本数不一致，"
            f"EEG={eeg.shape[0]}, label={label.shape[0]}"
        )

    if eeg.shape[0] != sample_session.shape[0]:
        raise ValueError(
            f"{subject}: EEG 和 sample_session 样本数不一致，"
            f"EEG={eeg.shape[0]}, sample_session={sample_session.shape[0]}"
        )

    if not np.all(np.isin(sample_session, [0, 1, 2])):
        raise ValueError(
            f"{subject}: sample_session 应为 0/1/2，但得到 {np.unique(sample_session)}"
        )

    for session in [0, 1, 2]:
        session_count = np.sum(sample_session == session)
        print(f"session {session} sample 数:", session_count)

        if session_count == 0:
            raise ValueError(
                f"{subject}: session {session} 没有样本，无法做三折 session-wise 测试。"
            )


def make_dataset(eeg_data, fnirs_data, label_data, batch_size, shuffle=False, seed=SEED):
    """
    构造 tf.data.Dataset。

    当前 sta.py 输入：
        eeg_input
        fnirs_input

    当前 sta.py 输出：
        class_output
        eeg_output
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "eeg_input": eeg_data,
                "fnirs_input": fnirs_data
            },
            {
                "class_output": label_data,
                "eeg_output": label_data
            }
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=128,
            seed=seed,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def create_and_compile_model():
    """
    创建并编译模型。

    当前 sta.py 输出：
        class_output
        eeg_output
    """
    tf.keras.backend.clear_session()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = sta_net()

    model.compile(
        loss={
            'class_output': 'categorical_crossentropy',
            'eeg_output': 'categorical_crossentropy'
        },
        loss_weights={
            'class_output': 1.0,
            'eeg_output': 1.0
        },
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def save_confusion_matrix(y_true, y_pred, title, save_path):
    """
    保存混淆矩阵图片。
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()

    return cm


def split_train_val_by_trial(
    eeg,
    fnirs,
    label,
    sample_session,
    test_session,
    windows_per_trial=10,
    val_trial_ratio=0.2,
    seed=SEED
):
    """
    按 session 做测试集，按 trial 抽验证集。

    关键点：
        1 个 trial = 10 个连续 window。
        同一个 trial 的 10 个 window 必须整体进入 train 或 val。
        不能把同一个 trial 的部分 window 放进 train，部分 window 放进 val。

    输入:
        eeg:            (samples, 16, 16, 600, 1)
        fnirs:          (samples, 11, 16, 16, 30, 2)
        label:          (samples, 2)
        sample_session: (samples,)
        test_session:   0 / 1 / 2

    返回:
        eeg_train, fnirs_train, label_train
        eeg_val, fnirs_val, label_val
        eeg_test, fnirs_test, label_test
        all_eeg, all_fnirs, all_label
    """

    # =========================
    # 1. session-wise test split
    # =========================
    test_mask = sample_session == test_session
    train_mask = sample_session != test_session

    eeg_test = eeg[test_mask]
    fnirs_test = fnirs[test_mask]
    label_test = label[test_mask]

    all_eeg = eeg[train_mask]
    all_fnirs = fnirs[train_mask]
    all_label = label[train_mask]
    all_session = sample_session[train_mask]

    if eeg_test.shape[0] == 0:
        raise ValueError(
            f"test_session={test_session}: 测试集为空，请检查 sample_session。"
        )

    if all_eeg.shape[0] == 0:
        raise ValueError(
            f"test_session={test_session}: 训练候选集为空，请检查 sample_session。"
        )

    train_total = all_eeg.shape[0]

    if train_total % windows_per_trial != 0:
        raise ValueError(
            f"训练候选样本数 train_total={train_total} 不能被 windows_per_trial={windows_per_trial} 整除。\n"
            f"这说明 model_input 中的 window/trial 结构可能不是每个 trial 固定 {windows_per_trial} 个 window。"
        )

    # =========================
    # 2. 生成 trial_id
    # =========================
    n_trials = train_total // windows_per_trial

    trial_ids = np.repeat(
        np.arange(n_trials),
        windows_per_trial
    )

    # 每个 trial 的 label，取该 trial 第一个 window 的标签
    label_class = np.argmax(all_label, axis=1)

    trial_labels = []
    trial_sessions = []

    for trial_id in range(n_trials):
        start = trial_id * windows_per_trial
        end = start + windows_per_trial

        one_trial_labels = label_class[start:end]
        one_trial_sessions = all_session[start:end]

        # 检查一个 trial 内部 10 个 window 的 label 是否一致
        if len(np.unique(one_trial_labels)) != 1:
            raise ValueError(
                f"trial_id={trial_id}: 同一个 trial 内部 label 不一致：{one_trial_labels}"
            )

        # 检查一个 trial 内部 10 个 window 的 session 是否一致
        if len(np.unique(one_trial_sessions)) != 1:
            raise ValueError(
                f"trial_id={trial_id}: 同一个 trial 内部 sample_session 不一致：{one_trial_sessions}"
            )

        trial_labels.append(one_trial_labels[0])
        trial_sessions.append(one_trial_sessions[0])

    trial_labels = np.asarray(trial_labels, dtype=np.int64)
    trial_sessions = np.asarray(trial_sessions, dtype=np.int64)

    # =========================
    # 3. 按 trial-label 分层抽 validation trial
    # =========================
    rng = np.random.default_rng(seed)

    val_trials = []

    for cls in [0, 1]:
        cls_trials = np.where(trial_labels == cls)[0]

        if len(cls_trials) == 0:
            raise ValueError(
                f"训练候选集中没有 class {cls} 的 trial，无法分层抽验证集。"
            )

        n_val_cls = max(1, int(round(len(cls_trials) * val_trial_ratio)))

        # 防止极端情况下把某一类全部抽到验证集
        if n_val_cls >= len(cls_trials):
            n_val_cls = max(1, len(cls_trials) - 1)

        chosen = rng.choice(
            cls_trials,
            size=n_val_cls,
            replace=False
        )

        val_trials.extend(chosen.tolist())

    val_trials = np.asarray(sorted(val_trials), dtype=np.int64)

    val_trial_mask = np.isin(trial_ids, val_trials)
    train_trial_mask = ~val_trial_mask

    eeg_train = all_eeg[train_trial_mask]
    fnirs_train = all_fnirs[train_trial_mask]
    label_train = all_label[train_trial_mask]

    eeg_val = all_eeg[val_trial_mask]
    fnirs_val = all_fnirs[val_trial_mask]
    label_val = all_label[val_trial_mask]

    if eeg_train.shape[0] == 0:
        raise ValueError(
            f"test_session={test_session}: trial-level validation 后训练集为空。"
        )

    if eeg_val.shape[0] == 0:
        raise ValueError(
            f"test_session={test_session}: trial-level validation 后验证集为空。"
        )

    # =========================
    # 4. 打印检查信息
    # =========================
    print("\n[Trial-level validation split]")
    print("test_session:", test_session)
    print("windows_per_trial:", windows_per_trial)
    print("train candidate samples:", train_total)
    print("train candidate trials:", n_trials)
    print("train candidate trial label count:", np.bincount(trial_labels, minlength=2))
    print("chosen val trials:", val_trials)
    print("chosen val trial label count:", np.bincount(trial_labels[val_trials], minlength=2))

    print("eeg_train:", eeg_train.shape)
    print("fnirs_train:", fnirs_train.shape)
    print("label_train:", label_train.shape)

    print("eeg_val:", eeg_val.shape)
    print("fnirs_val:", fnirs_val.shape)
    print("label_val:", label_val.shape)

    print("eeg_test:", eeg_test.shape)
    print("fnirs_test:", fnirs_test.shape)
    print("label_test:", label_test.shape)

    print("train label count:", np.sum(label_train, axis=0))
    print("val label count:", np.sum(label_val, axis=0))
    print("test label count:", np.sum(label_test, axis=0))

    return (
        eeg_train,
        fnirs_train,
        label_train,
        eeg_val,
        fnirs_val,
        label_val,
        eeg_test,
        fnirs_test,
        label_test,
        all_eeg,
        all_fnirs,
        all_label
    )


# =========================
# 主训练流程
# =========================
subject_list = sorted([f for f in os.listdir(subject_path) if f.endswith('.npz')])

print("\n==============================")
print(f"当前任务: {TASK}")
print(f"model_input 路径: {subject_path}")
print(f"结果 CSV: {csv_file_path}")
print(f"混淆矩阵保存路径: {cm_save_dir}")
print("==============================\n")

for subject in subject_list:
    subject_file = os.path.join(subject_path, subject)

    with np.load(subject_file, allow_pickle=True) as data:
        eeg = data['eeg']
        fnirs = data['fnirs']
        label = data['label']

        if 'sample_session' in data.files:
            sample_session = data['sample_session']
        else:
            raise KeyError(
                f"{subject} 中没有 sample_session。"
                f"请先使用新版 model_input.py 生成 sample_session。"
            )

    sample_session = np.asarray(sample_session).squeeze().astype(np.int64)

    # fNIRS 单位缩放：保持你原来的做法
    fnirs = fnirs * 1e3

    label = label.astype(np.float32)

    check_model_input(
        subject=subject,
        eeg=eeg,
        fnirs=fnirs,
        label=label,
        sample_session=sample_session
    )

    for session in range(3):
        print(f"\n{'=' * 70}")
        print(f"Subject: {subject} | Test session: {session}")
        print(f"{'=' * 70}")

        # =========================
        # 使用 sample_session 做三折 session-wise 划分
        # 并在训练 session 内部按 trial 抽验证集
        # =========================
        (
            eeg_train,
            fnirs_train,
            label_train,
            eeg_val,
            fnirs_val,
            label_val,
            eeg_test,
            fnirs_test,
            label_test,
            all_eeg,
            all_fnirs,
            all_label
        ) = split_train_val_by_trial(
            eeg=eeg,
            fnirs=fnirs,
            label=label,
            sample_session=sample_session,
            test_session=session,
            windows_per_trial=10,
            val_trial_ratio=0.2,
            seed=SEED
        )

        # =========================
        # 构造 Dataset
        # =========================
        first_train_dataset = make_dataset(
            eeg_data=eeg_train,
            fnirs_data=fnirs_train,
            label_data=label_train,
            batch_size=32,
            shuffle=True,
            seed=SEED
        )

        val_dataset = make_dataset(
            eeg_data=eeg_val,
            fnirs_data=fnirs_val,
            label_data=label_val,
            batch_size=32,
            shuffle=False
        )

        test_dataset = make_dataset(
            eeg_data=eeg_test,
            fnirs_data=fnirs_test,
            label_data=label_test,
            batch_size=32,
            shuffle=False
        )

        print("\n[数据划分]")
        print("eeg_train shape:", eeg_train.shape)
        print("fnirs_train shape:", fnirs_train.shape)
        print("label_train shape:", label_train.shape)

        print("eeg_val shape:", eeg_val.shape)
        print("fnirs_val shape:", fnirs_val.shape)
        print("label_val shape:", label_val.shape)

        print("eeg_test shape:", eeg_test.shape)
        print("fnirs_test shape:", fnirs_test.shape)
        print("label_test shape:", label_test.shape)

        # =========================
        # 创建模型
        # =========================
        model = create_and_compile_model()

        # =========================
        # 第一阶段训练
        # =========================
        stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_class_output_loss',
            patience=25,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )

        print('\nbegin first train')

        model.fit(
            first_train_dataset,
            epochs=300,
            verbose=2,
            validation_data=val_dataset,
            callbacks=[stopping]
        )

        # =========================
        # 第二阶段训练
        # =========================
        # 当前保留你原来的第二阶段训练逻辑。
        # 如果你要做“只用第一阶段训练”的实验，可以把下面这一整段注释掉。
        print('\nbegin second train')

        second_train_dataset = make_dataset(
            eeg_data=all_eeg,
            fnirs_data=all_fnirs,
            label_data=all_label,
            batch_size=64,
            shuffle=True,
            seed=SEED
        )

        stopping_second = tf.keras.callbacks.EarlyStopping(
            monitor='val_class_output_loss',
            patience=25,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_class_output_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1
        )

        model.fit(
            second_train_dataset,
            epochs=200,
            verbose=2,
            validation_data=val_dataset,
            callbacks=[stopping_second, reduce_lr]
        )

        # =========================
        # 测试
        # =========================
        print('\nbegin test')

        test_results = model.evaluate(
            test_dataset,
            verbose=2
        )

        metrics_dict = dict(zip(model.metrics_names, test_results))

        # =========================
        # 保存 CSV
        # =========================
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline='') as csv_file:
            fieldnames = ['subject', 'session'] + model.metrics_names
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row_data = {
                'subject': subject,
                'session': session
            }

            row_data.update(metrics_dict)
            writer.writerow(row_data)

        print(f"[CSV] 结果已写入: {csv_file_path}")

        # =========================
        # 混淆矩阵：同时保存 class_output 和 eeg_output
        # =========================
        preds = model.predict(
            test_dataset,
            verbose=0
        )

        # 当前 sta.py 有两个输出：
        # preds[0] = class_output
        # preds[1] = eeg_output
        y_true = np.argmax(label_test, axis=1)

        y_pred_class = np.argmax(preds[0], axis=1)
        y_pred_eeg = np.argmax(preds[1], axis=1)

        print("\n[预测类别分布检查]")
        print("true label count:", np.bincount(y_true, minlength=2))
        print("class_output pred count:", np.bincount(y_pred_class, minlength=2))
        print("eeg_output pred count:", np.bincount(y_pred_eeg, minlength=2))

        class_collapse_ratio = np.max(np.bincount(y_pred_class, minlength=2)) / len(y_pred_class)
        eeg_collapse_ratio = np.max(np.bincount(y_pred_eeg, minlength=2)) / len(y_pred_eeg)

        print("class_output collapse ratio:", class_collapse_ratio)
        print("eeg_output collapse ratio:", eeg_collapse_ratio)

        if class_collapse_ratio >= 0.90:
            print("[警告] class_output 发生明显类别塌缩！")

        if eeg_collapse_ratio >= 0.90:
            print("[警告] eeg_output 发生明显类别塌缩！")

        safe_subject_name = subject.replace('.npz', '')

        class_cm_save_path = os.path.join(
            cm_save_dir,
            f'CM_{safe_subject_name}_Session{session}_class_output.png'
        )

        eeg_cm_save_path = os.path.join(
            cm_save_dir,
            f'CM_{safe_subject_name}_Session{session}_eeg_output.png'
        )

        cm_class = save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred_class,
            title=f'Confusion Matrix: {subject} - Session {session} - class_output',
            save_path=class_cm_save_path
        )

        cm_eeg = save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred_eeg,
            title=f'Confusion Matrix: {subject} - Session {session} - eeg_output',
            save_path=eeg_cm_save_path
        )

        print(f"[{subject} - Session {session}] class_output 混淆矩阵已保存至: {class_cm_save_path}")
        print(cm_class)

        print(f"[{subject} - Session {session}] eeg_output 混淆矩阵已保存至: {eeg_cm_save_path}")
        print(cm_eeg)

print('all done')