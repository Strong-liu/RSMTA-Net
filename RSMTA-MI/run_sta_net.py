import os
import sys
import random
import csv
import time
from datetime import datetime

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
TASK = "MI"

MODEL_NAME = "RSMTA-Net"

RAW_ROOT = r'F:\学生文件夹\刘强\DataSet\MA+MI'

if TASK == "MI":
    data_root = os.path.join(RAW_ROOT, 'MI')
elif TASK == "MA":
    data_root = os.path.join(RAW_ROOT, 'MA')
else:
    raise ValueError("TASK 只能设置为 'MI' 或 'MA'")

subject_path = os.path.join(data_root, 'model_input')

csv_file_path = os.path.join(
    data_root,
    'Compare_Results-RFB+Mambalike-modify1.2.csv'
)

complexity_csv_file_path = os.path.join(
    data_root,
    'Model_Complexity-RFB+Mambalike-modify1.2.csv'
)

complexity_summary_csv_file_path = os.path.join(
    data_root,
    'Model_Complexity-Summary-RFB+Mambalike-modify1.2.csv'
)

cm_save_dir = os.path.join(
    data_root,
    'ConfusionMatrices-RFB+Mambalike-modify1.2'
)

if not os.path.exists(cm_save_dir):
    os.makedirs(cm_save_dir)


# =========================
# 时间统计配置
# =========================
# 说明：
#   1. 训练时间：
#       保存每个训练 batch 的平均耗时 avg_train_batch_time_ms
#       同时保存每个 epoch 的平均 fit 墙钟时间 avg_fit_time_per_epoch_ms
#
#   2. 推理时间：
#       保存每个测试 sample 的平均推理耗时 avg_infer_time_per_sample_ms
#       同时保存每个测试 batch 的平均推理耗时 avg_infer_time_per_batch_ms
#
#   3. warmup 的作用：
#       避免第一次 predict 包含图构建、CUDA 初始化等额外开销。
# =========================
INFER_WARMUP_RUNS = 3
INFER_REPEAT_RUNS = 10

FIRST_TRAIN_BATCH_SIZE = 32
SECOND_TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32


# =========================
# 本次运行的复杂度记录
# 用于最后汇总成 Summary CSV
# =========================
COMPLEXITY_RECORDS_THIS_RUN = []


# =========================
# 主实验被试列表
# =========================
# 主实验只跑:
# subject 01, 02, 03, 04, 06, 07, ..., 29
# subject 05 因 session 0 无有效样本，放到底部单独两折代码块中处理。
# =========================
MAIN_SUBJECT_IDS = list(range(1, 5)) + list(range(6, 30))


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


def get_runtime_device_info():
    """
    获取当前运行设备信息，用于写入复杂度 CSV。
    """
    gpus_local = tf.config.list_physical_devices('GPU')

    if gpus_local:
        gpu_names = []

        for gpu in gpus_local:
            try:
                details = tf.config.experimental.get_device_details(gpu)
                gpu_name = details.get('device_name', str(gpu))
            except Exception:
                gpu_name = str(gpu)

            gpu_names.append(gpu_name)

        return "GPU: " + " | ".join(gpu_names)

    return "CPU"


RUNTIME_DEVICE_INFO = get_runtime_device_info()


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


class TrainTimeCallback(keras.callbacks.Callback):
    """
    统计训练过程中的 batch 时间和 epoch 时间。

    注意：
        on_train_batch_begin / on_train_batch_end 统计的是训练 batch 时间，
        不包含 validation 的时间。

        on_epoch_begin / on_epoch_end 统计的是 epoch 时间，
        包含该 epoch 内部训练部分，但不一定完全等价于 model.fit 的外部墙钟时间。
        因此代码中还额外用 time.perf_counter() 统计 model.fit 的总墙钟时间。
    """
    def __init__(self, stage_name):
        super().__init__()
        self.stage_name = stage_name
        self.batch_times = []
        self.epoch_times = []
        self._batch_start_time = None
        self._epoch_start_time = None

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        if self._batch_start_time is not None:
            elapsed_time = time.perf_counter() - self._batch_start_time
            self.batch_times.append(elapsed_time)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if self._epoch_start_time is not None:
            elapsed_time = time.perf_counter() - self._epoch_start_time
            self.epoch_times.append(elapsed_time)

    def get_avg_batch_time_ms(self):
        if len(self.batch_times) == 0:
            return np.nan
        return float(np.mean(self.batch_times) * 1000.0)

    def get_std_batch_time_ms(self):
        if len(self.batch_times) == 0:
            return np.nan
        return float(np.std(self.batch_times) * 1000.0)

    def get_avg_epoch_time_ms(self):
        if len(self.epoch_times) == 0:
            return np.nan
        return float(np.mean(self.epoch_times) * 1000.0)

    def get_std_epoch_time_ms(self):
        if len(self.epoch_times) == 0:
            return np.nan
        return float(np.std(self.epoch_times) * 1000.0)


# =========================
# 工具函数
# =========================
def check_model_input(subject, eeg, fnirs, label, sample_session, allow_empty_session=False):
    """
    检查 model_input 文件中的数据是否符合训练要求。

    allow_empty_session=False:
        主实验模式。任意 session 为空都会报错。

    allow_empty_session=True:
        特例模式。允许某些 session 为空，并返回有效 session 列表。
        主要用于 subject 05 的两折 session-wise 验证。
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

    valid_sessions = []

    for session in [0, 1, 2]:
        session_count = np.sum(sample_session == session)
        print(f"session {session} sample 数:", session_count)

        if session_count > 0:
            valid_sessions.append(session)
        else:
            if allow_empty_session:
                print(
                    f"[Warning] {subject}: session {session} 没有样本，"
                    f"该 session 将不会作为测试折。"
                )
            else:
                raise ValueError(
                    f"{subject}: session {session} 没有样本，无法做三折 session-wise 测试。"
                )

    if allow_empty_session and len(valid_sessions) < 2:
        raise ValueError(
            f"{subject}: 有效 session 少于 2 个，无法进行 session-wise 交叉验证。"
        )

    return valid_sessions


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


def count_model_parameters(model):
    """
    统计模型参数量。

    返回：
        total_params:
            模型总参数量。
        trainable_params:
            可训练参数量。
        non_trainable_params:
            不可训练参数量。
    """
    trainable_params = int(
        np.sum([keras.backend.count_params(w) for w in model.trainable_weights])
    )

    non_trainable_params = int(
        np.sum([keras.backend.count_params(w) for w in model.non_trainable_weights])
    )

    total_params = int(model.count_params())

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params
    }


def get_dataset_num_batches(dataset):
    """
    获取 Dataset 的 batch 数量。
    优先使用 cardinality，失败时手动遍历。
    """
    cardinality = tf.data.experimental.cardinality(dataset).numpy()

    if cardinality >= 0:
        return int(cardinality)

    count = 0
    for _ in dataset:
        count += 1

    return count


def measure_inference_time(
    model,
    test_dataset,
    test_samples,
    warmup_runs=3,
    repeat_runs=10
):
    """
    统计模型推理时间，并返回最后一次预测结果。

    统计方式：
        1. 先 warm-up 若干次，不计入最终时间。
        2. 重复 predict 多次。
        3. 计算平均总推理时间。
        4. 换算成：
            avg_infer_time_per_batch_ms
            avg_infer_time_per_sample_ms
    """
    test_batches = get_dataset_num_batches(test_dataset)

    print("\n[推理时间统计]")
    print("warmup_runs:", warmup_runs)
    print("repeat_runs:", repeat_runs)
    print("test_samples:", test_samples)
    print("test_batches:", test_batches)

    # warm-up
    for _ in range(warmup_runs):
        _ = model.predict(
            test_dataset,
            verbose=0
        )

    infer_times = []
    preds = None

    for run_idx in range(repeat_runs):
        start_time = time.perf_counter()

        preds = model.predict(
            test_dataset,
            verbose=0
        )

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        infer_times.append(elapsed_time)

        print(f"inference run {run_idx + 1}/{repeat_runs}: {elapsed_time:.6f} s")

    infer_times = np.asarray(infer_times, dtype=np.float64)

    mean_total_infer_time_s = float(np.mean(infer_times))
    std_total_infer_time_s = float(np.std(infer_times))

    avg_infer_time_per_batch_ms = float(mean_total_infer_time_s / test_batches * 1000.0)
    std_infer_time_per_batch_ms = float(std_total_infer_time_s / test_batches * 1000.0)

    avg_infer_time_per_sample_ms = float(mean_total_infer_time_s / test_samples * 1000.0)
    std_infer_time_per_sample_ms = float(std_total_infer_time_s / test_samples * 1000.0)

    print("mean_total_infer_time_s:", mean_total_infer_time_s)
    print("std_total_infer_time_s:", std_total_infer_time_s)
    print("avg_infer_time_per_batch_ms:", avg_infer_time_per_batch_ms)
    print("avg_infer_time_per_sample_ms:", avg_infer_time_per_sample_ms)

    return {
        "preds": preds,
        "test_batches": test_batches,
        "mean_total_infer_time_s": mean_total_infer_time_s,
        "std_total_infer_time_s": std_total_infer_time_s,
        "avg_infer_time_per_batch_ms": avg_infer_time_per_batch_ms,
        "std_infer_time_per_batch_ms": std_infer_time_per_batch_ms,
        "avg_infer_time_per_sample_ms": avg_infer_time_per_sample_ms,
        "std_infer_time_per_sample_ms": std_infer_time_per_sample_ms
    }


def append_row_to_csv(csv_path, row_data, fieldnames):
    """
    向 CSV 追加一行。
    如果文件不存在或为空，则先写表头。
    """
    file_exists = os.path.isfile(csv_path)
    file_is_empty = (not file_exists) or os.path.getsize(csv_path) == 0

    with open(csv_path, mode='a', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if file_is_empty:
            writer.writeheader()

        writer.writerow(row_data)


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


def load_subject_data(subject):
    """
    读取单个 subject 的 model_input 数据。
    """
    subject_file = os.path.join(subject_path, subject)

    if not os.path.exists(subject_file):
        print(f"[Warning] 找不到 {subject_file}，跳过。")
        return None

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

    return eeg, fnirs, label, sample_session


def summarize_complexity_records(records, summary_save_path):
    """
    对本次运行的 complexity 记录求平均，保存为 summary CSV。

    注意：
        这里汇总的是本次脚本运行期间产生的记录，
        不会把历史 CSV 中旧的记录混进来。
    """
    if len(records) == 0:
        print("[Warning] 本次运行没有 complexity 记录，无法生成 summary。")
        return

    def mean_std(field):
        values = []

        for record in records:
            value = record.get(field, None)

            if value is None:
                continue

            try:
                value = float(value)
            except Exception:
                continue

            if np.isnan(value):
                continue

            values.append(value)

        if len(values) == 0:
            return np.nan, np.nan

        values = np.asarray(values, dtype=np.float64)

        return float(np.mean(values)), float(np.std(values))

    first_record = records[0]

    mean_avg_train_batch_time_ms, std_avg_train_batch_time_ms = mean_std(
        "avg_train_batch_time_ms"
    )

    mean_avg_fit_time_per_epoch_ms, std_avg_fit_time_per_epoch_ms = mean_std(
        "avg_fit_time_per_epoch_ms"
    )

    mean_avg_infer_time_per_batch_ms, std_avg_infer_time_per_batch_ms = mean_std(
        "avg_infer_time_per_batch_ms"
    )

    mean_avg_infer_time_per_sample_ms, std_avg_infer_time_per_sample_ms = mean_std(
        "avg_infer_time_per_sample_ms"
    )

    mean_total_fit_wall_time_s, std_total_fit_wall_time_s = mean_std(
        "total_fit_wall_time_s"
    )

    mean_total_epochs, std_total_epochs = mean_std(
        "total_epochs"
    )

    summary_row = {
        "task": TASK,
        "model": MODEL_NAME,
        "n_folds": len(records),

        "parameters": first_record["parameters"],
        "trainable_parameters": first_record["trainable_parameters"],
        "non_trainable_parameters": first_record["non_trainable_parameters"],

        "mean_avg_train_batch_time_ms": mean_avg_train_batch_time_ms,
        "std_avg_train_batch_time_ms": std_avg_train_batch_time_ms,

        "mean_avg_fit_time_per_epoch_ms": mean_avg_fit_time_per_epoch_ms,
        "std_avg_fit_time_per_epoch_ms": std_avg_fit_time_per_epoch_ms,

        "mean_avg_infer_time_per_batch_ms": mean_avg_infer_time_per_batch_ms,
        "std_avg_infer_time_per_batch_ms": std_avg_infer_time_per_batch_ms,

        "mean_avg_infer_time_per_sample_ms": mean_avg_infer_time_per_sample_ms,
        "std_avg_infer_time_per_sample_ms": std_avg_infer_time_per_sample_ms,

        "mean_total_fit_wall_time_s": mean_total_fit_wall_time_s,
        "std_total_fit_wall_time_s": std_total_fit_wall_time_s,

        "mean_total_epochs": mean_total_epochs,
        "std_total_epochs": std_total_epochs,

        "first_train_batch_size": FIRST_TRAIN_BATCH_SIZE,
        "second_train_batch_size": SECOND_TRAIN_BATCH_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,

        "infer_warmup_runs": INFER_WARMUP_RUNS,
        "infer_repeat_runs": INFER_REPEAT_RUNS,

        "device_info": RUNTIME_DEVICE_INFO,
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    summary_fieldnames = list(summary_row.keys())

    with open(summary_save_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"\n[Summary CSV] 模型复杂度汇总已保存至: {summary_save_path}")
    print("\n[本次运行复杂度汇总]")
    print("model:", MODEL_NAME)
    print("n_folds:", len(records))
    print("parameters:", first_record["parameters"])
    print("mean_avg_train_batch_time_ms:", mean_avg_train_batch_time_ms)
    print("mean_avg_fit_time_per_epoch_ms:", mean_avg_fit_time_per_epoch_ms)
    print("mean_avg_infer_time_per_batch_ms:", mean_avg_infer_time_per_batch_ms)
    print("mean_avg_infer_time_per_sample_ms:", mean_avg_infer_time_per_sample_ms)


def train_and_test_one_subject(
    subject,
    sessions_to_run,
    allow_empty_session=False,
    experiment_note=""
):
    """
    对单个 subject 进行 session-wise 训练与测试。

    subject:
        文件名，例如 'subject 01.npz'

    sessions_to_run:
        测试 session 列表。
        普通被试为 [0, 1, 2]。
        subject 05 特例为 [1, 2]。

    allow_empty_session:
        是否允许空 session。
        subject 05 设置为 True。

    experiment_note:
        打印用的备注。
    """
    loaded = load_subject_data(subject)

    if loaded is None:
        return

    eeg, fnirs, label, sample_session = loaded

    valid_sessions = check_model_input(
        subject=subject,
        eeg=eeg,
        fnirs=fnirs,
        label=label,
        sample_session=sample_session,
        allow_empty_session=allow_empty_session
    )

    # 确保 sessions_to_run 只包含有效 session
    sessions_to_run = [s for s in sessions_to_run if s in valid_sessions]

    if len(sessions_to_run) == 0:
        print(f"[Warning] {subject}: 没有可运行的测试 session，跳过。")
        return

    print(f"\n{subject} 本次运行测试 sessions:", sessions_to_run)
    if experiment_note:
        print("实验备注:", experiment_note)

    for session in sessions_to_run:
        print(f"\n{'=' * 70}")
        print(f"Subject: {subject} | Test session: {session}")
        if experiment_note:
            print(f"Note: {experiment_note}")
        print(f"{'=' * 70}")

        # =========================
        # 使用 sample_session 做 session-wise 划分
        # =========================
        test_mask = sample_session == session
        train_mask = sample_session != session

        eeg_test = eeg[test_mask]
        fnirs_test = fnirs[test_mask]
        label_test = label[test_mask]

        all_eeg = eeg[train_mask]
        all_fnirs = fnirs[train_mask]
        all_label = label[train_mask]

        if eeg_test.shape[0] == 0:
            raise ValueError(
                f"{subject} session {session}: 测试集为空，请检查 sample_session。"
            )

        if all_eeg.shape[0] == 0:
            raise ValueError(
                f"{subject} session {session}: 训练集为空，请检查 sample_session。"
            )

        # =========================
        # 从训练集内部划分验证集
        # =========================
        train_total = all_eeg.shape[0]

        if train_total >= 100:
            val_size = 80
        else:
            val_size = max(1, int(train_total * 0.2))

        np.random.seed(SEED)

        indices = np.random.choice(
            train_total,
            size=val_size,
            replace=False
        )

        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]

        eeg_train = np.delete(all_eeg, indices, axis=0)
        fnirs_train = np.delete(all_fnirs, indices, axis=0)
        label_train = np.delete(all_label, indices, axis=0)

        if eeg_train.shape[0] == 0:
            raise ValueError(
                f"{subject} session {session}: 训练集在划分验证集后为空，请减小 val_size。"
            )

        # =========================
        # 构造 Dataset
        # =========================
        first_train_dataset = make_dataset(
            eeg_data=eeg_train,
            fnirs_data=fnirs_train,
            label_data=label_train,
            batch_size=FIRST_TRAIN_BATCH_SIZE,
            shuffle=True,
            seed=SEED
        )

        val_dataset = make_dataset(
            eeg_data=eeg_val,
            fnirs_data=fnirs_val,
            label_data=label_val,
            batch_size=FIRST_TRAIN_BATCH_SIZE,
            shuffle=False
        )

        test_dataset = make_dataset(
            eeg_data=eeg_test,
            fnirs_data=fnirs_test,
            label_data=label_test,
            batch_size=TEST_BATCH_SIZE,
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

        print("train_total:", train_total)
        print("val_size:", val_size)

        print("train label count:", np.sum(label_train, axis=0))
        print("val label count:", np.sum(label_val, axis=0))
        print("test label count:", np.sum(label_test, axis=0))

        # =========================
        # 创建模型
        # =========================
        model = create_and_compile_model()

        # =========================
        # 统计模型参数量
        # =========================
        params_info = count_model_parameters(model)

        print("\n[模型参数量统计]")
        print("Total parameters:", params_info["total_params"])
        print("Trainable parameters:", params_info["trainable_params"])
        print("Non-trainable parameters:", params_info["non_trainable_params"])

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

        first_train_time_callback = TrainTimeCallback(stage_name="first_train")

        print('\nbegin first train')

        first_fit_start_time = time.perf_counter()

        history_first = model.fit(
            first_train_dataset,
            epochs=300,
            verbose=2,
            validation_data=val_dataset,
            callbacks=[stopping, first_train_time_callback]
        )

        first_fit_end_time = time.perf_counter()
        first_fit_wall_time_s = float(first_fit_end_time - first_fit_start_time)

        first_epochs = len(history_first.history.get("loss", []))
        first_avg_train_batch_time_ms = first_train_time_callback.get_avg_batch_time_ms()
        first_std_train_batch_time_ms = first_train_time_callback.get_std_batch_time_ms()
        first_avg_epoch_callback_time_ms = first_train_time_callback.get_avg_epoch_time_ms()

        print("\n[第一阶段训练时间统计]")
        print("first_epochs:", first_epochs)
        print("first_fit_wall_time_s:", first_fit_wall_time_s)
        print("first_avg_train_batch_time_ms:", first_avg_train_batch_time_ms)
        print("first_std_train_batch_time_ms:", first_std_train_batch_time_ms)
        print("first_avg_epoch_callback_time_ms:", first_avg_epoch_callback_time_ms)

        # =========================
        # 第二阶段训练
        # =========================
        print('\nbegin second train')

        second_train_dataset = make_dataset(
            eeg_data=all_eeg,
            fnirs_data=all_fnirs,
            label_data=all_label,
            batch_size=SECOND_TRAIN_BATCH_SIZE,
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

        second_train_time_callback = TrainTimeCallback(stage_name="second_train")

        second_fit_start_time = time.perf_counter()

        history_second = model.fit(
            second_train_dataset,
            epochs=200,
            verbose=2,
            validation_data=val_dataset,
            callbacks=[stopping_second, reduce_lr, second_train_time_callback]
        )

        second_fit_end_time = time.perf_counter()
        second_fit_wall_time_s = float(second_fit_end_time - second_fit_start_time)

        second_epochs = len(history_second.history.get("loss", []))
        second_avg_train_batch_time_ms = second_train_time_callback.get_avg_batch_time_ms()
        second_std_train_batch_time_ms = second_train_time_callback.get_std_batch_time_ms()
        second_avg_epoch_callback_time_ms = second_train_time_callback.get_avg_epoch_time_ms()

        print("\n[第二阶段训练时间统计]")
        print("second_epochs:", second_epochs)
        print("second_fit_wall_time_s:", second_fit_wall_time_s)
        print("second_avg_train_batch_time_ms:", second_avg_train_batch_time_ms)
        print("second_std_train_batch_time_ms:", second_std_train_batch_time_ms)
        print("second_avg_epoch_callback_time_ms:", second_avg_epoch_callback_time_ms)

        # =========================
        # 训练时间汇总
        # =========================
        total_fit_wall_time_s = float(first_fit_wall_time_s + second_fit_wall_time_s)
        total_epochs = int(first_epochs + second_epochs)

        all_train_batch_times = (
            first_train_time_callback.batch_times
            + second_train_time_callback.batch_times
        )

        if len(all_train_batch_times) > 0:
            avg_train_batch_time_ms = float(np.mean(all_train_batch_times) * 1000.0)
            std_train_batch_time_ms = float(np.std(all_train_batch_times) * 1000.0)
        else:
            avg_train_batch_time_ms = np.nan
            std_train_batch_time_ms = np.nan

        if total_epochs > 0:
            avg_fit_time_per_epoch_ms = float(total_fit_wall_time_s / total_epochs * 1000.0)
        else:
            avg_fit_time_per_epoch_ms = np.nan

        print("\n[总训练时间统计]")
        print("total_epochs:", total_epochs)
        print("total_fit_wall_time_s:", total_fit_wall_time_s)
        print("avg_train_batch_time_ms:", avg_train_batch_time_ms)
        print("std_train_batch_time_ms:", std_train_batch_time_ms)
        print("avg_fit_time_per_epoch_ms:", avg_fit_time_per_epoch_ms)

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
        # 保存原始结果 CSV
        # =========================
        result_fieldnames = ['subject', 'session'] + model.metrics_names

        row_data = {
            'subject': subject,
            'session': session
        }

        row_data.update(metrics_dict)

        append_row_to_csv(
            csv_path=csv_file_path,
            row_data=row_data,
            fieldnames=result_fieldnames
        )

        print(f"[CSV] 结果已写入: {csv_file_path}")

        # =========================
        # 推理时间统计 + 获得预测结果
        # =========================
        inference_info = measure_inference_time(
            model=model,
            test_dataset=test_dataset,
            test_samples=eeg_test.shape[0],
            warmup_runs=INFER_WARMUP_RUNS,
            repeat_runs=INFER_REPEAT_RUNS
        )

        preds = inference_info["preds"]

        # =========================
        # 保存复杂度 CSV
        # =========================
        complexity_record = {
            "task": TASK,
            "model": MODEL_NAME,
            "subject": subject,
            "session": session,

            "parameters": params_info["total_params"],
            "trainable_parameters": params_info["trainable_params"],
            "non_trainable_parameters": params_info["non_trainable_params"],

            "first_train_batch_size": FIRST_TRAIN_BATCH_SIZE,
            "second_train_batch_size": SECOND_TRAIN_BATCH_SIZE,
            "test_batch_size": TEST_BATCH_SIZE,

            "first_epochs": first_epochs,
            "second_epochs": second_epochs,
            "total_epochs": total_epochs,

            "first_fit_wall_time_s": first_fit_wall_time_s,
            "second_fit_wall_time_s": second_fit_wall_time_s,
            "total_fit_wall_time_s": total_fit_wall_time_s,

            "first_avg_train_batch_time_ms": first_avg_train_batch_time_ms,
            "first_std_train_batch_time_ms": first_std_train_batch_time_ms,
            "second_avg_train_batch_time_ms": second_avg_train_batch_time_ms,
            "second_std_train_batch_time_ms": second_std_train_batch_time_ms,

            "avg_train_batch_time_ms": avg_train_batch_time_ms,
            "std_train_batch_time_ms": std_train_batch_time_ms,
            "avg_fit_time_per_epoch_ms": avg_fit_time_per_epoch_ms,

            "infer_warmup_runs": INFER_WARMUP_RUNS,
            "infer_repeat_runs": INFER_REPEAT_RUNS,
            "infer_time_total_s_mean": inference_info["mean_total_infer_time_s"],
            "infer_time_total_s_std": inference_info["std_total_infer_time_s"],

            "test_samples": eeg_test.shape[0],
            "test_batches": inference_info["test_batches"],

            "avg_infer_time_per_batch_ms": inference_info["avg_infer_time_per_batch_ms"],
            "std_infer_time_per_batch_ms": inference_info["std_infer_time_per_batch_ms"],
            "avg_infer_time_per_sample_ms": inference_info["avg_infer_time_per_sample_ms"],
            "std_infer_time_per_sample_ms": inference_info["std_infer_time_per_sample_ms"],

            "device_info": RUNTIME_DEVICE_INFO,
            "experiment_note": experiment_note,
            "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        complexity_fieldnames = list(complexity_record.keys())

        append_row_to_csv(
            csv_path=complexity_csv_file_path,
            row_data=complexity_record,
            fieldnames=complexity_fieldnames
        )

        COMPLEXITY_RECORDS_THIS_RUN.append(complexity_record)

        print(f"[Complexity CSV] 复杂度结果已写入: {complexity_csv_file_path}")

        # =========================
        # 混淆矩阵：同时保存 class_output 和 eeg_output
        # =========================
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


# =========================
# 主训练流程：只跑 subject 01-04 和 subject 06-29
# =========================
print("\n==============================")
print(f"当前任务: {TASK}")
print(f"模型名称: {MODEL_NAME}")
print(f"运行设备: {RUNTIME_DEVICE_INFO}")
print(f"model_input 路径: {subject_path}")
print(f"结果 CSV: {csv_file_path}")
print(f"复杂度 CSV: {complexity_csv_file_path}")
print(f"复杂度汇总 CSV: {complexity_summary_csv_file_path}")
print(f"混淆矩阵保存路径: {cm_save_dir}")
print("主实验被试: subject 01-04, subject 06-29")
print("subject 05 已从主实验三折中排除，见底部两折特例代码块。")
print("==============================\n")

# for sub_id in MAIN_SUBJECT_IDS:
#     subject = f"subject {sub_id:02d}.npz"
#
#     train_and_test_one_subject(
#         subject=subject,
#         sessions_to_run=[0, 1, 2],
#         allow_empty_session=False,
#         experiment_note="main_3fold_sessionwise_excluding_subject05"
#     )


# =========================
# subject 05 两折 session-wise 特例代码块
# =========================
# 说明：
#   subject 05 的 session 0 因大段坏段覆盖，预处理后没有有效样本。
#   因此 subject 05 不能做严格三折 session-wise。
#
#   如果你想单独跑 subject 05 的两折 session-wise：
#       fold 1: train session 2, test session 1
#       fold 2: train session 1, test session 2
#
#   请把下面的 if False 改成 if True。
#
# 注意：
#   这个结果不能叫“三折 session-wise”，只能叫“两折 session-wise”。
#   最终统计时建议 subject 05 只对这两个有效 fold 求平均。
# =========================
if True:
    train_and_test_one_subject(
        subject="subject 05.npz",
        sessions_to_run=[1, 2],
        allow_empty_session=True,
        experiment_note="subject05_2fold_sessionwise_due_to_empty_session0"
    )


# =========================
# 汇总本次运行的复杂度结果
# =========================
summarize_complexity_records(
    records=COMPLEXITY_RECORDS_THIS_RUN,
    summary_save_path=complexity_summary_csv_file_path
)

print('all done')