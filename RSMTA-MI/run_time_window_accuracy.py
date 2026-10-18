import os
import sys
import random
import csv

# =========================
# 固定随机种子
# =========================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)

# =========================
# TensorFlow CUDA DLL 路径
# 必须放在 import tensorflow 之前
# =========================
conda_env_path = os.path.dirname(sys.executable)
conda_dll_path = os.path.join(conda_env_path, "Library", "bin")

if os.path.exists(conda_dll_path):
    os.environ["PATH"] = conda_dll_path + os.pathsep + os.environ["PATH"]
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(conda_dll_path)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sta import sta_net

np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# 任务配置
# =========================
TASK = "MI"  # 可选: "MI" 或 "MA"

RAW_ROOT = r"F:\学生文件夹\刘强\DataSet\MA+MI"

if TASK == "MI":
    data_root = os.path.join(RAW_ROOT, "MI")
elif TASK == "MA":
    data_root = os.path.join(RAW_ROOT, "MA")
else:
    raise ValueError("TASK 只能设置为 'MI' 或 'MA'")

subject_path = os.path.join(data_root, "model_input")

result_root = os.path.join(
    data_root,
    "TimeWindowAccuracy-RFB+Mambalike-modify1.2"
)
os.makedirs(result_root, exist_ok=True)

time_window_csv_path = os.path.join(
    result_root,
    f"{TASK}_TimeWindow_Accuracy_Records.csv"
)

time_window_summary_csv_path = os.path.join(
    result_root,
    f"{TASK}_TimeWindow_Accuracy_Summary.csv"
)

time_window_fig_png = os.path.join(
    result_root,
    f"{TASK}_TimeWindow_Accuracy.png"
)

time_window_fig_pdf = os.path.join(
    result_root,
    f"{TASK}_TimeWindow_Accuracy.pdf"
)

# =========================
# 主实验被试列表
# subject 05 特殊，这里先排除
# =========================
MAIN_SUBJECT_IDS = list(range(1, 5)) + list(range(6, 30))

# =========================
# 时间窗配置
# EEG: 10 个 3s window
# =========================
NUM_WINDOWS = 10

# =========================
# GPU 配置
# =========================
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n[INFO] 检测到 {len(gpus)} 个 GPU，已设置显存按需增长。")
    except RuntimeError as e:
        print(e)
else:
    print("\n[INFO] 未检测到 GPU，将使用 CPU。")


# =========================
# 工具函数
# =========================
def load_subject_data(subject):
    subject_file = os.path.join(subject_path, subject)

    if not os.path.exists(subject_file):
        print(f"[Warning] 找不到 {subject_file}，跳过。")
        return None

    with np.load(subject_file, allow_pickle=True) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]

        if "sample_session" not in data.files:
            raise KeyError(
                f"{subject} 中没有 sample_session。"
                f"请先使用新版 model_input.py 生成 sample_session。"
            )

        sample_session = data["sample_session"]

        # 如果你的 model_input 里已经保存了 sample_window，就优先使用
        if "sample_window" in data.files:
            sample_window = data["sample_window"]
        else:
            sample_window = None

    sample_session = np.asarray(sample_session).squeeze().astype(np.int64)

    if sample_window is not None:
        sample_window = np.asarray(sample_window).squeeze().astype(np.int64)

    fnirs = fnirs * 1e3
    label = label.astype(np.float32)

    return eeg, fnirs, label, sample_session, sample_window


def infer_sample_window(sample_session, num_windows=10):
    """
    如果 npz 里没有 sample_window，则根据样本顺序推断 window index。

    前提：
        每个 trial 产生连续 10 个 sliding-window samples；
        model_input 中同一 session 内样本按 trial-window 顺序保存。

    输出：
        sample_window: 0~9
    """
    sample_window = np.zeros_like(sample_session, dtype=np.int64)

    for sess in np.unique(sample_session):
        idx = np.where(sample_session == sess)[0]
        n = len(idx)

        if n % num_windows != 0:
            print(
                f"[Warning] session {sess} 样本数 {n} 不能被 {num_windows} 整除，"
                f"仍将按顺序取 modulo 推断 window index。请确认数据顺序是否正确。"
            )

        sample_window[idx] = np.arange(n) % num_windows

    return sample_window


def check_model_input(subject, eeg, fnirs, label, sample_session, sample_window):
    print(f"\n[{subject}] model_input 检查:")
    print("EEG:", eeg.shape)
    print("fNIRS:", fnirs.shape)
    print("label:", label.shape)
    print("sample_session:", sample_session.shape, "unique:", np.unique(sample_session))
    print("sample_window:", sample_window.shape, "unique:", np.unique(sample_window))

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

    n = eeg.shape[0]

    if fnirs.shape[0] != n or label.shape[0] != n:
        raise ValueError(f"{subject}: EEG/fNIRS/label 样本数不一致。")

    if sample_session.shape[0] != n:
        raise ValueError(f"{subject}: sample_session 样本数不一致。")

    if sample_window.shape[0] != n:
        raise ValueError(f"{subject}: sample_window 样本数不一致。")

    valid_sessions = []

    for session in [0, 1, 2]:
        count = np.sum(sample_session == session)
        print(f"session {session} sample 数:", count)
        if count > 0:
            valid_sessions.append(session)
        else:
            raise ValueError(
                f"{subject}: session {session} 没有样本，无法做三折 session-wise。"
            )

    return valid_sessions


def make_dataset(eeg_data, fnirs_data, label_data, batch_size, shuffle=False, seed=SEED):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "eeg_input": eeg_data,
                "fnirs_input": fnirs_data,
            },
            {
                "class_output": label_data,
                "eeg_output": label_data,
            },
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=128,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_and_compile_model():
    tf.keras.backend.clear_session()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = sta_net()

    model.compile(
        loss={
            "class_output": "categorical_crossentropy",
            "eeg_output": "categorical_crossentropy",
        },
        loss_weights={
            "class_output": 1.0,
            "eeg_output": 1.0,
        },
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def append_time_window_records(
    csv_path,
    subject,
    session,
    output_name,
    y_true,
    y_pred,
    sample_window_test,
    num_windows=10,
):
    """
    保存每个 time window 的 accuracy。
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        fieldnames = [
            "task",
            "subject",
            "session",
            "output",
            "window",
            "n_samples",
            "accuracy",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for w in range(num_windows):
            mask = sample_window_test == w

            if np.sum(mask) == 0:
                acc = np.nan
                n_samples = 0
            else:
                acc = accuracy_score(y_true[mask], y_pred[mask])
                n_samples = int(np.sum(mask))

            writer.writerow(
                {
                    "task": TASK,
                    "subject": subject,
                    "session": session,
                    "output": output_name,
                    "window": w + 1,
                    "n_samples": n_samples,
                    "accuracy": acc,
                }
            )


def summarize_and_plot_time_window_accuracy(
    record_csv_path,
    summary_csv_path,
    fig_png_path,
    fig_pdf_path,
    target_output="class_output",
):
    """
    汇总所有 subject/session 的 time-window accuracy，并画平均曲线。

    默认只画 class_output。
    如果想画 eeg_output，把 target_output 改成 "eeg_output"。
    """
    import pandas as pd

    if not os.path.exists(record_csv_path):
        print("[Warning] 没有找到 time-window accuracy 记录文件，无法画图。")
        return

    df = pd.read_csv(record_csv_path)

    df = df[df["output"] == target_output].copy()

    if df.empty:
        print(f"[Warning] 没有 {target_output} 的记录，无法画图。")
        return

    summary = (
        df.groupby("window")["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"[Summary CSV] 已保存: {summary_csv_path}")

    x = summary["window"].values
    y = summary["mean"].values
    yerr = summary["std"].values

    plt.figure(figsize=(6.5, 4.2))

    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        label="RSMTA-Net"
    )

    plt.fill_between(
        x,
        y - yerr,
        y + yerr,
        alpha=0.15
    )

    plt.xlabel("Time Window")
    plt.ylabel("Accuracy")
    plt.title(f"{TASK} Task")
    plt.xticks(np.arange(1, NUM_WINDOWS + 1))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(fig_png_path, dpi=600, bbox_inches="tight")
    plt.savefig(fig_pdf_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[Figure PNG] 已保存: {fig_png_path}")
    print(f"[Figure PDF] 已保存: {fig_pdf_path}")


def train_and_test_one_subject(subject):
    loaded = load_subject_data(subject)

    if loaded is None:
        return

    eeg, fnirs, label, sample_session, sample_window = loaded

    if sample_window is None:
        print(f"[Info] {subject}: 未找到 sample_window，将根据样本顺序自动推断。")
        sample_window = infer_sample_window(
            sample_session=sample_session,
            num_windows=NUM_WINDOWS,
        )

    valid_sessions = check_model_input(
        subject=subject,
        eeg=eeg,
        fnirs=fnirs,
        label=label,
        sample_session=sample_session,
        sample_window=sample_window,
    )

    for session in valid_sessions:
        print(f"\n{'=' * 70}")
        print(f"Subject: {subject} | Test session: {session}")
        print(f"{'=' * 70}")

        test_mask = sample_session == session
        train_mask = sample_session != session

        eeg_test = eeg[test_mask]
        fnirs_test = fnirs[test_mask]
        label_test = label[test_mask]
        sample_window_test = sample_window[test_mask]

        all_eeg = eeg[train_mask]
        all_fnirs = fnirs[train_mask]
        all_label = label[train_mask]

        train_total = all_eeg.shape[0]

        if train_total >= 100:
            val_size = 80
        else:
            val_size = max(1, int(train_total * 0.2))

        np.random.seed(SEED)

        indices = np.random.choice(
            train_total,
            size=val_size,
            replace=False,
        )

        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]

        eeg_train = np.delete(all_eeg, indices, axis=0)
        fnirs_train = np.delete(all_fnirs, indices, axis=0)
        label_train = np.delete(all_label, indices, axis=0)

        print("\n[数据划分]")
        print("eeg_train:", eeg_train.shape)
        print("fnirs_train:", fnirs_train.shape)
        print("label_train:", label_train.shape)
        print("eeg_val:", eeg_val.shape)
        print("fnirs_val:", fnirs_val.shape)
        print("label_val:", label_val.shape)
        print("eeg_test:", eeg_test.shape)
        print("fnirs_test:", fnirs_test.shape)
        print("label_test:", label_test.shape)
        print("test window count:", np.bincount(sample_window_test, minlength=NUM_WINDOWS))

        first_train_dataset = make_dataset(
            eeg_train,
            fnirs_train,
            label_train,
            batch_size=32,
            shuffle=True,
            seed=SEED,
        )

        val_dataset = make_dataset(
            eeg_val,
            fnirs_val,
            label_val,
            batch_size=32,
            shuffle=False,
        )

        second_train_dataset = make_dataset(
            all_eeg,
            all_fnirs,
            all_label,
            batch_size=64,
            shuffle=True,
            seed=SEED,
        )

        test_dataset = make_dataset(
            eeg_test,
            fnirs_test,
            label_test,
            batch_size=32,
            shuffle=False,
        )

        model = create_and_compile_model()

        stopping_first = tf.keras.callbacks.EarlyStopping(
            monitor="val_class_output_loss",
            patience=25,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        )

        print("\nbegin first train")

        model.fit(
            first_train_dataset,
            epochs=300,
            verbose=2,
            validation_data=val_dataset,
            callbacks=[stopping_first],
        )

        stopping_second = tf.keras.callbacks.EarlyStopping(
            monitor="val_class_output_loss",
            patience=25,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_class_output_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        )

        print("\nbegin second train")

        model.fit(
            second_train_dataset,
            epochs=200,
            verbose=2,
            validation_data=val_dataset,
            callbacks=[stopping_second, reduce_lr],
        )

        print("\nbegin test")

        preds = model.predict(
            test_dataset,
            verbose=0,
        )

        y_true = np.argmax(label_test, axis=1)
        y_pred_class = np.argmax(preds[0], axis=1)
        y_pred_eeg = np.argmax(preds[1], axis=1)

        print("[Overall class_output acc]:", accuracy_score(y_true, y_pred_class))
        print("[Overall eeg_output acc]:", accuracy_score(y_true, y_pred_eeg))

        append_time_window_records(
            csv_path=time_window_csv_path,
            subject=subject,
            session=session,
            output_name="class_output",
            y_true=y_true,
            y_pred=y_pred_class,
            sample_window_test=sample_window_test,
            num_windows=NUM_WINDOWS,
        )

        append_time_window_records(
            csv_path=time_window_csv_path,
            subject=subject,
            session=session,
            output_name="eeg_output",
            y_true=y_true,
            y_pred=y_pred_eeg,
            sample_window_test=sample_window_test,
            num_windows=NUM_WINDOWS,
        )

        print(f"[Time-window CSV] 已更新: {time_window_csv_path}")


# =========================
# 主流程
# =========================
print("\n==============================")
print(f"当前任务: {TASK}")
print(f"model_input 路径: {subject_path}")
print(f"time-window 结果 CSV: {time_window_csv_path}")
print(f"time-window 图保存路径: {result_root}")
print("==============================\n")

# 如果已有旧结果，建议删除，避免重复追加
if os.path.exists(time_window_csv_path):
    print(f"[Warning] 发现旧 CSV，将删除后重新生成: {time_window_csv_path}")
    os.remove(time_window_csv_path)

for sub_id in MAIN_SUBJECT_IDS:
    subject = f"subject {sub_id:02d}.npz"
    train_and_test_one_subject(subject)

# 汇总并画图
summarize_and_plot_time_window_accuracy(
    record_csv_path=time_window_csv_path,
    summary_csv_path=time_window_summary_csv_path,
    fig_png_path=time_window_fig_png,
    fig_pdf_path=time_window_fig_pdf,
    target_output="class_output",
)

print("all done")

