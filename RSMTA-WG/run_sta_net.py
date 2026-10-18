import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras


# 【记录日志配置】将所有 print 和训练输出备份到指定的 txt 文件
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        # 使用 utf-8 编码追加写入，避免覆盖之前的结果
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 实时将缓冲写入文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# 替换系统标准输出
sys.stdout = Logger(r"F:\学生文件夹\刘强\DataSet\WG\WG-SRFB+SSE_EGTA.txt")

# 【核心配置区】强制让 TensorFlow 找到 conda 虚拟环境里装好的 CUDA DLL 库
conda_env_path = os.path.dirname(sys.executable)
conda_dll_path = os.path.join(conda_env_path, 'Library', 'bin')
if os.path.exists(conda_dll_path):
    # 兼容老版找库方式（加到 PATH 环境变量开头）
    os.environ['PATH'] = conda_dll_path + os.pathsep + os.environ['PATH']
    # 兼容 Python 3.8+ 找库方式（直接注册动态加载目录以防止安全阻断）
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(conda_dll_path)
from sta import sta_net

# 自动检测 GPU 并进行配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置显存按需分配，避免 TensorFlow 占用全部显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n[INFO] 成功检测到 {len(gpus)} 个 GPU（已配置显存按需增长），将使用 GPU 进行训练。")
    except RuntimeError as e:
        print(e)
else:
    print("\n[INFO] 未检测到 GPU，将使用 CPU 进行训练。")


class targetacccallback(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super().__init__()

        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs={}):
        if (logs['class_output_loss'] <= self.target_acc):
            print("\nReached target loss value {} so cancelling training!\n".format(self.target_acc))
            self.model.stop_training = True


subject_path = r'F:\学生文件夹\刘强\DataSet\WG\model_input'
subject_list = os.listdir(subject_path)

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        fnirs = data['fnirs']
        label = data['label']

    fnirs *= 1e3

    label = label.astype(float)  # 确保标签类型是浮点数

    for session in range(3):
        all_eeg = np.delete(eeg, slice(session * 200, (session + 1) * 200), 0)
        all_fnirs = np.delete(fnirs, slice(session * 200, (session + 1) * 200), 0)
        all_label = np.delete(label, slice(session * 200, (session + 1) * 200), 0)

        second_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": all_eeg, "fnirs_input": all_fnirs, "label_input": all_label},
                {"class_output": all_label, 'eeg_output': all_label}
            )
        )
        second_train_dataset = second_train_dataset.shuffle(buffer_size=128).batch(64).prefetch(tf.data.AUTOTUNE)

        eeg_test = eeg[session * 200:(session + 1) * 200,]
        fnirs_test = fnirs[session * 200:(session + 1) * 200,]
        label_test = label[session * 200:(session + 1) * 200,]

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_test, "fnirs_input": fnirs_test, "label_input": label_test},
                {"class_output": label_test, 'eeg_output': label_test}
            )
        )
        test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        np.random.seed(42)
        indices = np.random.choice(all_eeg.shape[0], size=80, replace=False)

        eeg_train = np.delete(all_eeg, indices, axis=0)
        fnirs_train = np.delete(all_fnirs, indices, axis=0)
        label_train = np.delete(all_label, indices, axis=0)
        first_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_train, "fnirs_input": fnirs_train, "label_input": label_train},
                {"class_output": label_train, 'eeg_output': label_train}
            )
        )
        first_train_dataset = first_train_dataset.shuffle(buffer_size=128).batch(32).prefetch(tf.data.AUTOTUNE)

        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_val, "fnirs_input": fnirs_val, "label_input": label_val},
                {"class_output": label_val, 'eeg_output': label_val}
            )
        )
        val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        print('eeg_train shape:', eeg_train.shape)
        print('fnirs_train shape:', fnirs_train.shape)
        print('label_train shape:', label_train.shape)

        print('eeg_val shape:', eeg_val.shape)
        print('fnirs_val shape:', fnirs_val.shape)
        print('label_val shape:', label_val.shape)

        print(subject)
        print(session)

        # 这里才开始模型的调用与训练
        tf.keras.backend.clear_session()
        model = sta_net()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        stopping = tf.keras.callbacks.EarlyStopping(monitor='val_class_output_loss', patience=50,
                                                    restore_best_weights=True, verbose=1)

        print('begin first train')
        first_history = model.fit(first_train_dataset, epochs=300,
                                  verbose=2, validation_data=val_dataset,
                                  callbacks=[stopping])

        print('begin second train')
        # 修改1：监控验证集分类损失，缩短早停耐心值避免过拟合
        stopping_second = tf.keras.callbacks.EarlyStopping(
            monitor='val_class_output_loss', patience=25, mode='min',
            restore_best_weights=True, verbose=1)

        # 修改2：增加学习率衰减（LRScheduler），在验证集指标平缓时降低学习率帮助收敛
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_class_output_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1
        )

        # 开始第二次训练
        model.fit(second_train_dataset, epochs=200,
                  verbose=2, validation_data=val_dataset, callbacks=[stopping_second, reduce_lr])

        print('begin test')
        test_results = model.evaluate(test_dataset)

        # ------------------ 新增的记录代码 ------------------
        import csv

        metrics_dict = dict(zip(model.metrics_names, test_results))

        # 定义验证结果 CSV 保存路径
        csv_file_path = r"F:\学生文件夹\刘强\DataSet\WG\WG-SRFB+SSE_EGTA.csv"
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline='') as csv_file:
            fieldnames = ['subject', 'session'] + model.metrics_names
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row_data = {'subject': subject, 'session': session}
            row_data.update(metrics_dict)
            writer.writerow(row_data)
        # ----------------------------------------------------
        # ------------------ 新增的混淆矩阵代码 ------------------
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        # 1. 获取模型预测
        # 因为模型输出了两个结果 [class_output, eeg_output]，所以 preds[0] 就是分类层 class_output
        preds = model.predict(test_dataset, verbose=0)
        y_pred = np.argmax(preds[0], axis=1)

        # 2. 拿到真实标签
        # label_test是独热编码或具有概率分布，将其转回具体的类别索引
        y_true = np.argmax(label_test, axis=1)

        # 3. 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 4. 设置混淆矩阵图片的保存路径
        cm_save_dir = r"F:\学生文件夹\刘强\DataSet\WG\WG-SRFB+SSE_EGTA"
        if not os.path.exists(cm_save_dir):
            os.makedirs(cm_save_dir)

        plt.figure(figsize=(8, 6))

        # 使用 seaborn 的热力图来进行可视化绘制
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix: {subject} - Session {session}')

        # 拼接文件名并保存高分辨率图像 (dpi=300)
        save_path = os.path.join(cm_save_dir, f'CM_{subject}_Session{session}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[{subject} - Session {session}] 混淆矩阵图片已保存至: {save_path}")
        # ----------------------------------------------------

print('all done')
