import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


# If you have any questions, please contact us at shulmt@shu.edu.cn


def pearson_r(eeg, fnirs):
    mx = tf.math.reduce_mean(eeg, axis=1, keepdims=True)
    my = tf.math.reduce_mean(fnirs, axis=1, keepdims=True)
    xm, ym = eeg - mx, fnirs - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym), axis=1)
    r_den = tf.math.reduce_std(xm, axis=1) * tf.math.reduce_std(ym, axis=1) + 1e-6
    plcc = r_num / r_den
    plcc = tf.math.abs(plcc)
    plcc_meanbatch = tf.math.reduce_mean(plcc)

    return plcc_meanbatch


class pos_embedding(layers.Layer):
    def __init__(self):
        super(pos_embedding, self).__init__()

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, input_shape[-2], input_shape[-1]),
            initializer=tf.keras.initializers.HeUniform(),
            trainable=True
        )

    def call(self, inputs):
        return inputs + self.pos_embedding


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(
        10000,
        (2 * (i // 2)) / np.float32(d_model)
    )
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class e_f_attention(keras.layers.Layer):
    """
    Mamba-like enhanced EGTA module.

    原始 EGTA:
        EEG 作为 Query
        fNIRS 11 个延迟片段作为 Key / Value
        使用 MultiHeadAttention 选择最相关的 fNIRS 延迟特征

    当前版本:
        1. 先使用 Mamba-like selective state update 对 fNIRS 11 个延迟片段进行动态增强
        2. 再保留原来的 EGTA，即 EEG-Q / enhanced fNIRS-KV attention
        3. Mamba-like 不直接做最终选择，只作为 fNIRS delay feature enhancement
    """

    def __init__(self, emb_size, d_model, heads, drop):
        super(e_f_attention, self).__init__()

        self.emb_size = emb_size
        self.d_model = d_model
        self.heads = heads
        self.drop = drop

        self.q_flat = layers.Flatten()
        self.q_proj = layers.Dense(emb_size)

        self.fusion_proj = layers.Dense(emb_size)

        self.k_flat = layers.Reshape((11, -1))
        self.k_proj = layers.Dense(emb_size)
        self.pos = pos_embedding()

        # ======================= Mamba-like fNIRS 延迟动态增强开始 =======================

        self.fnirs_temporal_conv = layers.Conv1D(
            filters=emb_size,
            kernel_size=3,
            dilation_rate=2,
            padding="causal",
            activation="swish"
        )

        self.eeg_condition_proj = layers.Dense(
            emb_size,
            activation="swish"
        )

        self.delta_proj = layers.Dense(
            emb_size,
            activation="sigmoid"
        )

        self.input_gate_proj = layers.Dense(
            emb_size,
            activation="sigmoid"
        )

        self.output_gate_proj = layers.Dense(
            emb_size,
            activation="sigmoid"
        )

        self.state_candidate_proj = layers.Dense(
            emb_size,
            activation="tanh"
        )

        self.mamba_out_proj = layers.Dense(
            emb_size,
            activation="swish"
        )

        self.mamba_dropout = layers.Dropout(drop)

        self.mamba_norm = layers.LayerNormalization()

        self.mamba_residual_alpha = self.add_weight(
            name="mamba_residual_alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(-2.0),
            trainable=True
        )

        # ======================= Mamba-like fNIRS 延迟动态增强结束 =======================

        self.dot_product_attention = layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=d_model,
            dropout=drop
        )

        self.ef_plcc_tracker = keras.metrics.Mean(name="ef_plcc")

    def call(self, inputs, training=False):
        eeg, fnirs = inputs

        q_eeg_flat = self.q_flat(eeg)

        fusion_output = self.fusion_proj(q_eeg_flat)

        q_eeg_vector = self.q_proj(q_eeg_flat)

        q_eeg = tf.expand_dims(q_eeg_vector, axis=1)

        k_fnirs = self.k_flat(fnirs)

        k_fnirs = self.pos(k_fnirs)

        k_fnirs = self.k_proj(k_fnirs)

        # ======================= Mamba-like 只增强 fNIRS，不替代 EGTA 开始 =======================

        fnirs_seq = self.fnirs_temporal_conv(k_fnirs)

        eeg_condition = self.eeg_condition_proj(q_eeg_vector)
        eeg_condition = tf.expand_dims(eeg_condition, axis=1)

        conditioned_fnirs = fnirs_seq + eeg_condition

        delta = self.delta_proj(conditioned_fnirs)

        input_gate = self.input_gate_proj(conditioned_fnirs)

        output_gate = self.output_gate_proj(conditioned_fnirs)

        state_candidate = self.state_candidate_proj(conditioned_fnirs)

        batch_size = tf.shape(k_fnirs)[0]

        initial_state = tf.zeros(
            shape=(batch_size, self.emb_size),
            dtype=k_fnirs.dtype
        )

        delta_t = tf.transpose(delta, perm=[1, 0, 2])
        input_gate_t = tf.transpose(input_gate, perm=[1, 0, 2])
        candidate_t = tf.transpose(state_candidate, perm=[1, 0, 2])

        def selective_scan_step(prev_state, current_inputs):
            current_delta = current_inputs[0]
            current_input_gate = current_inputs[1]
            current_candidate = current_inputs[2]

            new_state = (
                (1.0 - current_delta) * prev_state
                + current_delta * current_input_gate * current_candidate
            )

            return new_state

        hidden_states = tf.scan(
            fn=selective_scan_step,
            elems=(delta_t, input_gate_t, candidate_t),
            initializer=initial_state
        )

        hidden_states = tf.transpose(hidden_states, perm=[1, 0, 2])

        mamba_enhanced = output_gate * hidden_states

        mamba_enhanced = self.mamba_out_proj(mamba_enhanced)

        mamba_enhanced = self.mamba_dropout(
            mamba_enhanced,
            training=training
        )

        alpha = tf.nn.sigmoid(self.mamba_residual_alpha)

        enhanced_fnirs = self.mamba_norm(
            alpha * mamba_enhanced + (1.0 - alpha) * k_fnirs
        )

        # ======================= Mamba-like 只增强 fNIRS，不替代 EGTA 结束 =======================

        fnirs_weighted, attention_weights = self.dot_product_attention(
            query=q_eeg,
            value=enhanced_fnirs,
            key=enhanced_fnirs,
            return_attention_scores=True,
            training=training
        )

        attention_weights = tf.math.reduce_mean(
            attention_weights,
            axis=(1, 2)
        )

        q_eeg_for_loss = tf.math.reduce_mean(q_eeg, axis=1)

        fnirs_weighted = tf.math.reduce_mean(fnirs_weighted, axis=1)

        ef_loss = pearson_r(q_eeg_for_loss, fnirs_weighted)

        self.add_loss(0.1 * (1 - ef_loss))

        self.ef_plcc_tracker.update_state(ef_loss)

        return fusion_output, fnirs_weighted, attention_weights


class gap(keras.layers.Layer):
    def __init__(self):
        super(gap, self).__init__()

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-2, keepdims=True)


class fga(keras.layers.Layer):
    def __init__(self, tem_kernel_size, fga_loss_name):
        super(fga, self).__init__()

        self.channel_pooling = layers.Conv3D(
            filters=1,
            kernel_size=(3, 3, tem_kernel_size),
            strides=(1, 1, 1),
            padding='same'
        )

        self.tap_fnirs = gap()

        self.residual_para = self.add_weight(
            name='residual_para',
            initializer="zeros",
            trainable=True
        )

        self.add_eeg = layers.Add()
        self.add = layers.Add()

        self.eeg_flatten = layers.Flatten()
        self.fnirs_flatten = layers.Flatten()
        self.fusion_flatten = layers.Flatten()

        self.fga_loss_tracker = keras.metrics.Mean(name=fga_loss_name)

    def call(self, inputs):
        eeg_fusion, eeg, fnirs = inputs

        fnirs_attention = self.channel_pooling(fnirs)

        fnirs_attention_map = self.tap_fnirs(fnirs_attention)
        fnirs_attention_map = tf.math.reduce_mean(fnirs_attention_map, axis=1)

        fnirs_attention_map_norm = keras.activations.sigmoid(
            fnirs_attention_map
        )

        eeg_fusion_guided = tf.math.multiply(
            eeg_fusion,
            fnirs_attention_map_norm
        )

        residual_para_norm = keras.activations.sigmoid(self.residual_para)

        eeg_add = self.add_eeg([
            residual_para_norm * eeg,
            (1 - residual_para_norm) * eeg_fusion
        ])

        fga_feature = self.add([
            eeg_fusion_guided,
            eeg_add
        ])

        eeg_plcc = tf.math.reduce_mean(eeg, axis=(-1, -2))
        eeg_plcc = self.eeg_flatten(eeg_plcc)

        fnirs_attention_map_norm_plcc = self.fnirs_flatten(
            fnirs_attention_map_norm
        )

        fga_loss = pearson_r(
            eeg_plcc,
            fnirs_attention_map_norm_plcc
        )

        self.add_loss(1 - fga_loss)

        self.fga_loss_tracker.update_state(fga_loss)

        return fga_feature


class conv_block(keras.layers.Layer):
    def __init__(self, eeg_filter, eeg_size, eeg_stride,
                 fnirs_filter, fnirs_size, fnirs_stride,
                 eegfusion_filter, eegfusion_size, eegfusion_stride,
                 tem_kernel_size, fga_loss_name, padding):
        super(conv_block, self).__init__()

        self.eeg_conv = layers.Conv3D(
            filters=eeg_filter,
            kernel_size=eeg_size,
            strides=eeg_stride,
            padding=padding
        )
        self.eeg_act = layers.Activation('elu')
        self.eeg_bn = layers.BatchNormalization()

        self.fnirs_conv = layers.Conv3D(
            filters=fnirs_filter,
            kernel_size=fnirs_size,
            strides=fnirs_stride,
            padding=padding
        )
        self.fnirs_act = layers.Activation('elu')
        self.fnirs_bn = layers.BatchNormalization()

        self.eegfusion_conv = layers.Conv3D(
            filters=eegfusion_filter,
            kernel_size=eegfusion_size,
            strides=eegfusion_stride,
            padding=padding
        )
        self.eegfusion_act = layers.Activation('elu')
        self.eegfusion_bn = layers.BatchNormalization()

        self.fga = fga(
            tem_kernel_size,
            fga_loss_name
        )

    def call(self, inputs):
        eegfusion, eeg, fnirs = inputs

        eeg_feature = self.eeg_conv(eeg)
        eeg_feature = self.eeg_bn(eeg_feature)
        eeg_feature = self.eeg_act(eeg_feature)

        fnirs_feature = self.fnirs_conv(fnirs)
        fnirs_feature = self.fnirs_bn(fnirs_feature)
        fnirs_feature = self.fnirs_act(fnirs_feature)

        eegfusion_feature = self.eegfusion_conv(eegfusion)
        eegfusion_feature = self.eegfusion_bn(eegfusion_feature)
        eegfusion_feature = self.eegfusion_act(eegfusion_feature)

        eegfusion_fga = self.fga((
            eegfusion_feature,
            eeg_feature,
            fnirs_feature
        ))

        return eegfusion_fga, eeg_feature, fnirs_feature


class reduce_sum_layer(keras.layers.Layer):
    def __init__(self, axis, keepaxis, name=None, *args, **kwargs):
        super(reduce_sum_layer, self).__init__(name=name)

        self.axis = axis
        self.keepaxis = keepaxis

    def call(self, inputs):
        return tf.math.reduce_sum(
            inputs,
            axis=self.axis,
            keepdims=self.keepaxis
        )


class reduce_mean_layer(keras.layers.Layer):
    def __init__(self, axis, keepaxis, name=None, *args, **kwargs):
        super(reduce_mean_layer, self).__init__()

        self.axis = axis
        self.keepaxis = keepaxis

    def call(self, inputs):
        return tf.math.reduce_mean(
            inputs,
            axis=self.axis,
            keepdims=self.keepaxis
        )


class expand_dims_layer(keras.layers.Layer):
    def __init__(self, axis, *args, **kwargs):
        super(expand_dims_layer, self).__init__()

        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)


class prediction_weight_layer(keras.layers.Layer):
    def __init__(self, name=None):
        super(prediction_weight_layer, self).__init__(name=name)

        self.p_weight = self.add_weight(
            name='p_weight',
            shape=(2, 1),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        p_weights_softmax = tf.nn.softmax(
            self.p_weight,
            axis=0
        )

        p_weights_softmax = tf.expand_dims(
            p_weights_softmax,
            axis=0
        )

        the_prediction = tf.math.multiply(
            inputs,
            p_weights_softmax
        )

        the_prediction = tf.math.reduce_sum(
            the_prediction,
            axis=1
        )

        return the_prediction

class SRFB(keras.layers.Layer):

    def __init__(self, out_channels, **kwargs):
        super(SRFB, self).__init__(**kwargs)

        # 分支1
        self.b1_c1 = layers.Conv3D(filters=out_channels, kernel_size=1, strides=1, padding='valid')
        self.b1_bn1 = layers.BatchNormalization()
        self.b1_c2 = layers.Conv3D(filters=out_channels, kernel_size=3, strides=1, padding='same', dilation_rate=1)
        self.b1_bn2 = layers.BatchNormalization()

        # 分支2
        self.b2_c1 = layers.Conv3D(filters=out_channels, kernel_size=1, strides=1, padding='valid')
        self.b2_bn1 = layers.BatchNormalization()
        self.b2_c2 = layers.Conv3D(filters=out_channels, kernel_size=(3, 3, 1), strides=1, padding='same')
        self.b2_bn2 = layers.BatchNormalization()
        self.b2_c3 = layers.Conv3D(filters=out_channels, kernel_size=(1, 1, 3), strides=1, padding='same')
        self.b2_bn3 = layers.BatchNormalization()
        self.b2_c4 = layers.Conv3D(filters=out_channels, kernel_size=3, strides=1, padding='same', dilation_rate=(3, 3, 1))
        self.b2_bn4 = layers.BatchNormalization()

        # 分支3
        self.b3_c1 = layers.Conv3D(filters=out_channels, kernel_size=1, strides=1, padding='valid')
        self.b3_bn1 = layers.BatchNormalization()
        self.b3_c2 = layers.Conv3D(filters=out_channels, kernel_size=(5, 5, 1), strides=1, padding='same')
        self.b3_bn2 = layers.BatchNormalization()
        self.b3_c3 = layers.Conv3D(filters=out_channels, kernel_size=(1, 1, 5), strides=1, padding='same')
        self.b3_bn3 = layers.BatchNormalization()
        self.b3_c4 = layers.Conv3D(filters=out_channels, kernel_size=3, strides=1, padding='same', dilation_rate=(5, 5, 1))
        self.b3_bn4 = layers.BatchNormalization()

        # 三个分支沿时间维拼接后，将时间长度从 3T 压缩回 T。
        # Conv3D 的输入为 [B, D, H, T, C]，因此时间轴 axis=-2
        # 对应 kernel_size/strides 的第三个空间维度。
        self.conv_cat = layers.Conv3D(
            filters=out_channels,
            kernel_size=(1, 1, 3),
            strides=(1, 1, 3),
            padding='valid'
        )
        self.bn_cat = layers.BatchNormalization()

        # 残差连接
        self.conv_residual = layers.Conv3D(filters=out_channels, kernel_size=1, strides=1, padding='valid')
        self.bn_residual = layers.BatchNormalization()

    def call(self, inputs):
        # branch 1
        x1 = tf.nn.relu(self.b1_bn1(self.b1_c1(inputs)))
        x1 = tf.nn.relu(self.b1_bn2(self.b1_c2(x1)))

        # branch 2
        x2 = tf.nn.relu(self.b2_bn1(self.b2_c1(inputs)))
        x2 = tf.nn.relu(self.b2_bn2(self.b2_c2(x2)))
        x2 = tf.nn.relu(self.b2_bn3(self.b2_c3(x2)))
        x2 = tf.nn.relu(self.b2_bn4(self.b2_c4(x2)))

        # branch 3
        x3 = tf.nn.relu(self.b3_bn1(self.b3_c1(inputs)))
        x3 = tf.nn.relu(self.b3_bn2(self.b3_c2(x3)))
        x3 = tf.nn.relu(self.b3_bn3(self.b3_c3(x3)))
        x3 = tf.nn.relu(self.b3_bn4(self.b3_c4(x3)))

        # 按同一时间点交错排列三个分支：
        # [x1_t0, x2_t0, x3_t0, x1_t1, x2_t1, x3_t1, ...]。
        # 最终张量仍是在时间轴上拼接，形状由 T 变为 3T。
        x_cat = tf.stack([x1, x2, x3], axis=-2)
        x_shape = tf.shape(x_cat)
        x_cat = tf.reshape(
            x_cat,
            tf.concat(
                [
                    x_shape[:-3],
                    [x_shape[-3] * x_shape[-2]],
                    x_shape[-1:]
                ],
                axis=0
            )
        )
        x_cat = tf.ensure_shape(
            x_cat,
            [x1.shape[0], x1.shape[1], x1.shape[2], None, x1.shape[-1]]
        )
        x_cat = self.conv_cat(x_cat)
        x_cat = self.bn_cat(x_cat)
        x_cat = tf.nn.relu(x_cat)

        # 残差
        x_res = self.conv_residual(inputs)
        x_res = self.bn_residual(x_res)
        x_res = tf.nn.relu(x_res)

        out = tf.add(x_cat, x_res)
        return tf.nn.relu(out)


def sta_net():
    eeg_input = keras.Input(
        shape=(16, 16, 600, 1),
        name="eeg_input"
    )

    fnirs_input = keras.Input(
        shape=(11, 16, 16, 30, 2),
        name="fnirs_input"
    )

    eegfusion1, eeg1, fnirs1 = conv_block(
        eeg_filter=16,
        eeg_size=(2, 2, 13),
        eeg_stride=(2, 2, 6),
        fnirs_filter=16,
        fnirs_size=(2, 2, 5),
        fnirs_stride=(2, 2, 2),
        eegfusion_filter=16,
        eegfusion_size=(2, 2, 13),
        eegfusion_stride=(2, 2, 6),
        tem_kernel_size=5,
        fga_loss_name='fgsa1_plcc',
        padding='same'
    )((eeg_input, eeg_input, fnirs_input))

    # 将 SRFB 模块放至第一次下采样 (stride) 之后，大幅度减少多分支的大张量运算，并且作用于Fusion这条线的同时，EEG的维度也要进行匹配，方便下一次fga
    eeg1 = SRFB(out_channels=16)(eeg1)
    eegfusion1 = SRFB(out_channels=16)(eegfusion1)

    # eeg1 = layers.Conv3D(
    #     filters=16,
    #     kernel_size=(1, 1, 3),
    #     strides=(1, 1, 2),
    #     padding='same',
    #     use_bias=False
    # )(eeg1)
    # eeg1 = layers.BatchNormalization()(eeg1)
    # eeg1 = layers.Activation('elu')(eeg1)
    #
    # eegfusion1 = layers.Conv3D(
    #     filters=16,
    #     kernel_size=(1, 1, 3),
    #     strides=(1, 1, 2),
    #     padding='same',
    #     use_bias=False
    # )(eegfusion1)
    # eegfusion1 = layers.BatchNormalization()(eegfusion1)
    # eegfusion1 = layers.Activation('elu')(eegfusion1)


    eegfusion1 = layers.Dropout(0.5)(eegfusion1)
    eeg1 = layers.Dropout(0.5)(eeg1)
    fnirs1 = layers.Dropout(0.5)(fnirs1)

    eegfusion2, eeg2, fnirs2 = conv_block(
        eeg_filter=32,
        eeg_size=(2, 2, 5),
        eeg_stride=(2, 2, 2),
        fnirs_filter=32,
        fnirs_size=(2, 2, 3),
        fnirs_stride=(2, 2, 2),
        eegfusion_filter=32,
        eegfusion_size=(2, 2, 5),
        eegfusion_stride=(2, 2, 2),
        tem_kernel_size=3,
        fga_loss_name='fgsa2_plcc',
        padding='same'
    )((eegfusion1, eeg1, fnirs1))

    # 第二次SRFB
    # eegfusion2 = SRFB(out_channels=32)(eegfusion2)

    eegfusion2 = gap()(eegfusion2)
    eeg2 = gap()(eeg2)
    fnirs2 = gap()(fnirs2)

    eegfusion2 = layers.Dropout(0.5)(eegfusion2)
    eeg2 = layers.Dropout(0.5)(eeg2)
    fnirs2 = layers.Dropout(0.5)(fnirs2)

    eegfusion_feature, fnirs_feature, _ = e_f_attention(
        emb_size=256,
        d_model=256,
        heads=10,
        drop=0.5
    )((eegfusion2, fnirs2))

    eegfusion_feature_pweight = layers.Activation('elu')(
        eegfusion_feature
    )

    fnirs_feature_pweight = layers.Activation('elu')(
        fnirs_feature
    )

    eegfusion_feature_pweight = layers.Dense(
        256,
        activation='elu'
    )(eegfusion_feature_pweight)

    fnirs_feature_pweight = layers.Dense(
        256,
        activation='elu'
    )(fnirs_feature_pweight)

    eeg_feature = layers.Flatten()(eeg2)

    eeg_feature = layers.Dense(
        256,
        activation='elu'
    )(eeg_feature)

    eegfusion_pred = layers.Dense(2)(
        eegfusion_feature_pweight
    )

    fnirs_pred = layers.Dense(2)(
        fnirs_feature_pweight
    )

    eeg_pred = layers.Dense(2)(
        eeg_feature
    )

    # ======================= 修改点 1 =======================
    # 只保留 eeg_output 作为模型辅助输出。
    eeg_pred = layers.Activation(
        'softmax',
        name='eeg_output'
    )(eeg_pred)

    eegfusion_pred = layers.Activation(
        'softmax'
    )(eegfusion_pred)

    # ======================= 修改点 2 =======================
    # fNIRS 分支仍然参与最终 class_output 的加权融合，
    # 但不再命名为 fnirs_output，也不会作为模型输出。
    fnirs_pred = layers.Activation(
        'softmax'
    )(fnirs_pred)

    eegfusion_pred = expand_dims_layer(axis=1)(
        eegfusion_pred
    )

    fnirs_pred_exp = expand_dims_layer(axis=1)(
        fnirs_pred
    )

    the_pred = layers.Concatenate(axis=1)([
        eegfusion_pred,
        fnirs_pred_exp
    ])

    fnirs_p_weight = layers.Dense(1)(
        fnirs_feature_pweight
    )

    eegfusion_p_weight = layers.Dense(1)(
        eegfusion_feature_pweight
    )

    p_weight = layers.Concatenate()([
        eegfusion_p_weight,
        fnirs_p_weight
    ])

    p_weight = layers.Activation('softmax')(
        p_weight
    )

    p_weight = expand_dims_layer(axis=-1)(
        p_weight
    )

    the_pred = layers.Multiply()([
        the_pred,
        p_weight
    ])

    the_pred = reduce_sum_layer(
        axis=1,
        keepaxis=False,
        name='class_output'
    )(the_pred)

    # ======================= 修改点 3 =======================
    # 模型最终只有两个输出：
    #   1. class_output：融合最终分类输出
    #   2. eeg_output：EEG 辅助分类输出
    # 不再输出 fnirs_output。
    model = keras.Model(
        inputs=[
            eeg_input,
            fnirs_input
        ],
        outputs=[
            the_pred,
            eeg_pred
        ]
    )

    return model