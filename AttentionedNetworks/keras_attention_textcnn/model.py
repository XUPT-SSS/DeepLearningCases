from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf

from AttentionedNetworks.keras_attention_textcnn.attention_layer import AttentionBeforeConvolution
from AttentionedNetworks.keras_self_attention.attention_layers.scaled_dot_attention import ScaledDotProductAttention
from AttentionedNetworks.keras_self_attention.attention_layers.seq_self_attention import SeqSelfAttention


def attention_cnn(maxlen, attention_hidden_dim, max_features, v_size, dp_rate=0.5, filter_sizes=[2, 3, 4], NUM_FILTERS=128):
    inputs = Input(shape=(None,), dtype="int32", name='inputs')
    x = Embedding(max_features, v_size, mask_zero=False)(inputs)

    # attention layer before convolution
    x = AttentionBeforeConvolution(maxlen, attention_hidden_dim=attention_hidden_dim)(x)

    convs = []
    for kernel_size in filter_sizes:
        c = Conv1D(NUM_FILTERS, kernel_size, activation='relu')(x)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)

    if dp_rate > 0:
        # 加dropout层
        x = Dropout(dp_rate)(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)

    model.summary()
    return model


def scaled_dot_attention_cnn(max_features, v_size, dp_rate=0.5, filter_sizes=[2, 3, 4], NUM_FILTERS=128):
    inputs = Input(shape=(None,), dtype="int32", name='inputs')
    x = Embedding(max_features, v_size, mask_zero=False)(inputs)

    # attention layer before convolution
    x = ScaledDotProductAttention(name='Attention')(x)

    convs = []
    for kernel_size in filter_sizes:
        c = Conv1D(NUM_FILTERS, kernel_size, activation='relu')(x)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)

    if dp_rate > 0:
        # 加dropout层
        x = Dropout(dp_rate)(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)

    model.summary()
    return model


def seq_self_attention_cnn(max_features, v_size, dp_rate=0.5, filter_sizes=[2, 3, 4], NUM_FILTERS=128):
    inputs = Input(shape=(None,), dtype="int32", name='inputs')
    x = Embedding(max_features, v_size, mask_zero=False)(inputs)

    # attention layer before convolution
    x = SeqSelfAttention(attention_activation='sigmoid')(x)

    convs = []
    for kernel_size in filter_sizes:
        c = Conv1D(NUM_FILTERS, kernel_size, activation='relu')(x)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)

    if dp_rate > 0:
        # 加dropout层
        x = Dropout(dp_rate)(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)

    model.summary()
    return model


