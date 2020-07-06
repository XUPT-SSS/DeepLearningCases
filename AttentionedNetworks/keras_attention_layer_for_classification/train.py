# from tensorflow import keras
# from AttentionedNetworks.keras_attention_layer_for_classification.model import attention_lstm, lstm, attention_lstm_explicable
# from keract import get_activations
#
# max_features = 20000  # Only consider the top 20k words
# maxlen = 200  # Only consider the first 200 words of each movie review
# (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=max_features)
# print(len(x_train), "Training sequences")
# print(len(x_val), "Validation sequences")
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
#
# hidden_size = 64
# v_size = 128
#
# model = attention_lstm(hidden_size, maxlen, max_features, v_size)
#
# model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
# model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))



from keract import get_activations
import matplotlib.pyplot as plt
from tensorflow import keras
from AttentionedNetworks.keras_attention_layer_for_classification.model import attention_lstm, lstm, attention_lstm_expliable
from AttentionedNetworks.keras_attention_layer_for_classification.attention_layer import AttentionWithContext


max_features = 10000  # Only consider the top 20k words
maxlen = 100  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

hidden_size = 32
v_size = 64

model, fn = attention_lstm_expliable(hidden_size, maxlen, max_features, v_size)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))

x_val_single = x_val[0:1]
y_val_single = y_val[0:1]
a = model.predict(x_val_single)
att_weights = fn(x_val_single)


# top is attention map.
# bottom is ground truth.
plt.imshow(att_weights[0], cmap='hot')
# seq_len = len(x_val_single[0])
# scales_ls = range(seq_len)
#
# tt = x_val_single[0].tolist()
# text_labels = [str(i) for i in tt]
# plt.xticks(scales_ls, text_labels)
# plt.ylabel(y_val_single)
plt.axis('off')
plt.savefig('attention.png')
plt.close()
plt.clf()
