from tensorflow import keras
from AttentionedNetworks.keras_self_attention.attention_layers.model import seq_self_attention, scaled_dot_self_attention, seq_weighted_self_attention

max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

hidden_size = 64
v_size = 128

# model = seq_self_attention(hidden_size, maxlen, max_features, v_size)
model = scaled_dot_self_attention(hidden_size, maxlen, max_features, v_size)
# model = seq_weighted_self_attention(hidden_size, maxlen, max_features, v_size)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
