"""
The AttentionBeforeConvolution implement as a keras layer the attention mechanism  mentioned in
https://github.com/tcxdgit/cnn_multilabel_classification

Performance: much slower than the many-to-one-attention or the self-attention

max_features = 2000  # Only consider the top 20k words
maxlen = 30  # Only consider the first 200 words of each movie review
25000/25000 [==============================] - 332s 13ms/sample - loss: 0.4246 - acc: 0.7987 - val_loss: 0.4486 - val_acc: 0.7844


If the attention layer is substituted with the self-attention mechanisms before performing convolution, the performance becomes:

> for the scaled_dot_attention:
25000/25000 [==============================] - 15s 594us/sample - loss: 0.4411 - acc: 0.7890 - val_loss: 0.4735 - val_acc: 0.7686

> for the seq_sef_attention:
25000/25000 [==============================] - 22s 869us/sample - loss: 0.4387 - acc: 0.7907 - val_loss: 0.4704 - val_acc: 0.7700

"""