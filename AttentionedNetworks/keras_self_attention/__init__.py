"""
Self-Attention mechanism for processing sequential data that considers the context for each timestamp.
https://github.com/CyberZHG/keras-self-attention

seq_self_attention:
25000/25000 [==============================] - 799s 32ms/sample - loss: 0.1904 - acc: 0.9287 - val_loss: 0.3274 - val_acc: 0.8646


scaled_dot_self_attention:
25000/25000 [==============================] - 244s 10ms/sample - loss: 0.1888 - acc: 0.9286 - val_loss: 0.3293 - val_acc: 0.8664

seq_weighted_self_attention：
25000/25000 [==============================] - 146s 6ms/sample - loss: 0.1860 - acc: 0.9296 - val_loss: 0.3175 - val_acc: 0.8693

"""