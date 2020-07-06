import numpy as np
import matplotlib.pyplot as plt


def plot_saliency(loaded_model, pure_txt, text_class, pred_labels, text_sequence):
    text_class = 'Positive' if text_class==1 else 'Negative'
    pred_labels = 'Positive' if pred_labels==1 else 'Negative'

    input_tensors = [loaded_model.input, K.learning_phase()]
    model_input = loaded_model.layers[2].input # the input for convolution layer
    model_output = loaded_model.output[0][1]
    gradients = loaded_model.optimizer.get_gradients(model_output,model_input)
    compute_gradients = K.function(inputs=input_tensors, outputs=gradients)
    # matrix = compute_gradients([text_sequence.reshape(1,30), text_class])[0][0]
    matrix = compute_gradients([text_sequence.reshape(1, 30), 0])[0][0]

    matrix = matrix[:len(pure_txt),:]

    matrix_magnify=np.zeros((matrix.shape[0]*10,matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(10):
            matrix_magnify[i*10+j,:]=matrix[i,:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(normalize_array(np.absolute(matrix_magnify)), interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(5, matrix.shape[0]*10, 10), pure_txt, weight='bold',fontsize=24)
    plt.xticks(np.arange(0, matrix.shape[1], 50), weight='bold',fontsize=24)
    plt.title('True Label: "{}" Predicted Label: "{}"'.format(text_class,pred_labels), weight='bold',fontsize=24)
    plt.colorbar()
    plt.show()