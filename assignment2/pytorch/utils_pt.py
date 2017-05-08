from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F

# input is a Variable tensor from pytorch model
def get_predict_labels(tensor_var):
    pred = get_output_softmax_list(tensor_var)
    pred = np.array(pred)
    pred_labels = []
    for aList in pred:
        pred_labels.append(np.argmax(aList))
    return pred_labels

# draw confusion matrix for y_pred: List(predicted labels), y_true: List(real labels)
def draw_confusion_matrix(y_true, y_pred, classes):
    c = confusion_matrix(y_true, y_pred)
    hm = sns.heatmap(c, cbar=False, annot=True, fmt='g')
    plt.xticks(np.array(range(10)), classes, rotation=45)
    plt.yticks(np.array(range(10)), classes[::-1], rotation=30)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    #plt.show()
    plt.savefig('confusion_matrix.png')

def get_list_predicted_data(y_pred, probabilities):
    aList = []
    for index, i in enumerate(probabilities):
        aList.append({'classes': y_pred[index], 'probabilities': i})
    return aList

def get_output_softmax_list(tensor_var):
    pred = F.softmax(tensor_var)
    pred = pred.data.tolist()
    return pred
