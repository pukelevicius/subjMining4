import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report, confusion_matrix


def get_trained_model(model_name: str, train_data: pd.DataFrame, train_data_name: str, output_dir: str):
 
    model = ClassificationModel('bert', model_name, args={"output_dir": 'output/' + train_data_name + '_' + output_dir,
                                                          'overwrite_output_dir': True},use_cuda=False)
    model.train_model(train_data[['text','labels']])

    return model
    
def load_model(model_path: str):
    model = ClassificationModel('bert', model_path, use_cuda=False)

    return model

def predict(no_models: bool):
    # train models if none exist
    if no_models:
        # load data sets
        olid = pd.read_csv('data/olid-train-small.csv', nrows=10)
        hasoc = pd.read_csv('data/hasoc-train.csv', nrows=10)
        # create dict of datasets
        data = {'olid': olid, 'hasoc': hasoc}

        # train models on datasets
        # loop over datasets
        for train_data_name, train_data in data.items():
            models = {'bert': 'bert-base-uncased', 'hatebert': 'GroNLP/hateBERT', 'fbert': 'diptanu/fBERT'}
            # loop over models
            for output_dir, model_name in models.items(): 
                # get trained model and predict
                model = get_trained_model(model_name, train_data, train_data_name, output_dir)

    # load models and predict
    for filename in os.listdir('output/'):
        # load model and predict
        model = load_model('output/' + filename)
        predictions = model.predict(olid['text'].tolist())
        test_data = pd.read_csv('data/olid-test.csv', nrows=10)

        # Specify the file path where you want to save the dictionary
        file_path = 'results/' + output_dir + '_' + train_data_name

        # get metrics
        results(predictions[0], test_data['labels'], file_path)
  


def results(predictions, true_labels, file_path):
    # metrics: precision, recall, f1-score
    metrics = classification_report(y_true=true_labels, y_pred=predictions)
    # confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the heatmap as an image
    plt.savefig(file_path + '_confusion_matrix.png')
    # Serialize and save the dictionary to a file
    with open(file_path + "metrics.json", "w") as file:
        json.dump(metrics, file)
    
if __name__ == '__main__':
    predict(no_models=True)

