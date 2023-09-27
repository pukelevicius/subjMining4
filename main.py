import pandas as pd
import numpy as np
import os
import json

from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report, confusion_matrix


def get_trained_model(model_name: str, train_data: pd.DataFrame, train_data_name: str, output_dir: str):
 
    model = ClassificationModel('bert', model_name, args={"output_dir": 'output/'+train_data_name+output_dir,
                                                          'overwrite_output_dir': True},use_cuda=False)
    model.train_model(train_data[['text','labels']])

    return model
    
def load_model(model_path: str):

    model = ClassificationModel('bert', model_path, use_cuda=False)

    return model

def predict(models_exist: bool):
    # check if models exist
    if models_exist:
        # Loop over the files in the directory
        for filename in os.listdir('output/'):
            # load model and predict
            model = load_model(filename)
            predictions = model.predict(olid['text'].tolist())
            
    # if no models exist
    else:
        # load data sets
        olid = pd.read_csv('data/olid-train-small.csv', nrows=10)
        hasoc = pd.read_csv('data/hasoc-train.csv', nrows=10)
        test_data = pd.read_csv('data/olid-test.csv', nrows=10)
        # create dict of datasets
        data = {'olid_': olid, 'hasoc_': hasoc}

        # train models on datasets
        # loop over datasets
        for train_data_name, train_data in data.items():
            models = {'bert': 'bert-base-uncased', 'hatebert': 'GroNLP/hateBERT', 'fbert': 'diptanu/fBERT'}
            # loop over models
            for output_dir, model_name in models.items(): 
                # get trained model and predict
                model = get_trained_model(model_name, train_data, train_data_name, output_dir)
                predictions = model.predict(olid['text'].tolist())
                metrics = classification_report(y_true=test_data['labels'], y_pred=predictions[0])
                cm = confusion_matrix(test_data['labels'], predictions[0])
                
                # Specify the file path where you want to save the dictionary
                file_path = 'results/'+output_dir+'_'+train_data_name+"metrics.json"
                np.savetxt('results', cm, fmt='%d', delimiter='\t')

                # Serialize and save the dictionary to a file
                with open(file_path, "w") as file:
                    json.dump(metrics, file)



def get_metrics(predictions, true_labels):
    pass

    
if __name__ == '__main__':
    predict(models_exist=False)

