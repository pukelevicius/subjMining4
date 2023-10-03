import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack 
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from preprocess import preprocess_pipeline
import matplotlib.pyplot as plt
import json

def SVM(X_train, Y_train, X_test, Y_test, results_path):
    # create n-grams for the datasets
    vec_word = CountVectorizer(analyzer = 'word', ngram_range = (1,1), lowercase = False)
    vec_char = CountVectorizer(analyzer = 'char', ngram_range = (1,1), lowercase = False)

    # transform data
    X_train = hstack((vec_word.fit_transform(X_train),vec_char.fit_transform(X_train)))
    X_test = hstack((vec_word.transform(X_test),vec_char.transform(X_test)))

    # train the model
    lsv = LinearSVC(random_state = 0)
    lsv.fit(X_train, Y_train)

    # predict the labels
    predictions = lsv.predict(X_test)
    
    # saving the matrix 
    results(predictions, Y_test, results_path)

    return 

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
    plt.close()

    with open(file_path + "_metrics.json", "a") as file:
        json.dump(metrics, file)
    
    file.close()
    
if __name__ == '__main__':

    # load the data
    olid_test_cleaned_df = pd.read_csv('data\olid-test-cleaned.csv', sep = ',', header = 0)
    olid_train_cleaned_df = pd.read_csv('data\olid-train-small-cleaned.csv', sep = ',', header = 0)
    hasoc = pd.read_csv('data\hasoc-train.csv', sep = ',', header = 0)
    
    # cleaning the data
    hasoc['text'] = hasoc['text'].apply(preprocess_pipeline)
    
    # getting the predictions
    SVM(olid_train_cleaned_df['text'], olid_train_cleaned_df['labels'], olid_test_cleaned_df['text'], 
        olid_test_cleaned_df['labels'], 'results\SVM_in_domain')
    
    SVM(hasoc['text'], hasoc['labels'], olid_test_cleaned_df['text'], 
        olid_test_cleaned_df['labels'], 'results\SVM_cross_domain') 
