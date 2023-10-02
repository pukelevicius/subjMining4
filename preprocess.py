import pandas as pd 
import re 
import string 



def remove_emojis(text):
    return ''.join(char for char in text if char in (string.ascii_letters + string.digits + string.punctuation + ' '))

def remove_tags(text):
    return re.sub(r'@\w+', '', text)

def remove_multispace(text):
    return re.sub(r'[\s]+|[\t]|[.,"\']', ' ', text)

def preprocess_pipeline(text):
    
    text = remove_emojis(text)
    text = remove_tags(text)
    text = remove_multispace(text)
    
    return text
    
if __name__ == "__main__":
    
    olid_small = pd.read_csv("data\olid-train-small.csv", sep=',', header = 0)
    olid_small = olid_small.drop('id', axis = 1)
    olid_small['text'] = olid_small['text'].apply(remove_tags)
    olid_small['text'] = olid_small['text'].apply(remove_emojis)
    olid_small['text'] = olid_small['text'].apply(remove_multispace)

    olid_small.to_csv('data\olid-train-small-cleaned.csv', index=False)




