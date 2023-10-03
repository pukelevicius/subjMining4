import pandas as pd 
import re 
import string 
from langdetect import detect


def remove_emojis(text): #remove everything that's not a text
    return ''.join(char for char in text if char in (string.ascii_letters + string.digits + ' '))

def remove_tags(text):
    return re.sub(r'@USER', '', text)

def remove_multispace(text):
    return re.sub(r'[\s]+|[\t]', ' ', text)

#def remove_not_english(text):
#    words = text.split()
#    english_words = [word for word in words if detect(word) == 'en']
#    return ' '.join(english_words)

def preprocess_pipeline(text):
    
    text = remove_emojis(text)
    text = remove_tags(text)
    text = remove_multispace(text)
    
    return text
    
if __name__ == "__main__":
    
    olid_small = pd.read_csv("data\olid-train-small.csv", sep=',', header = 0)
    olid_small = olid_small.drop('id', axis = 1)    
    
    olid_test = pd.read_csv('data\olid-test.csv', sep = ',', header = 0)
    olid_test = olid_test.drop('id', axis = 1)  
    
    olid_small['text'] = olid_small['text'].apply(remove_tags)
    olid_small['text'] = olid_small['text'].apply(remove_emojis)
    olid_small['text'] = olid_small['text'].apply(remove_multispace)

    olid_test['text'] = olid_test['text'].apply(remove_emojis)
    olid_test['text'] = olid_test['text'].apply(remove_tags) 
    olid_test['text'] = olid_test['text'].apply(remove_multispace)

    olid_test.to_csv('data\olid-test-cleaned.csv', index=False)
    olid_small.to_csv('data\olid-train-small-cleaned.csv', index=False)




