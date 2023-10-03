import pandas as pd 
import re 
import string 
from langdetect import detect

olid_small = pd.read_csv("data\olid-train-small.csv", sep = ',', header = 0)
olid_test = pd.read_csv('data\olid-test.csv', sep = ',', header = 0)
olid_small = olid_small.drop('id', axis = 1)

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


olid_small['text'] = olid_small['text'].apply(remove_tags)
olid_small['text'] = olid_small['text'].apply(remove_emojis)
olid_small['text'] = olid_small['text'].apply(remove_multispace)
#olid_small['text'] = olid_small['text'].apply(remove_not_english)


olid_test['text'] = olid_test['text'].apply(remove_tags)
olid_test['text'] = olid_test['text'].apply(remove_emojis)
olid_test['text'] = olid_test['text'].apply(remove_multispace)
#olid_test['text'] = olid_test['text'].apply(remove_not_english)

olid_small.to_csv('data\olid-train-small-cleaned_2.csv')
olid_test.to_csv('data\olid-train-test-cleaned_2.csv')





