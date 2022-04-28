import pandas as pd
import numpy as np

#load data
location = 'C:/Users/andti/OneDrive/Documents/Datasets/WELFake_Dataset.csv'
d = pd.read_csv(location)
d.drop(['Unnamed: 0'], axis=1, inplace = True)

DataSet = d.copy()
DataSet.label.value_counts()

#using text as an indicator 
TextDS = d[['text', 'label']]

#remove stop words
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def TextWrangle (DataElement):
    LowerCase = DataElement.str.lower()
    RemoveStopWords = LowerCase.astype(str).apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    return RemoveStopWords

TextDS['text'] = TextWrangle(TextDS['text'])


