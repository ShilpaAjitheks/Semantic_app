#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler with the data range you want to scale
similarity_range = (-1, 1)  # Range of similarity scores
scaler.fit(np.reshape(similarity_range, (2, 1)))  # Fit the scaler with the specified range

import string
import re
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('omw-1.4')
import spacy
nlp = spacy.load('en_core_web_sm')
stopwords1 = nlp.Defaults.stop_words

def clean_text(text):
    if isinstance(text, str):# isinstance(object, classinfo)
        text = text.lower()
        text="".join([i for i in text if i not in string.punctuation])
        text = re.sub('\S*\d\S*\s*','', text).strip()
    return text
def tokenization(text):
    if isinstance(text, str):
        tokens = re.split('\W+', text)
        tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stopwords]
        output = " ".join(tokens)
        return output
    return text
def lemmatizer(text):
    if isinstance(text, str):
        doc = nlp(text)
        sent = [token.lemma_ for token in doc if not token.text in set(stopwords1)]
        return ' '.join(sent)
    return text


# Load the saved model
model_name='sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(model_name)

# Streamlit app code
st.title("Semantic Similarity Analysis App")
st.markdown("By ABC")
image = Image.open("semantic_similarity.png")
st.image(image, use_column_width=True)

st.subheader("Enter your text1 here:")
user_input1 = st.text_area("Input1",placeholder="please enter a text")
st.subheader("Enter your text2 here:")
user_input2 = st.text_area("Input2",placeholder="please enter another text")

if st.button("Predict"):
    if user_input1 and user_input2:  # Check if both inputs are provided
        # Preprocess the input text
        
        user_input1 = clean_text(user_input1)
        user_input2 = clean_text(user_input2)
        user_input1 = tokenization(user_input1)
        user_input2 = tokenization(user_input2)
        user_input1 = lemmatizer(user_input1)
        user_input2 = lemmatizer(user_input2)
        
        sentence_vec1 = model.encode([user_input1])[0]
        sentence_vec2 = model.encode([user_input2])[0]

        # Make predictions
        similarity = cosine_similarity([sentence_vec1], [sentence_vec2])[0][0]
        scaled_similarity = scaler.transform(np.reshape(similarity, (1, 1)))[0][0]
        st.header("Prediction:")
        st.subheader(similarity)
    else:
        st.subheader("Please enter both sentences.")
else:
    st.subheader("Please enter sentences.")


# In[ ]:




