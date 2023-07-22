#!/usr/bin/env python
# coding: utf-8

# In[8]:

#loading all needed libraries 
import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler with the data range you want to scale
similarity_range = (0, 1)  # Range of similarity scores
scaler.fit(np.reshape(similarity_range, (2, 1)))  # Fit the scaler with the specified range

import string
import re
import nltk
nltk.download('stopwords')#downloading nltk stopwords
stopwords = nltk.corpus.stopwords.words('english')#assigning to variable english stopwords
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()#object of wordlemmatizer created
nltk.download('wordnet')#wordnet database is download 
nltk.download('omw-1.4')#Open Multilingual Wordnet database is loaded
import spacy
nlp = spacy.load('en_core_web_sm')#spacy model en_core_web_sm js loaded
stopwords1 = nlp.Defaults.stop_words#model stopwords are assigned to variable

#Function creation
def clean_text(text):
    if isinstance(text, str):# isinstance(object, classinfo),checking if string as input
        text = text.lower()#convert to lowercase
        text="".join([i for i in text if i not in string.punctuation])#punctuations are removed and remaining are joined
        text = re.sub('\S*\d\S*\s*','', text).strip()#pattern like words with digits are removed
    return text
def nltk_tokenization(text):
    if isinstance(text, str):
        tokens = re.split('\W+', text)#tokenisation
        tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stopwords]
        output = " ".join(tokens)#lemmatized words not in stopwords(nltk) and joined to return 
        return output#return to if loop
    return text#return to function
def spacy_lemmatizer(text):
    if isinstance(text, str):
        doc = nlp(text)#inputvtext loaded as spacy document 
        sent = [token.lemma_ for token in doc if not token.text in set(stopwords1)]
        return ' '.join(sent)#lemmatization and stopword removal using spacy Library
    return text


# Load the pre-trained model from sentence_transformer Library
#Cache function is created so that model is loaded only once and we can avoid model loading traffic and hence preserve memory 
@st.cache(allow_output_mutation=True)
def load_model():
	  return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

model = load_model()

# Streamlit app code
st.title("Semantic Similarity Analysis App")#title
st.markdown("By Shilpa Ajith")#markdown
image = Image.open("semantic_similarity.png")#image loaded
st.image(image, use_column_width=True)

#2 user inputs are taken
st.subheader("Enter your text1 here:")
user_input1 = st.text_area("Input1",placeholder="please enter a text")#placeholder and label added
st.subheader("Enter your text2 here:")
user_input2 = st.text_area("Input2",placeholder="please enter another text")

#pedict button is created
if st.button("Predict"):
    if user_input1 and user_input2:  # Check if both inputs are provided
        # Preprocess the input text
        
        user_input1 = clean_text(user_input1)
        user_input2 = clean_text(user_input2)
        user_input1 = nltk_tokenization(user_input1)
        user_input2 = nltk_tokenization(user_input2)
        user_input1 = spacy_lemmatizer(user_input1)
        user_input2 = spacy_lemmatizer(user_input2)
	
        #user input is encoded,dense vector of each sentence is created
        sentence_vec1 = model.encode([user_input1])[0]
        sentence_vec2 = model.encode([user_input2])[0]

        # Make predictions
        similarity = cosine_similarity([sentence_vec1], [sentence_vec2])[0][0]#cosine similarity is calculated 
        scaled_similarity = scaler.transform(np.reshape(similarity, (1, 1)))[0][0]#normalisation
        st.header("Prediction:")
        st.subheader(similarity)
    else:
        st.subheader("Please enter both sentences.")#if both inputs are not there
else:
    st.subheader("Please enter sentences.")#if Prediction buttin is not pressed 


# In[ ]:




