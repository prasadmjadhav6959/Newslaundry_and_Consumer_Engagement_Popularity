import streamlit as st
import pickle
import numpy as np

# Load the models and TF-IDF vectorizer
artl_model = pickle.load(open('top_artl.pkl', 'rb'))
pop_model = pickle.load(open('pop.pkl', 'rb'))
tfidf_vec = pickle.load(open('tfidf_vec.pkl', 'rb'))

def clean_title(title):
    # Add your title cleaning code here
    return title

def title_score(title):
    text = clean_title(title)
    text = tfidf_vec.transform([text])
    top_cat = artl_model.predict(text)
    pop = pop_model.predict(text)
    return top_cat, pop

# Streamlit app
st.title('Newslaundry and Consumer Engagement Popularity')

title_input = st.text_input('Enter the news title:')
if st.button('Predict'):
    if title_input:
        top_cat, pop = title_score(title_input)
        st.write('Top Article:', bool(top_cat[0]))
        st.write('Popularity Score:', round(pop[0], 2))
        st.write('Total Engagement:', int(np.expm1(pop[0])))
    else:
        st.write('Please enter a news title.')
