import os
import itertools
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
# use keybert and keyprase_vectorizers as wrapper for CountVectorizer and embedding
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer

@st.cache
def keyword_extraction(doc, n): 
    vectorizer = KeyphraseCountVectorizer()
    sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    kw_model = KeyBERT(model=sentence_model)
    kw = kw_model.extract_keywords(doc, vectorizer=vectorizer,  top_n=n)
    df=pd.DataFrame(kw)[0]
    return df

st.set_page_config(
    page_title = "Recommand keywords for your next publication",
    page_icon="🔑"
)

st.title("🔑 Recommand keywords for your next publication")
st.header("")
st.write("""
- This web app will generate keywords for your manuscript based on the abstract. 
- It was built with [KeyBERT](https://github.com/MaartenGr/KeyBERT) that uses the [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) Transformer model 🤗 which is fast and light. You can get your results with a click of the button.
""")

form = st.form(key="annotation")

with form:
    doc = st.text_area("Please paste your abstract here: ", height=300)
    n = st.slider("Select the number of keywords you'd like to generate", min_value=1,  max_value=10,step=1, value=5)
    submitted = st.form_submit_button(label="Submit")

if submitted:
    # do calculation
    df=keyword_extraction(doc, n)
    st.dataframe(df)


if not submitted:
    st.stop()
