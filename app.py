import streamlit as st
# from streamlit_chat import message

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import openai

from skllm.config import SKLLMConfig
from skllm import ZeroShotGPTClassifier, FewShotGPTClassifier
from skllm.preprocessing import GPTSummarizer

import nltk
from nltk.corpus import stopwords
import spacy
import contractions
import string

from wordcloud import WordCloud


#st.set_page_config(layout="wide") # Page expands to full width


st.image('s4g4-waits-moodguard-banner.png')
st.write("MoodGuard is not a replacement for professional mental health guidance; rather, it is an exploration of how LLMs can contribute to mental health services. Seeking assistance from professionals is highly recommended")
st.write("Instructions: Upload the input file through the sidebar and then select the \"Start Analyze Data\" button.")

#--------------------------------------------------------------------------------------------------

#######################################################
# Initialize session state
#######################################################

# First Initialization
if "nlp" not in st.session_state:
    st.session_state.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])

    nltk.download('punkt') # Downloads the Punkt tokenizer models
    nltk.download('stopwords') # Downloads the list of stopwords
    nltk.download('wordnet') # Downloads the WordNet lemmatizer data
    nltk.download('averaged_perceptron_tagger')

# Open AI Model
if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"

    # Set OpenAI Keys
    openai.api_key = st.secrets["OPENAI-API-KEY"]
    SKLLMConfig.set_openai_key(openai.api_key)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.GPT_instuction = """
        You are warmth and approachable mental health expert and therapist, your expertise is in helping people in thier teens overcome obstacle
        regarding motivation, career, school, relationships and self esteem and you have done this for a few decades. Your task is to provide the best advice for
        helping improve mental health. Answer in concise and bullet form. Format your response for markdown processor
        """
        
#--------------------------------------------------------------------------------------------------

#######################################################
# Function Name: sp_preprocess
# Description  : Get token using SpaCy
#######################################################
@st.cache_data
def sp_preprocess(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = st.session_state.nlp
    clean_text = contractions.fix(text)
    tokens = nlp(clean_text)

    lemmatized_tokens = [token.lemma_.lower() for token in tokens
                         if token.pos_ in allowed_postags
                         if token.is_alpha
                         if not token.is_stop]

    return lemmatized_tokens


#######################################################
# Function Name: generate_donut_chart
# Description  : Show distribution via Donut Chart
#######################################################
@st.cache_data
def generate_donut_chart(series_data):
    # Sample data
    labels = [x.capitalize() for x in series_data.index]
    sizes = series_data.to_list()             # Sum of sizes should be 100 for percentages to work
    doc_count = series_data.sum()
    # colors = ['#7770B1' if x == 'Depressed' 
    #             else '#AC9FE2' if x == 'Anxious'
    #             else '#E6E4EF' for x in labels]

    colors = ['#B42913' if x == 'Self-harm' 
                else '#ED9041' if x == 'Panic'
                else '#F8CAA2' if x == 'Neglect'
                else '#E6E4EF' for x in labels]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(3,3))

    # Outer pie chart (wedgeprops creates a donut effect)
    ax.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90, pctdistance=0.85,
        wedgeprops={'width': 0.4, 'edgecolor': 'w'}, colors=colors, textprops={'fontsize': 7})

    # Inner white circle (to create a hole)
    circle = plt.Circle((0, 0), 0.7, color='white')
    ax.add_artist(circle)
    ax.axis('equal')        # Equal aspect ratio ensures that pie is drawn as a circle

    # Set a title
    #ax.set_title('Document Classification Distribution', fontsize=15)
    plt.annotate(f'{doc_count:,d}', (0, 0), fontsize=50, color='black', ha='center', va='center')
    plt.annotate(f'Documents', (0, -0.3), fontsize=8, color='black', ha='center', va='center')

    # Display the chart
    st.markdown("### Topic Classification")
    st.pyplot(fig)


#######################################################
# Function Name: generate_bar_chart
# Description  : Show distribution via Donut Chart
#######################################################
@st.cache_data
def generate_bar_chart(data):
    fontcolor = '#262564'

    df = data.gpt_label.value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(7,2), facecolor="#e6e4ef")

    ax = sns.barplot(x=df.values, y=df.index, palette="dark:#7770B1_r")
    # Change the background color
    plt.xlabel('Counts')
    plt.ylabel('Topics')

    plt.gca().set_facecolor("#e6e4ef")
    plt.gca().invert_yaxis()

    for spine in ['right', 'top']:
        plt.gca().spines[spine].set_visible(False)

    for j, v in enumerate(list(df.values)):
        plt.text(v, j, ' ' + str(round(v,4)), ha='left', va='center', color=fontcolor)

   # Set the text color of the labels on the x-axis and y-axis
    ax.xaxis.label.set_color(fontcolor)
    ax.yaxis.label.set_color(fontcolor)

    # Set the font color of the tick labels on the x-axis and y-axis
    ax.tick_params(axis='x', colors=fontcolor)
    ax.tick_params(axis='y', colors=fontcolor)

    # Display the chart
    st.markdown("### Topic Classification")
    st.pyplot(fig)


#######################################################
# Function Name: generate_wordcloud_image
# Description  : Create image for wordcloud
#######################################################
@st.cache_data
def generate_wordcloud_image(tokens, mask=None, colormap=None):
    wordcloud = WordCloud(width=1600, height=900,
                          background_color='#AB9EE2',
                          stopwords=set(stopwords.words('english')),
                          min_font_size=10, mask=mask, colormap=colormap)
    wordcloud.generate_from_frequencies(tokens)

    return wordcloud


#######################################################
# Function Name: plot_wordcloud
# Description  : Plot Word Cloud
#######################################################
@st.cache_data
def plot_wordcloud(joined_tokens):
    top_words = nltk.FreqDist(joined_tokens)
    top_words = top_words.most_common(top_words.B())
    token_dict = {x[0]:x[1] for x in top_words}

    fig, ax = plt.subplots(figsize=(10,10), facecolor="#AB9EE2")

    plt.imshow(generate_wordcloud_image(token_dict, colormap='Reds'), interpolation='bilinear')

    plt.axis("off")
    plt.tight_layout(pad = 0)

    st.markdown("### Word Cloud")
    st.pyplot(fig)



#######################################################
# Function Name: data_classification
# Description  : Classify Data from input
#######################################################
@st.cache_data
def data_classification(data):
    df = data.copy()

   # Initialize the ZeroShotGPTClassifier
    clf = ZeroShotGPTClassifier(openai_model="gpt-3.5-turbo")

    # Fit the classifier with some dummy data and the classes you're interested in
    # clf.fit(None, ["confident", "anxious", "depressed"])
    clf.fit(None, ["neglect", "education", "self-harm", "panic"])
    labels = clf.predict(df['text'])

    # Add the predicted labels to data
    df['gpt_label'] = labels

    return df


#######################################################
# Function Name: train_classifier
# Description  : Train data classifier
#######################################################
@st.cache_resource
def train_classifier():
    df = read_csv("https://drive.google.com/uc?export=download&id=1TLXzReb5HQttKa-vrZkGEIvkx7HHX7-0")

   # Initialize the ZeroShotGPTClassifier
    clf = FewShotGPTClassifier(openai_model='gpt-3.5-turbo')

    # Fit the classifier with sample data
    clf.fit(df['text'], df['main_topic'])
    return clf


#######################################################
# Function Name: data_classification
# Description  : Classify Data from input
#######################################################
@st.cache_data
def data_classification_2(data):
    df = data.copy()

    clf = train_classifier()
    labels = clf.predict(df['text'])

    # Add the predicted labels to data
    df['gpt_label'] = labels

    return df


#######################################################
# Function Name: summarize_corpus
# Description  : Summarize Text
#######################################################
@st.cache_data
def summarize_corpus(data):
    GPTSum = GPTSummarizer(openai_model='gpt-3.5-turbo', max_words=50)

    # Generate summary for the concatenated sample of positive reviews.
    summary = GPTSum.fit_transform([' '.join(data)])[0]
    return summary


#######################################################
# Function Name: generate_response
# Description  : Generate response from openai
#######################################################
@st.cache_data
def generate_response(prompt):

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": st.session_state.GPT_instuction
            },
            {
                "role": "user",
                "content": 'Provide Top 3 recommendations to the patient based on the following text: ' + prompt
            }       
        ],
        max_tokens=256,
        temperature=0.6
    )

    response = completion.choices[0].message.content.strip()
    return response


#######################################################
# Function Name: read_csv
# Description  : Read CSV File
#######################################################
@st.cache_data
def read_csv(url):
    return pd.read_csv(url)


#######################################################
# Function Name: plot_wordcloud
# Description  : Plot Word Cloud
#######################################################
def analyze_data(data):

    df = data_classification_2(data)
    generate_bar_chart(df)

    #df = data.copy()
    #generate_donut_chart(df['gpt_label'].value_counts())
    
    #st.markdown("#### Data")
    df1 = df[['text','gpt_label']].copy()
    df1.columns = ['Text Response', 'Label']
    st.dataframe(df1)
    st.write("---")

    # Generate tokens
    df['token'] = df.text.apply(sp_preprocess)
    joined_tokens = [token for token_list in df.token for token in token_list]

    if len(joined_tokens) > 0:
        plot_wordcloud(joined_tokens)

    # Summarize Text
    summary = summarize_corpus(df.text.to_list())
    st.markdown(summary)
    st.write("---")

    # Recommendations
    st.markdown("#### Recommendations")
    recommendations = generate_response(summary)
    st.markdown(recommendations)
  

#--------------------------------------------------------------------------------------------------

#######################################################
# MAIN Program
#######################################################
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://drive.google.com/file/d/18honVLHoQZ5iFU_zz6P51ASyC8ajvByL/view?usp=drive_link)
""")


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.sidebar.button("Start Analyzing Data", on_click=analyze_data(input_df))


#--------------------------------------------------------------------------------------------------
