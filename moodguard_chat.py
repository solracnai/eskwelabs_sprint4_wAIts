import streamlit as st
import pandas as pd
import numpy as np
import time
import openai
from streamlit_chat import message

openai.api_key = st.secrets["OPENAI-API-KEY"]


st.image('s4g4-waits-moodguard-banner.png')
st.write("MoodGuard is not a replacement for professional mental health guidance; rather, it is an exploration of how LLMs can contribute to mental health services. Seeking assistance from professionals is highly recommended")
st.markdown('')

#--------------------------------------------------------------------------------------------------

#######################################################
# Initialize session state
#######################################################
# Open AI Model
if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"

# Chat history
if "messages" not in st.session_state:
    st.session_state.end_session = False
    st.session_state.messages = []
    st.session_state.GPT_instuction = """
        You are warmth and approachable mental health expert and therapist, your expertise is in helping people in thier teens overcome obstacle
        regarding motivation, career, school, relationships and self esteem and you have done this for a few decades. Your task is to provide the best advice for
        helping improve mental health. You must ask a question before answering so you have a more precise answer
        Answer in a kind and loving voice and make it concise.

        STEPS:
        Step 1: Start the conversation by questions below. Ask the patient 1 question at a time.
            What are the obstacle you are currently facing regarding motivation, career, school, relationships and self esteem?
            Are there any issues or experiences that have contributed to these challenge?
            How have these obstacles affected your overall mental health and well being?

        Step 2: Gather enough information to form an analysis, then provide possible cause and possible action for the patient to improve his mental health.
        In this step you can answer in bullet points.

        Step 3: Ask if they have other concerns. if none you can end the session
        """

#######################################################
# Function Name: display_chat_messages
# Description  : Display Chat History
#######################################################
def display_chat_messages():
    for message in st.session_state.messages:

        if message["role"] =="assistant":
            avatar = '👨‍⚕️'
        else: 
            avatar = None
        
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
        

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
                "content": prompt
            }       
        ],
        max_tokens=256,
        temperature=0.6
    )

    response = completion.choices[0].message.content.strip()

    return response


#######################################################
# Function Name: start_session_chat
# Description  : Start Mental Health Session
#######################################################
def start_session_chat():
    with st.chat_message("assistant", avatar='👨‍⚕️'):
        prompt = 'Start the session'

        try:
            response = generate_response(prompt)
        except:
            response = "Chat is unavailable."

        st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


#######################################################
# Function Name: start_new_session
# Description  : Start New Therapy Session
#######################################################
def start_new_session():
    st.session_state.messages = []
    st.session_state.end_session = False

#######################################################
# Function Name: end_session
# Description  : End Current Therapy Session
#######################################################
def end_session():
    st.session_state.end_session = True


#######################################################
# Function Name: download_response
# Description  : Download User Responses from Messages
#######################################################
@st.cache_data
def download_response(messages):
    user_prompt_list = [message["content"] for message in messages if message["role"] =="user"]
    csv_file = pd.DataFrame(user_prompt_list, columns=['text']).to_csv(index=False)
    return csv_file
#--------------------------------------------------------------------------------------------------


#######################################################
# MAIN Program
#######################################################

# Display chat messages from history on app rerun
display_chat_messages()

# Start Therapy Session
if (len(st.session_state.messages) == 0) and (st.session_state.end_session == False):
    start_session_chat()

# Waiting for user input
if prompt := st.chat_input("Enter Message Here"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar='👨‍⚕️'):
        message_placeholder = st.empty()
        response = ""

        try:
            assistant_response = generate_response(prompt)
        except:
            assistant_response = "Chat is unavailable."

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            response += chunk + " "
            time.sleep(0.05)

            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(response + "▌")
        message_placeholder.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Start New Therapy Session
if st.session_state.end_session == True:

    csv_file = download_response(st.session_state.messages)
    st.markdown("Click the button below to download your responses")
    st.download_button(label="Download response as CSV", data=csv_file, file_name='user_response.csv', mime='text/csv')

    st.markdown("")
    st.button("Start New Session", on_click=start_new_session)

# End Current Therapy Session
if st.session_state.end_session == False:
    st.button("End Current Session", on_click=end_session)