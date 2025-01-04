import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# create an object of genai
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model_name="gemini-1.5-pro"

# give a title to the application
st.title("Chat with Seth")

st.text("Hi there, I am Seth. Let's do bible study together. Ask me any question.")

# check if messages are available
if "messages" not in st.session_state:

    # assign the messages
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "enter the question"
        }

    ]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


 # create a function for the model
def llm_call(query,
            model_name=model_name,
            temperature=0.7,
            max_tokens=500):
    model = genai.GenerativeModel(
        model_name,
        system_instruction=messages['system'],
    )

    prompt = messages['prompt']
    response = model.generate_content([prompt])

    #response = model.generate_content(query)

    with st.chat_message("assistant"):
        # output
        st.markdown(response.text)

    # to persist to session state
    st.session_state.messages.append(
        {
            "role" : "user",
            "content": prompt
        }
    )

    st.session_state.messages.append(
        {
            "role" : "assistant",
            "content": response.text
        }
    )

query= st.chat_input("What is your question")

messages = {
        "system": f"""
                    When ask any question from the bible use the blue letter commentary by David Guzik to 
                    answer the it.
                """,

        "prompt": f"""\
                         {query}
                    """
    }

if query:
    with st.chat_message("user"):
        st.markdown(query)


    llm_call(messages)