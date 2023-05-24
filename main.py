# Bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# Prompt templates
member_template = PromptTemplate(
    input_variables=['number'],
    template='Сome up with {number} of different combination of personal data: Human first name, last name, '
             'phone number and email. You might think that every Name, Surname, email and phone number is belong to '
             'Ukrainian person. Output information in next way:'
             'Number: number of iteration'
             'Name: first name that you generated; '
             'Last Name: last name that you generated; '
             'Email: email that you generated; '
             'Phone Number: phone that you generated;'
)

# Memory
member_memory = ConversationBufferMemory(input_key='number', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9)
memebers_generation_chain = LLMChain(llm=llm, prompt=member_template, verbose=True, output_key='title',
                                     memory=member_memory)
wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt:
    memebers_data = memebers_generation_chain.run(prompt)

    st.write(memebers_data)

    with st.expander('Title History'):
        st.info(member_memory.buffer)


