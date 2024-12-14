import os
from dotenv import load_dotenv
import streamlit as st
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

url = "https://us-south.ml.cloud.ibm.com"

watsonx_project_id = ""
api_key = ""

def get_credentials():

    load_dotenv()

    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)

    print("*** Got credentials***")

def get_model(model_type,max_tokens,min_tokens,decoding,stop_sequences):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES:stop_sequences
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
        )

    return model

def get_prompt(question):

    instruction = "Answer this question briefly."
    examples = "\n\nQuestion: What is the capital of Germany\nAnswer: Berlin\n\nQuestion: What year was George Washington born?\nAnswer: 1732\n\nQuestion: What are the main micro nutrients in food?\nAnswer: Protein, carbohydrates, and fat\n\nQuestion: What language is spoken in Brazil?\nAnswer: Portuguese \n\nQuestion: "
    your_prompt = question
    end_prompt = "Answer:"

    final_prompt = instruction + examples + your_prompt + end_prompt

    return final_prompt

def answer_questions():

    get_credentials()

    st.title('🌠Test watsonx.ai LLM')
    user_question = st.text_input('Ask a question, for example: What is IBM?')

    if len(user_question.strip())==0:
        user_question="What is IBM?"

    final_prompt = get_prompt(user_question)

    print(final_prompt)

    model_type = ModelTypes.FLAN_UL2
    max_tokens = 100
    min_tokens = 20
    decoding = DecodingMethods.GREEDY
    stop_sequences = ['.']

    model = get_model(model_type, max_tokens, min_tokens, decoding,stop_sequences)

    generated_response = model.generate(prompt=final_prompt)
    model_output = generated_response['results'][0]['generated_text']
    print("Answer: " + model_output)

    formatted_output = f"""
        **Answer to your question:** {user_question} \
        *{model_output}*</i>
        """
    st.markdown(formatted_output, unsafe_allow_html=True)

answer_questions()
