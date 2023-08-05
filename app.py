import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="data.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a world class startup development representative. 
I will share with you a conversation in a podcast about startup and you will learn information from the conversation then response.
You must follow ALL these rule on your response:

1/ Respoinse should be based on the information i gave you

2/ If the question are irrelevant, then try to makeup your own answer and add a "*" at the end of the sentence

Below is a piece of information I extracted from the podcast:
{message}

"""

prompt = PromptTemplate(
    input_variables=["message"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Question response generator", page_icon=":bird:")

    st.header("Question response generator :bird:")
    message = st.text_area("Questions")

    if message:
        st.write("Generating best answer...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()