from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass



import os

app = Flask(__name__)

load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')


embeddings = download_hugging_face_embeddings

  
pc = Pinecone(api_key=PINECONE_API_KEY)  # Intialize the pinecone
index_name = "test"
index = pc.Index(index_name)                                # connect to index

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

os.environ["GOOGLE_API_KEY"] = getpass.getpass(GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key = "AIzaSyDgA0uI5oefVdp5KrOst26-owsa9KFFzvM"
    # other params...
)


vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings,pinecone_api_key= "b8153aea-8890-4efb-b1f5-506be0d53b43")

retriever = vectorstore.as_retriever()


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain( retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    query = msg
    print(query)
    result= rag_chain.invoke({"input": query})
    print("Response : ", result["answer"])
    return str(result["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)