from src.helper import load_data,text_split,download_hugging_face_embeddings
from langchain.vectorstores import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore


load_dotenv()


PINECONE_API_KEY =  os.environ.get('PINECONE_API_KEY')



extracted_data =  load_data("data/")


text_chunks = text_split(extracted_data)


embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key="b8153aea-8890-4efb-b1f5-506be0d53b43")  # Intialize the pinecone
index_name = "test"
index = pc.Index(index_name)    


# Create the Embedding and store it in the index with the metadata.

def store_embeddings(text_chunks, embeddings):
    for i, t in enumerate(text_chunks):
        # Create embeddings for the document
        vector = embeddings.embed_query(t.page_content)
        
        # Generate a unique ID for each document
        doc_id = f"doc_{i}"

        metadata = {
            "text": t.page_content,

        }
        
        # Upsert the vector with the generated ID
        index.upsert(
            vectors=[{"id": doc_id, "values": vector,"metadata":metadata}]
        )

# Store embeddings for the text chunks
docsearch = store_embeddings(text_chunks, embeddings)

print("Embeddings stored successfully.")


vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings,pinecone_api_key= "b8153aea-8890-4efb-b1f5-506be0d53b43")

retriever = vectorstore.as_retriever()