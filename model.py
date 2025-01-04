#from langchain_google_genai import GoogleGenerativeAI as genai
import google.generativeai as genai
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import  load_dotenv
#from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import CSVLoader
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import  os

load_dotenv()

model="gemini-pro"
llm = genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model_name = "hkunlp/instructor-large"

instructor_embedding = HuggingFaceInstructEmbeddings(model_name=model_name)

vectordb_file_path="faiss_index"


def create_vector_db():
    loader = CSVLoader(file_path="RCCGNationalDates2025.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embedding)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    #load the vector database from the local folder
    print("this is the file path", vectordb_file_path)
    vectordb = FAISS.load_local(vectordb_file_path,
                                instructor_embedding,
                                allow_dangerous_deserialization = True
                                )

    # create a retriever for querying the vector db
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer.
    Where the answer is not found in the context, you can use your intuition.
    Where a bible explanation is needed, please use 'Blue letter bible commentary by David Guzik'.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        chain_type_kwargs={"prompt": prompt},
                                        input_key = "query",
                                        return_source_documents = True
                                        )
    return chain

if __name__ == "__main__":
    #pass
    create_vector_db()
    #chain = get_qa_chain()
    #print(chain("What is in open heaven for Jan2?"))
