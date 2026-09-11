from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


loader = DirectoryLoader(
    path = "./data",
    glob = "**/*.pdf",
    loader_cls = UnstructuredFileLoader,
    show_progress = True,
    use_multithreading = True
)

data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap = 200,
    separators = [
        "\n\n",
        "\n",
        ". ",
        " "
        ""
    ]
)

chunks = splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(
    api_key = os.getenv("GEMINI_API_KEY"),
    model = "gemini-embedding-2-preview"
)

vectorstores = FAISS.from_documents(
    documents = chunks,
    embedding = embeddings
)

retriever = vectorstores.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 5, "score_threshold": 0.3}
)

template = (
    "You ara a strict, citation-focused assistant for private knowledge base. \n"
    "RULE: \n"
    "1) Use ONLY the provided context to answer. \n"
    "2) If the answer is not clearly contained in the context, say: 'I do not know based on the provided context'. \n"
    "3) Do not use the outside knowledge, guessing, or web information. \n"
    "4) If applicable, cite sources as (source: page) using the metadata. \n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.7-flash",
    temperature = 0
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_input = input("Question: ")
answer = rag_chain.invoke(user_input)
print(answer)

