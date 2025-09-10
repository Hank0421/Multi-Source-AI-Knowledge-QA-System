from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

loader = TextLoader("state_of_the_union.txt",
                    autodetect_encoding=True,
)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
db = FAISS.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson?"
results = db.similarity_search_with_score(query, 1)
#print(results[0][0].page_content)

from langchain_google_genai import ChatGoogleGenerativeAI

from configparser import ConfigParser

# Set up config parser
config = ConfigParser()
config.read("config.ini")

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=config["Gemini"]["API_KEY"],
)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}
請用中英對照方式回答。
"""
)

output_parser = StrOutputParser()

'''chain = prompt | llm_gemini | output_parser

# query = "What did the president say about Ketanji Brown Jackson?"
query = "How much we are giving to the Ukraine?"
# query = "How many nations in NATO Alliance?"
results = db.similarity_search_with_score(query, 1)
print("Retrieved related content :")
print(results[0][0].page_content)
print("====================================================")

llm_result = chain.invoke(
    {
        "input": query,
        "context": [results[0][0]],
    }
)

print("Question: ", query)
print("LLM Answer: ", llm_result)'''

# LLM - Ollama

from langchain_ollama.llms import OllamaLLM
# ollama_llm = OllamaLLM(model="gemma3:1b")
#ollama_llm = OllamaLLM(model="phi4:latest")
ollama_llm = OllamaLLM(model="llama3.1:8b")


# query = "What did the president say about Ketanji Brown Jackson?"
# query = "How much we are giving to the Ukraine?"
query = "How many nations in NATO Alliance?"
results = db.similarity_search_with_score(query, 1)
print("Retrieved related content :")
print(results[0][0].page_content)
print("====================================================")

chain_ollama = prompt | ollama_llm | output_parser

ollama_result = chain_ollama.invoke(
    {
        "input": query,
        "context": [results[0][0]],
    }
)
print("Question: ", query)
print("LLM Answer from Ollama: ", ollama_result)

'''from configparser import ConfigParser

# Set up the config parser
config = ConfigParser()
config.read("config.ini")

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loader = TextLoader(
    "state_of_the_union.txt",
    autodetect_encoding=True,
)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, 
                                      chunk_overlap=0)
docs = text_splitter.split_documents(documents)
db = FAISS.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson?"
results = db.similarity_search_with_score(query, 1)
print(results[0][0].page_content)

from langchain_google_genai import ChatGoogleGenerativeAI

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=config["Gemini"]["API_KEY"],
)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}
請用中英對照方式回答。
"""
)

output_parser = StrOutputParser()

chain = prompt | llm_gemini | output_parser

# query = "What did the president say about Ketanji Brown Jackson?"
# query = "How much we are giving to the Ukraine?"
query = "How many nations in NATO Alliance?"
results = db.similarity_search_with_score(query, 1)
print("Retrieved related content :")
print(results[0][0].page_content)
print("====================================================")

llm_result = chain.invoke(
    {
        "input": query,
        "context": [results[0][0]],
    }
)

print("Question: ", query)
print("LLM Answer: ", llm_result)

# LLM - Ollama

from langchain_ollama.llms import OllamaLLM

ollama_llm = OllamaLLM(model="llama3.1:8b")
# ollama_llm = OllamaLLM(model="gemma3:1b")

# query = "What did the president say about Ketanji Brown Jackson?"
# query = "How much we are giving to the Ukraine?"
# query = "How many nations in NATO Alliance?"
query = "What did the president say about Danold Trump?"
results = db.similarity_search_with_score(query, 1)
print("Retrieved related content :")
print(results[0][0].page_content)
print("====================================================")

chain_ollama = prompt | ollama_llm | output_parser

ollama_result = chain_ollama.invoke(
    {
        "input": query,
        "context": [results[0][0]],
    }
)
print("Question: ", query)
print("LLM Answer from Ollama: ", ollama_result)'''