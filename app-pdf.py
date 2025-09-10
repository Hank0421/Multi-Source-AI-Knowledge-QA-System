from langchain_community.vectorstores import FAISS
from configparser import ConfigParser

# Get pdf data by PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

# Load pdf file
loader = PyPDFLoader("Jabra-Talk25.pdf")
data = loader.load()

loader2 = PyPDFLoader("Jabra-Talk45.pdf")
data2 = loader2.load()

data_all = data + data2

# Set up config parser
config = ConfigParser()
config.read("config.ini")

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=100,
    chunk_overlap=20,
)

docs = text_splitter.split_documents(data_all)
db = FAISS.from_documents(docs, embeddings)


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    api_key=config["Gemini"]["API_KEY"]
)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
"""
    Answer the following question based only on the provided context,
    besides the answer, please also provide a list of comparison, 
    just list item, not table.
    {context}
    {context1}
    {context2}
    {context3}
    {context4}
    Question: {input}
    請用繁體中文回答
"""
)

#f rom langchain.chains.combine_documents import create_stuff_documents_chain

# document_chain = create_stuff_documents_chain(llm, prompt)

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

document_chain = prompt | llm | output_parser

# query = "請列出Jabra Talk25與Jabra Talk45分別是為什麼而打造的?"
# query = "哪一個產品的修眠模式待機時間比較長？"
# query = "兩個產品的USB接頭是否相同？"
query = """
我有一個倉庫，裡面的溫度是50度C，
請問哪一個產品可以存放在我的倉庫中？
"""
result = db.similarity_search_with_score(query, 4)

llm_result = document_chain.invoke(
    {
        "input": query,
        "context": "",
        "context1": {
            "Product Name": result[0][0].metadata["source"][:-4],
            "content": result[0][0].page_content,
        },
        "context2": {
            "Product Name": result[1][0].metadata["source"][:-4],
            "content": result[1][0].page_content,
        },
        "context3": {
            "Product Name": result[2][0].metadata["source"][:-4],
            "content": result[2][0].page_content,
        },
        "context4": {
            "Product Name": result[3][0].metadata["source"][:-4],
            "content": result[3][0].page_content,
        }
    }
)

print("Question: ", query)
print("LLM Answer: ", llm_result)