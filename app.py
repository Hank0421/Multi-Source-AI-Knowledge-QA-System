
'''from configparser import ConfigParser

# Set up the config parser
config = ConfigParser()
config.read("config.ini")

from langchain_google_genai import ChatGoogleGenerativeAI

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite", 
    google_api_key=config["Gemini"]["API_KEY"]
)

user_input = "科技大樓在哪裡？"

role_description = """
你是一個台灣人，請用繁體中文回答。
"""

messages = [
    ("system", role_description),
    ("human", user_input),
]

response_gemini = llm_gemini.invoke(messages)

print(f"問 : {user_input}")
print(f"Gemini : {response_gemini.content}")'''

from configparser import ConfigParser

# Set up the config parser
config = ConfigParser()
config.read("config.ini")

from langchain import FAISS
import pandas as pd

# Load dataset
animal_data = pd.read_csv("animal-fun-facts-dataset.csv")

# Embedding function - using SentenceTransformer

from langchain.embeddings import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Embedding function - using Google Generative AI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

embedding_function_gemini = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=config["Gemini"]["API_KEY"],
)


metadatas = []
for i, row in animal_data.iterrows():
    metadatas.append(
        {
            "Animal Name": row["animal_name"],
            "Source URL": row["source"],
            # "Media URL": row["media_link"],
            # "Wikipedia URL": row["wikipedia_link"],
        }
    )

animal_data["text"] = animal_data["text"].astype(str)

faiss_gemini = FAISS.from_texts(animal_data["text"].to_list(), 
                         # embedding_function, 
                         embedding_function_gemini,
                         metadatas)

faiss_gemini.similarity_search_with_score("What is ship of the desert?", 3)
faiss_gemini.similarity_search_with_score("What is 科技大樓?", 3)
faiss_gemini.similarity_search_with_score("What is ITRI?", 3)

# Save the FAISS index
faiss_gemini.save_local("faiss_db","animal-fun-facts")