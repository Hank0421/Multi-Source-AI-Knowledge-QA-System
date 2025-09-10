'''#import vectorstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

prebuilt_faiss = FAISS.load_local("faiss_db", embedding_function, "animal-fun-facts", allow_dangerous_deserialization=True)

#question = "What is ship of the desert?"
question = "What is 科技大樓?"
results = prebuilt_faiss.similarity_search_with_score(question, 3)

# 處理查詢結果
if results:
    # 取分數最低的（最相似）
    best_result, score = results[0]
    if score > 1:
        print(f"Question: {question}")
        print("Sorry, I don't know.")
    else:
        print(f"Question: {question}")
        print(f"Answer: {best_result.page_content}")
else:
    print(f"Question: {question}")
    print("Sorry, I don't know.")'''


# import vectorstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

prebuilt_faiss = FAISS.load_local("faiss_db", 
                                  embedding_function,
                                  "animal-fun-facts",
                                  allow_dangerous_deserialization=True)

# question = "What is ship of the desert?"
question = "What is 科技大樓?"
results = prebuilt_faiss.similarity_search_with_score(question, 3)
if results[0][1] > 1.0:
    print(f"問 : {question}")
    print("答：這個跟動物沒關係吧")
else:
    print(f"問 : {question}")
    print(f"答 : {results[0][0].page_content}")
