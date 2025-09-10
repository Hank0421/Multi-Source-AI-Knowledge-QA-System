# Multi-Source-AI-Knowledge-QA-System

## Project Description

This project is a multi-modal and multi-source knowledge Q&A system demo. It integrates various data sources such as PDF, Word, plain text, CSV, and images, and combines vector databases (FAISS, Qdrant) with multiple embedding models (HuggingFace, Google Gemini, SentenceTransformer) for semantic search and generative AI Q&A.  
It also demonstrates how to connect with LLMs like Google Gemini and Ollama, supporting answers in Traditional Chinese.

### Main Files

- **app.py**  
  Builds a FAISS vector store from a CSV animal facts dataset, supports Gemini embeddings and Q&A.

- **app_csv.py**  
  Uses SentenceTransformer to embed animal facts and save to FAISS, supports semantic search.

- **qna-bot.py**  
  Loads a prebuilt FAISS vector store, performs semantic search, and replies with the answer or "Sorry, I don't know." based on similarity score.

- **app_word.py**  
  Loads Word (docx) files, splits into chunks, builds vector store, and integrates Gemini for Q&A.

- **app-pdf.py**  
  Loads multiple PDFs, splits into chunks, builds vector store, and uses Gemini for multi-context comparison and Q&A.

- **app-article.py**  
  Loads plain text files, builds vector store, supports both Gemini and Ollama LLM Q&A.

- **app-Qdrant.py**  
  Uses Qdrant cloud vector store for ML case storage and retrieval, integrates Gemini for Q&A.

- **app-image.py**  
  Uses Gemini Vision for image understanding and generation, supports image upload and Traditional Chinese answers.

- **config.ini**  
  Stores API keys and Qdrant connection info.

### Requirements

- Python 3.8+
- Main packages: `langchain`, `langchain_community`, `langchain_google_genai`, `langchain_huggingface`, `langchain_ollama`, `qdrant-client`, `pandas`, `docx2txt`, `PyPDF2`, etc.
- Google Gemini API Key (fill in `config.ini`)
- For Ollama, install Ollama locally and download the required models

---

## 專案說明

本專案為多模態與多來源的知識問答系統範例，整合了 PDF、Word、純文字、CSV、圖片等多種資料來源，並結合向量資料庫（FAISS、Qdrant）與多種嵌入模型（如 HuggingFace、Google Gemini、SentenceTransformer），可進行語意檢索與生成式 AI 問答。  
本專案亦示範如何串接 Google Gemini 與 Ollama 等大型語言模型，並支援繁體中文回答。

### 主要檔案說明

- **app.py**  
  以 CSV 動物知識資料集建立 FAISS 向量庫，並可用 Gemini 產生嵌入與問答。

- **app_csv.py**  
  以 SentenceTransformer 產生動物知識資料集的向量，並儲存至 FAISS，支援語意檢索。

- **qna-bot.py**  
  載入已建立的 FAISS 向量庫，根據問題進行語意檢索，並根據相似度分數回應答案或「Sorry, I don't know.」。

- **app_word.py**  
  讀取 Word 檔（docx），分段後建立向量庫，並結合 Gemini 進行問答。

- **app-pdf.py**  
  讀取多份 PDF，分段後建立向量庫，並結合 Gemini 進行多段內容比較與問答。

- **app-article.py**  
  讀取純文字檔，建立向量庫，支援 Gemini 及 Ollama 兩種 LLM 問答。

- **app-Qdrant.py**  
  以 Qdrant 雲端向量庫儲存與查詢機器學習案例，並結合 Gemini 進行問答。

- **app-image.py**  
  以 Gemini Vision 處理圖片理解與生成，支援圖片上傳與繁體中文回答。

- **config.ini**  
  儲存 API 金鑰與 Qdrant 連線資訊。

### 執行需求

- Python 3.8+
- 主要套件：`langchain`, `langchain_community`, `langchain_google_genai`, `langchain_huggingface`, `langchain_ollama`, `qdrant-client`, `pandas`, `docx2txt`, `PyPDF2` 等
- 需申請 Google Gemini API Key，並填入 `config.ini`
- 若需使用 Ollama，請於本機安裝 Ollama 並下載對應模型

---

如需執行範例，請先設定好 `config.ini`，並依需求安裝相關 Python 套件。
