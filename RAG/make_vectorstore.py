import os
import pickle
import faiss
from dotenv import load_dotenv

# 新しい推奨インポート方法（langchain_communityパッケージから）
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# .env ファイルから環境変数を読み込む
load_dotenv()

# 1. ドキュメントの読み込みと分割
with open("data/go2_about.txt", "r", encoding="utf-8") as f:
    text = f.read()

# テキストを適切なサイズに分割（例：1チャンクあたり1000文字、200文字のオーバーラップ）
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
docs = text_splitter.create_documents([text])

# 2. Embeddings の作成
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 3. ベクトルストアの作成（FAISSを例に）
vectorstore = FAISS.from_documents(docs, embeddings)

# --- 保存時 ---
# FAISSのインデックスを保存
faiss.write_index(vectorstore.index, "faiss_index.bin")

# ドキュメント（およびメタデータ）のマッピングをpickleで保存
with open("docstore.pkl", "wb") as f:
    pickle.dump(vectorstore.docstore, f)

# ※ 新たに index_to_docstore_id も保存する（最新バージョンではこの mapping が必要）
with open("index_to_docstore_id.pkl", "wb") as f:
    pickle.dump(vectorstore.index_to_docstore_id, f)
