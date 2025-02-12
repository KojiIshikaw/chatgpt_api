import os
import pickle
import faiss
from dotenv import load_dotenv

# 推奨される import 先に変更（必要に応じてパッケージをアップデートしてください）
# ※ Embedding 用のクラス。モデル名を新しい埋め込みモデルに合わせる
from langchain_community.embeddings import OpenAIEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# チャットモデルは ChatOpenAI を利用（従来の OpenAI クラスは非推奨）
from langchain_community.chat_models import ChatOpenAI

# .env ファイルから環境変数を読み込む
load_dotenv()

# ① Embeddings の作成（新しい embedding モデル "text-embedding-3-small" を利用）
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"  # もしくは "text-embedding-3-large" 等、お好みで指定
)

# ② 保存済みの FAISS インデックスとドキュメントマッピング（docstore）の読み込み
index = faiss.read_index("faiss_index.bin")
with open("docstore.pkl", "rb") as f:
    docstore = pickle.load(f)
with open("index_to_docstore_id.pkl", "rb") as f:
    index_to_docstore_id = pickle.load(f)

# ③ FAISS ラッパーによる vectorstore の再構築
vectorstore = FAISS(embeddings, index, docstore, index_to_docstore_id)

# ========================
# ④ LLM（ChatGPT）インスタンスの作成
# ========================
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # 利用するモデル名（GPT-3.5-turbo や GPT-4 など）
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,        # 回答の多様性を制御（必要に応じて調整）
)

# ========================
# ⑤ 既存のベクトルストアから Retriever を作成
# ========================
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ========================
# ⑥ RetrievalQA チェーンの作成
# ========================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" のほか、 "refine" や "map_reduce" も選択可能
    retriever=retriever,
    verbose=True,        # 内部処理ログを表示（デバッグ時に有用）
)

# ========================
# ⑦ ユーザーからのクエリをチェーンに渡して回答を取得
# ========================
query = "Go2のバッテリーについて教えて"
answer = qa_chain.run(query)
print("回答:", answer)
