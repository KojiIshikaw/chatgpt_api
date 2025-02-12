import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # .env ファイルを読み込む

# OpenAIクライアントの作成
client = OpenAI(
  organization='org-0QYOrBgl5NSLGKIINN6hcJ8Y',
  project='proj_FasJ3WxW7uCMcHNDZ9gs3iyn',
  api_key=os.getenv("OPENAI_API_KEY"),
)


stream = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages=[{"role": "user", "content": "エンジニアリングにおける開発について教えて"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

# def get_chatgpt_response(prompt):
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",  # "gpt-4o-mini"
#             messages=[
#                 {"role": "system", "content": "あなたは優秀なアシスタントです。"},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=150,  # 応答の最大トークン数
#             temperature=0.7  # 生成のクリエイティビティを制御
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"エラーが発生しました: {e}"

# if __name__ == "__main__":
#     user_prompt = input("ユーザー: ")
#     answer = get_chatgpt_response(user_prompt)
#     print(f"ChatGPT: {answer}")
