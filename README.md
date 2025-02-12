

# OpenAI API keys

https://platform.openai.com/api-keys

.env
```sh
# .env
OPENAI_API_KEY=your_api_key_here
```

# setup

```sh
python -m venv venv
source venv/bin/activate
pip install openai
pip install python-dotenv
pip install jinja2 typeguard
pip install tiktoken
pip install -U langchain-community
pip install langchain openai faiss-cpu  # FAISSを使う場合
```

# run

```sh
source venv/bin/activate
source .env
python3 chatgpt_example.py
```

# error

下記となる場合は、さらに課金必要。
```sh
openai.RateLimitError: Error code: 429
```