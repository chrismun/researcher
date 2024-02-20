from autogen import oai

# create a text completion request
response = oai.Completion.create(
    config_list=[
        {
            "model": "chatglm2-6b",
            "base_url": "http://localhost:8000/v1",
            "api_type": "openai",
            "api_key": "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", # just a placeholder
        }
    ],
    prompt="Hi",
)
print(response)

# create a chat completion request
response = oai.ChatCompletion.create(
    config_list=[
        {
            "model": "chatglm2-6b",
            "base_url": "http://localhost:8000/v1",
            "api_type": "openai",
            "api_key": "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",
        }
    ],
    messages=[{"role": "user", "content": "Hi"}]
)
print(response)
