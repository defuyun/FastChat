import json
import requests

data = {
    "message": "Be concise: Can you tell me something about gmail?",
    "session_id": "12345", # need to be unique for each session
    "temperature": 0.7, # default value
    "max_new_tokens": 1024
}

# Send the POST request to the endpoint
response = requests.post("https://llm-commercial-dev.eng.nianticlabs.com/model/falcon40b/api/chat", json=data)

# Print the response
print(response.text)