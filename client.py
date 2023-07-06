import json
import requests

data = {
    "message": """
given the list of function definitions
def create_group(name: str): creates a group
def delete_group(name: str): deletes a group
def create_entity(name: str): creates an entity
def add_to_group(group_name: str, entity: str): add entity to group

create a group and an entity that sounds like a pokemon, put the entity into the group, then delete the group
list the function calls that you need to make and give reason

Assistant:
1.
""",
    "session_id": "", # need to be unique for each session, leave empty to not record session
    "temperature": 0.7, # default value
    "top_k": 50,
    "top_p": 1.0,
    "max_new_tokens": 1024,
}

# Send the POST request to the endpoint
response = requests.post("https://llm-commercial-dev.eng.nianticlabs.com/model/falcon40b/api/chat", json=data)

# Print the response
print(response.text)