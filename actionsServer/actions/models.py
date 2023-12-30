import requests
import json
from typing import List, Any, Dict
import os
from .document import *
from .stages import *
from .vecQuery import *
def callGpt(
        messages: List[Dict[str,str]],
        temperature: float = 0.3
    ) -> str:
    url: str = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "max_tokens": 800,
        "presence_penalty": 0,
        "frequency_penalty": 0
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer '+ os.environ.get("ChatGPTApiKey", "None")
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        return str(response.json()['choices'][0]['message']['content'])
    except:
        return "OPEN API DOWN -> ERROR:"+str(response.text)


client = createClient()
assert(checkClient(client))



#
def callGPTByRAG(userId: str, Stype: str, userText: str) -> str:

    botReply: str = ""
    backgrounds = search_redis(user_query=text, k=4)
    prompts = [{
            "role": "system",
            "content": f"reply like a teacher with your background. you background: \n{'\n'.join([i[0] for i in backgrounds])}"
        },{
            "role": "user",
            "content": userText
        }]
        
        # botReply += "\n***"+"\n***".join([str(unit) for unit in prompts])
        botReply += "\n"+callGpt(prompts, 0.3)
    return botReply       


