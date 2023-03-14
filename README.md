# ChatGPTQueryTool
Python script to process data through ChatGPT API

## Usage
Create a `.env` file with your chatGPT token
'''
CHAT_GPT_TOKEN = 'your token here'
'''

Create a conda environment (optional)
'''
conda create --name (your environment name)
'''

Then install packages
'''
pip install -r requirements.txt
'''
## Batch processing
In the chatGPT constructor, change the dataset path to your own dataset to pass through ChatGPT

**Note : ** the initial data should be a csv with one single column or else some changes to the code might be needed.

Change context and query to your desired data treatment. 

**Note : ** specify to ChatGPT to respond in only JSON or else the script will try to continuously query the api until it responds with correct JSON.

Specify batch size in process method and export file path
'''
chat = chatGPTBot("gpt-3.5-turbo")
chat.read("dataset/dataset.csv")
context = "You are instructed to only respond in JSON format and you must provide synonyms to these car manufacturer options."
query = "give me correct JSON format ONLY (ONLY JSON DATA, VERY IMPORTANT) with options as the keys and 1d array of 6 synonyms as value of this following array of vehicle manufacturer car options:"
chat.process(60, query, context)
chat.export("data/dataset_synonyms.csv", "csv")
'''
## whisper api call example

'''
chat = whisperGPTBot("whisper-1")
print(chat.call("audio/Deep Learning In 5 Minutes.mp3"))
'''

## normal chatgpt api call example

'''

'''