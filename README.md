# ChatGPTQueryTool
Python script to process data through ChatGPT API

## Usage
Create a `.env` file with your chatGPT token
'''
CHAT_GPT_TOKEN = 'your token here'
'''
In the chatGPT constructor, change the dataset path to your own dataset to pass through ChatGPT

**Note : ** the initial data should be a csv with one single column or else some changes might be needed.

Change context and query to your desired data treatment. 

**Note : ** specify to ChatGPT to respond in only JSON or else the script will try to continuously query the api until it responds with correct JSON.

Specify batch size in process method and export file path
