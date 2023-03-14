import openai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()


class chatGPTBot:
    def __init__(self, model):
        self.GPT_token = os.getenv("CHAT_GPT_TOKEN")
        self.org = os.getenv("CHAT_GPT_ORG")
        self.model = model
        self.processed_df = None
        self.processed_JSON = None

    def read(self, filename):
        self.data = pd.read_csv(filename)
        return

    def call(self, query, context=None):
        openai.api_key = self.GPT_token
        message = [{"role": "user", "content": query}]
        if context is not None:
            message.insert(0, {"role": "system", "content": context})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=message,
        )
        return response

    def process(self, batch_size, query, context=None):
        if self.data is None:
            print("No data to process")
            return
        raw_data = {}
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0
        processed_data = {"synonyms": {}}
        i = 0
        while i < len(self.data) // batch_size:
            error = False
            array = self.data[i * batch_size : (i + 1) * batch_size]
            array = array.to_numpy().flatten().tolist()
            array_string = "[" + ",".join(array) + "]"
            queryText = f"{query}:{array_string}"
            response = self.call(context, queryText)
            print(f"Completed batch {i+1} of {len(self.data) // batch_size}")
            raw_data = {}
            raw_data[f"query{i}"] = (
                response["choices"][0]["message"]["content"]
                .replace("\n", " ")
                .replace('"', '"')
            )
            raw_data[f"query{i}" + "_usage"] = (
                str(response["usage"]).replace("\n", " ").replace('"', '"')
            )

            for (key, value) in raw_data.items():
                try:
                    value = value[str(value).index("{") :]
                    if "_usage" in key:
                        val = json.loads(value)
                        print(
                            f"Total tokens for this batch: {val['total_tokens']}, estimated price: {val['total_tokens']*0.000002}$"
                        )
                        completion_tokens += val["completion_tokens"]
                        prompt_tokens += val["prompt_tokens"]
                        total_tokens += val["total_tokens"]
                    else:
                        for (k, v) in json.loads(value).items():
                            processed_data["synonyms"][k] = v
                except json.decoder.JSONDecodeError as e:
                    error_name = datetime.now().strftime(
                        "error_logs/error_log-%Y-%m-%d_%H-%M-%S.json"
                    )
                    error = True
                    print(e, error_name)
                    with open(error_name, "w") as outfile:
                        json.dump(raw_data, outfile, indent=4)
                    break
            if not error:
                i += 1

        print("Batch processing complete")
        print("Total completion tokens: ", completion_tokens)
        print("Total prompt tokens: ", prompt_tokens)
        print("Total tokens: ", total_tokens)
        print("Estimated price: ", total_tokens * 0.000002, "$")

        self.processed_JSON = processed_data
        options, synonyms = [k for k, v in processed_data["synonyms"].items()], [
            v for k, v in processed_data["synonyms"].items()
        ]
        df_data = {"original_option": options, "synonyms": synonyms}
        df = pd.DataFrame(df_data)
        self.processed_df = df
        return

    def export(self, output_file, output_type="json"):
        if output_type == "json" and self.processed_JSON is not None:
            with open(output_file, "w") as outfile:
                json.dump(self.processed_JSON, outfile, indent=4)
        elif output_type == "csv" and self.processed_df is not None:
            self.processed_df.to_csv(output_file, index=False)
        else:
            print("Invalid output type")
        return


class whisperGPTBot:
    def __init__(self, model):
        self.model = model
        self.GPT_token = os.getenv("CHAT_GPT_TOKEN")
        openai.api_key = self.GPT_token

    def call(self, file):
        audio = open(file, "rb")
        text = openai.Audio.transcribe(self.model, audio)
        return text


##batch
# chat = chatGPTBot("gpt-3.5-turbo", "dataset/all_bb_distinct_options.csv")
# context = "You are instructed to only respond in JSON format and you must provide synonyms to these car manufacturer options."
# query = "give me correct JSON format ONLY (ONLY JSON DATA, VERY IMPORTANT) with options as the keys and 1d array of 6 synonyms as value of this following array of vehicle manufacturer car options:"
# chat.process(60, query, context)
# chat.export("data/dataset_synonyms_test_2.csv", "csv")

##whisperAPI
# chat = whisperGPTBot("whisper-1")
# print(chat.call("audio/Deep Learning In 5 Minutes.mp3"))

##chatAPI


def main():
    chat = chatGPTBot("gpt-3.5-turbo")
    query = "I want you to extract all medicals entity in this abstract. Organise your extraction by categories:Loss of p53's tumor-suppressive function, either via TP53 mutation or hyperactive p53 inhibitory proteins, is one of the most frequent events in the development of human cancer. Here, we describe a strategy of pharmacologically inhibiting iASPP, a negative regulator of p53, to restore wild-type p53's tumor-suppressive function. iASPP knockdown in the colon cancer cell line HCT116 efficiently promoted p53's transcriptional activity and induced p53-dependent cell death, suggesting a key role for iASPP in silencing p53 in this cell line. Screening of a preclinical and clinical drug library using isogenic HCT116 cell models revealed that cyclin-dependent kinase 9 (CDK9) inhibitors preferentially inhibit p53+/+, rather than p53-/-, cells. Mechanistically, CDK9 inhibitors downregulated iASPP at the transcriptional level. This downregulation was dose- and time-dependent. CDK9 inhibitors further showed synergistic effects in killing p53+/+ HCT116 cells when combined with the MDM2 inhibitor Nutlin-3. In a large TCGA pan-cancer cohort, iASPP overexpression predicted poor overall survival (OS) in wild-type p53 patients, with worse OS observed when MDM2 was simultaneously overexpressed. Our study identifies CDK9 inhibitors as p53-reactivating agents, and proposes a strategy to treat cancer by efficiently reactivating p53 via the concurrent inhibition of iASPP and MDM2."
    response = chat.call(query)
    with open("data/.json", "w") as outfile:
        json.dump(response, outfile, indent=4)


if __name__ == "__main__":
    main()
