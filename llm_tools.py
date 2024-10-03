import google.generativeai as genai
import time
import os
import multiprocessing
import re
import ollama
from pathlib import Path
from google.generativeai import GenerativeModel
from abc import ABC, abstractmethod

from file_tools import aggregate_files

NUM_PROCESS = 1

class LLM(ABC):
    @abstractmethod
    def summarize(self, text):
        pass


class Gemini(LLM):
    def __init__(self, prompt: str, model: GenerativeModel):
        self.model = model
        self.prompt = prompt

    def summarize(self, text: str):
        i=0
        while i<3:
            try:
                full_prompt = self.prompt+ f"\n Summarize the following article : {text}"
                full_prompt = re.sub(r'[\uD800-\uDFFF]', '', full_prompt)
                response = self.model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                print(e)
                i+=1
                time.sleep(1)
        return ""



class SeqGemini(LLM):
    def __init__(self, prompt: str, seq_instruct: list[(str, str)],model: GenerativeModel):
        self.model = model
        self.prompt = prompt
        self.seq_instruct = seq_instruct

    def summarize(self, text: str):
        summary = ""

        chat = self.model.start_chat(history=[])

        full_prompt = self.prompt + f"\n Here is the article : {text} \n"
        full_prompt = re.sub(r'[\uD800-\uDFFF]', '', full_prompt)
        chat.send_message(full_prompt)

        for i in range(len(self.seq_instruct)):
            summary+=f"\n\n## {self.seq_instruct[i][0]}\n\n"
            response = chat.send_message(self.seq_instruct[i][1])
            summary += response.text
            time.sleep(3)
        return summary


class GradLlama31(LLM):
    def __init__(self, prompt: str):
        self.prompt = prompt

    def summarize(self, text: str):
        full_prompt = self.prompt+"\n"+"Here is the state of the art : \n"+ text
        temp = ollama.chat(model='gradient-llama3.1-8b:latest',
                               messages=[{'role': 'user', 'content': full_prompt}],
                               stream=False)
        return temp



class LLMFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_chatbot(**kwargs):
        pass


class GeminiFactory(LLMFactory):
    @staticmethod
    def create_chatbot(prompt: str, model_name="gemini-1.5-flash"):
        model = genai.GenerativeModel(model_name)
        return Gemini(prompt, model)


class SeqGeminiFactory(LLMFactory):
    @staticmethod
    def create_chatbot(prompt: str, seq_instruct: list[(str, str)], model_name="gemini-1.5-flash"):
        model = genai.GenerativeModel(model_name)
        return SeqGemini(prompt, seq_instruct, model)

def scrapper(scrapper_id: int, pdf_extractor, llm_factory: LLMFactory, articles: list[(str, str)], **kwargs):
    summaries_path = '.summaries'
    os.makedirs(summaries_path, exist_ok=True)
    summaries_paths = []
    article_id = scrapper_id

    # Iterate over the shared list and process each item
    while article_id < len(articles):
        print(f"Task {scrapper_id} is processing {articles[article_id][0]}...")
        article_iid = (articles[article_id][1].split('/')[-1]).split('.pdf')[0]
        file_name = f'summary_{article_iid}.txt'
        file_path = os.path.join(summaries_path, file_name)

        if (not os.path.exists(file_path)) or os.path.getsize(file_path) == 0:
            with open(file_path, 'w+') as f:
                # Simulate work for each list item
                article_text = pdf_extractor(articles[article_id][1])
                llm = llm_factory.create_chatbot(**kwargs)
                summary = llm.summarize(article_text)
                f.write(f'# {articles[article_id][0]}\n')
                f.write(summary)  # Write to the temp file
                f.write('\n---\n')
        time.sleep(0.5)
        article_id = article_id + NUM_PROCESS
        summaries_paths.append(file_path)
    print(f"Task {scrapper_id} done, results saved to {summaries_path}")
    return summaries_paths

def run_parallel_summarizer(pdf_extractor, llm_factory, articles: list[(str, str)], output_file="final_output.md", **kwargs):
    # Use a Pool to parallelize tasks and collect the file names
    with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
        # Call process_task for each task_id and get the file names as results
        file_names = pool.starmap(scrapper, [(i, pdf_extractor, llm_factory, articles, kwargs) for i in range(NUM_PROCESS)])

        print("All tasks completed!")
        file_names_temp = [x for xs in file_names for x in xs]

        # Aggregate all task output files into a single file
        aggregate_files(file_names_temp, output_file)

def run_single_summarizer(pdf_extractor, llm_factory, articles, output_file="final_output.md", **kwargs):
    file_names = scrapper(0 ,pdf_extractor, llm_factory, articles, **kwargs)
    aggregate_files(file_names, output_file)


def configure_gemini(api: str):
    genai.configure(api_key=api)


