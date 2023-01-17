import os
import re
from configparser import ConfigParser
from typing import List
from time import time, sleep
from uuid import uuid4

import openai

from constants import (
    PROMPT_NOTES_TEMPLATE,
    PROMPT_RESPONSE_TEMPLATE,
    RAVEN_NAME,
    USER_NAME,
    CONFIG_DEFAULT_KEY,
    CONFIG_OPENAI_API_KEY,
    INPUT_KEY,
    NOTES_KEY,
    CONVERSATION_KEY,
    CHAT_LOG_DIR,
)

from utils import (
    save_file,
    load_json,
    log_json_message,
    cosine_similarity,
)

config = ConfigParser()
config.read("config.ini")
openai.api_key = config[CONFIG_DEFAULT_KEY][CONFIG_OPENAI_API_KEY]


def gpt3_embedding(content: str, engine='text-embedding-ada-002') -> List[float]:
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

def fetch_memories(vector: List[float], logs: List[dict], count: int) -> List[dict]:
    scores = list()
    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = cosine_similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered


def load_convo() -> List[dict]:
    files = os.listdir(CHAT_LOG_DIR)
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json(os.path.join(CHAT_LOG_DIR, file))
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return ordered


def summarize_memories(memories: List[dict]) -> List[str]:  # summarize a block of memories into one payload
    memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically
    block = ''
    for mem in memories:
        block += '%s: %s\n\n' % (mem['speaker'], mem['message'])
    block = block.strip()
    prompt = PROMPT_NOTES_TEMPLATE.replace(INPUT_KEY, block)
    # TODO - do this in the background over time to handle huge amounts of memories
    notes = gpt3_completion(prompt)
    return notes


def get_last_messages(conversation: List[dict], limit: int) -> str:
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ''
    for i in short:
        output += '%s: %s\n\n' % (i['speaker'], i['message'])
    output = output.strip()
    return output


def gpt3_completion(prompt:str, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'RAVEN:']) -> str:
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def format_log_message(speaker: str, vector, message: str) -> dict:
    return {
        'speaker': speaker,
        'time': time(),
        'vector': vector, # TODO this can be compressed into a hash
        'message': message,
        'uuid': str(uuid4())
    }


def process_user_input(user_input: str) -> List[float]:
    #### vectorize and save user input
    input_embedding = gpt3_embedding(user_input, engine='text-embedding-ada-002')
    info = format_log_message(USER_NAME, input_embedding, user_input)
    log_json_message(info, USER_NAME, CHAT_LOG_DIR)
    return input_embedding


def generate_corpus(user_input_vector: List[float]) -> str:
    #### load conversation
    conversation = load_convo()
    #### compose corpus (fetch memories, etc)
    memories = fetch_memories(user_input_vector, conversation, 10)  # pull episodic memories
    # TODO - fetch declarative memories (facts, wikis, KB, company data, internet, etc)
    notes = summarize_memories(memories)
    recent = get_last_messages(conversation, 4)
    prompt: str = PROMPT_RESPONSE_TEMPLATE.replace(NOTES_KEY, notes)
    return prompt.replace(CONVERSATION_KEY, recent)


def process_gpt_output(prompt: str) -> str:
    output = gpt3_completion(prompt)
    output_embedding = gpt3_embedding(output, engine='text-embedding-ada-002')
    info = format_log_message(RAVEN_NAME, output_embedding, output)
    log_json_message(info, RAVEN_NAME, CHAT_LOG_DIR)
    return output


if __name__ == '__main__':
    while True:
        #### get user input
        user_input = input(f'\n\n{USER_NAME}: ')

        #### vectorize and save user input
        user_input_vector = process_user_input(user_input)

        #### Generate corpus prompt
        prompt = generate_corpus(user_input_vector)

        #### generate response, vectorize, save, etc
        output = process_gpt_output(prompt)

        #### print output
        print(f'\n\n{RAVEN_NAME}: {output}')
