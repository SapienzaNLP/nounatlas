import json, copy
from typing import List
from pprint import pprint
import time
from tqdm import tqdm
import tiktoken
from openai import OpenAI
import google.generativeai as genai

MODELS = {
    "gpt-3.5-turbo-1106": {
        "provider": "openai",
        "model_name":"gpt-3.5-turbo-1106",
        "price_per_prompt_token": 0.0010/1000,
        "price_per_response_token": 0.0020/1000,
        "type": "gpt_chat",
        "base_url": "https://api.openai.com/v1",
        "requests_per_minute": 3
    },
    
    "gpt-3.5-turbo-0613": {
        "provider": "openai",
        "model_name":"gpt-3.5-turbo-0613",
        "price_per_prompt_token": 0.0015/1000,
        "price_per_response_token": 0.0020/1000,
        "type": "gpt_chat",
        "base_url": "https://api.openai.com/v1",
        "requests_per_minute": 3
    },
    
    "gpt-3.5-turbo-instruct": {
        "provider": "openai",
        "model_name":"gpt-3.5-turbo-instruct",
        "price_per_prompt_token": 0.0015/1000,
        "price_per_response_token": 0.0020/1000,
        "type": "gpt_instruct",
        "base_url": "https://api.openai.com/v1",
        "requests_per_minute": 3
    },
    
    "gpt-4-1106": {
        "provider": "openai",
        "model_name":"gpt-4-1106-preview",
        "price_per_prompt_token": 0.0100/1000,
        "price_per_response_token": 0.0300/1000,
        "type": "gpt_chat",
        "base_url": "https://api.openai.com/v1",
        "requests_per_minute": 3
    },
    
    "gpt-4": {
        "provider": "openai",
        "model_name":"gpt-4",
        "price_per_prompt_token": 0.0300/1000,
        "price_per_response_token": 0.0600/1000,
        "type": "gpt_chat",
        "base_url": "https://api.openai.com/v1",
        "requests_per_minute": 3
    },
    
    "text-davinci-003": {
        "provider": "openai",
        "model_name":"text-davinci-003",
        "price_per_prompt_token": 0.0200/1000,
        "price_per_response_token": 0.0400/1000,
        "type": "gpt_instruct",
        "base_url": "https://api.openai.com/v1",
        "requests_per_minute": 3
    },
    
    "text-davinci-002": {
        "provider": "openai",
        "model_name":"text-davinci-002",
        "price_per_prompt_token": 0.0200/1000,
        "price_per_response_token": 0.0400/1000,
        "type": "gpt_instruct",
        "base_url": "https://api.openai.com/v1",
        "requests_per_minute": 3
    },
    
    "mixtral-8x7b": {
        "provider": "fireworks",
        "model_name":"accounts/fireworks/models/mixtral-8x7b",
        "price_per_prompt_token": 0.4000/1000000,
        "price_per_response_token": 1.6000/1000000,
        "type": "gpt_instruct",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "requests_per_minute": 10
    },
    
    "gemini-pro": {
        "provider": "google",
        "model_name": "models/gemini-pro",
        "price_per_prompt_token": 0,
        "price_per_response_token": 0,
        "type": "google_instruct",
        "base_url": "https://www.google.com",
        "requests_per_minute": 60
    }
    
    
}
MODELS["gpt-3.5"] = MODELS["gpt-3.5-turbo-1106"]

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-1106"):
    """
    Returns the number of tokens used by a list of messages.
    gpt-3.5-turbo-0301 behaves differently, don't use it (is deprecated anyways)
    """
    model_name = MODELS[model]["model_name"]
    price_per_token = MODELS[model]["price_per_prompt_token"]
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 3 # every message follows <|im_start|>{role/name}\n{content}<|im_end|> ("\n" are counted tokens apparently)
    tokens_per_assistant_message = 3 # every reply is primed with <|im_start|>assistant\n. Add this at the end of every request to prompt a response from the model
    tokens_per_name = 1 # If a name is specified, add an additional token
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += tokens_per_assistant_message
    return {
        "num_tokens": num_tokens,
        "request_price": num_tokens*price_per_token
    }

def num_tokens_from_response(response:str, model="gpt-3.5-turbo-1106"):
    """
    Returns the number of tokens used by a list of messages.
    gpt-3.5-turbo-0301 behaves differently, don't use it (is deprecated anyways)
    """
    model_name = MODELS[model]["model_name"]
    price_per_token = MODELS[model]["price_per_response_token"]
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(response))
    return {
        "num_tokens": num_tokens,
        "request_price": num_tokens*price_per_token
    }

def ask_gpt_chat(client, prompt, model="gpt-3.5", examples: List[str] = [], system_directives: List[str] = [], temperature = 1, top_p = 1, **kwargs):
    model_name = MODELS[model]["model_name"]
    messages = []
    for system_rule in system_directives:
        messages.append({"role": "system", "content": system_rule})
        
    for example_idx in range(0,len(examples)):
        one_shot_prompt = examples[example_idx][0]
        one_shot_response = examples[example_idx][1]
        messages.append({"role": "user", "content": one_shot_prompt})
        messages.append({"role": "assistant", "content": one_shot_response})   
        
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return {
        "response": response.choices[0].message.content,
        "finish_reason": response.choices[0].finish_reason,
        "prompt_tokens": response.usage.prompt_tokens,
        "response_tokens": response.usage.completion_tokens,
        "full_response_object": response
    }
    
def ask_gpt_instruct(client, prompt, model="gpt-3.5-turbo-instruct", examples: List[str] = [], system_directives: List[str] = [], temperature = 1, top_p = 1, **kwargs):
    model_name = MODELS[model]["model_name"]
    
    full_prompt = ""
    full_prompt += (". ".join(system_directives))
    full_prompt += ".\n"

    for example_idx in range(0,len(examples)):
        one_shot_prompt = examples[example_idx][0]
        one_shot_response = examples[example_idx][1]
        full_prompt += f"Human: {one_shot_prompt}\n"
        full_prompt += f"AI: {one_shot_response}\n"
    
    full_prompt += f"Human: {prompt}\nAI:"
    
    response = client.completions.create(
        model=model_name,
        prompt=full_prompt,
        temperature=temperature,
        max_tokens=1024,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
        stop= ["Human:", "AI:"]
    )
    
    return {
        "response": response.choices[0].text,
        "finish_reason": response.choices[0].finish_reason,
        "prompt_tokens": response.usage.prompt_tokens,
        "response_tokens": response.usage.completion_tokens,
        "full_response_object": response
    }
    
def ask_google_instruct(api_key, prompt, model="gpt-3.5-turbo-instruct", examples: List[str] = [], system_directives: List[str] = [], temperature = 1, top_p = 1, **kwargs):
    genai.configure(api_key=api_key)
    model_name = MODELS[model]["model_name"]
    
    prompt_parts = []
    
    if len(system_directives) > 0:
        prompt_parts.append(", ".join(system_directives))
        
    for example_idx in range(0,len(examples)):
        one_shot_prompt = examples[example_idx][0]
        one_shot_response = examples[example_idx][1]
        prompt_parts.append(f"input: {one_shot_prompt}\n")
        prompt_parts.append(f"output: {one_shot_response}\n")
    
    prompt_parts.append(f"input: {prompt}\n")
    prompt_parts.append("output:")
    
    
    generation_config = genai.GenerationConfig(
        temperature = temperature,
        top_p = top_p,
        max_output_tokens = 4196,
        stop_sequences = ["input:", "output:"],
    )
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT","threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH","threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT","threshold": "BLOCK_NONE"},
    ]
    
    model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config, safety_settings=safety_settings)
    response = model.generate_content(prompt_parts)
    
    response_parts = [part.text for part in response.candidates[0].content.parts]
    
    prompt_tokens = model.count_tokens(prompt_parts).total_tokens if len(prompt_parts) > 0 else 0
    response_tokens = model.count_tokens(response_parts).total_tokens if len(response_parts) > 0 else 0
    return {
        "response": " ".join(response_parts),
        "finish_reason": response.candidates[0].finish_reason,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "full_response_object": response
    }