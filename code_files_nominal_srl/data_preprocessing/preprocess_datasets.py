import sys, os

import nltk
from nltk.corpus import semcor
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Lemma
import stanza

import json
from copy import deepcopy
import hashlib
from tqdm import tqdm
import requests

import copy

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# nltk.download('wordnet')
# nltk.download('semcor')

DIR = os.path.realpath(os.path.dirname(__file__))

sys.path.append(os.path.abspath(DIR + "/../../code_files_nominal_classification/data_preprocessing/"))
import wordnet_utils as wnutils

NO_ELEMENT_SYMBOL = "_"

# region ###########################   Common   ###########################

def save_dataset_split(data, save_path_dir, dataset_name, is_verbal, prompt_name = None, additional_info: dict = {}):
    ''' Types of datasets:
        - datasetX_verbal_sentences
        - datasetX_nominal_sentences
        - datasetX_verbalized_sentences_promptN
        - datasetX_nominalized_sentences_promptN
    '''
    ds_type = "_verbal" if is_verbal else "_nominal"
    ds_type += "ized" if prompt_name is not None else ""
    ds_prompt = f"_{prompt_name}" if prompt_name is not None else ""
    additional_info_str = ("@" + f"@".join([f"{key}={value}" for key,value in additional_info.items()])) if len(additional_info)>0 else ""
    file_name = f'{dataset_name}{ds_type}_sentences{ds_prompt}{additional_info_str}'
    file_path = os.path.join(save_path_dir, file_name+".json")

    if data is not None:
        os.makedirs(save_path_dir, exist_ok=True)
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    
    return file_path

def invero_request(text = "Marco is eating an apple.", lang = "EN", invero_url = 'http://127.0.0.1:3003/api/model'):
    if type(text) == list:
        http_input = []
        for sentence in text:
            http_input.append({'text':sentence, 'lang':lang})
    else:
        http_input = [{'text':text, 'lang':lang}]
    res = requests.post(invero_url, json = http_input)
    if res.status_code == 200:
        return json.loads(res.text)
    else:
        None

def amuse_request(text = "Pasquale is eating an apple.", lang = "EN", invero_url = 'http://127.0.0.1:3002/api/model'):
    if type(text) == list:
        http_input = []
        for sentence in text:
            http_input.append({'text':sentence, 'lang':lang})
    else:
        http_input = [{'text':text, 'lang':lang}]
    res = requests.post(invero_url, json = http_input)
    if res.status_code == 200:
        return json.loads(res.text)
    else:
        None

def generate_verbal_nominal_mapping(verbal_to_drf_filepath = "../datasets/base_nominal_classification/verbal_to_derivationally_related_forms.tsv"):
    verbal_to_nominal = {}
    nominal_to_verbal = {}
    with open(verbal_to_drf_filepath) as f:
        for line in f:
            elements = line.strip().split("\t")
            verbal_syn_name = elements[0]
            nominal_syn_names = elements[1:]
            
            if len(nominal_syn_names) == 0: continue
            
            verbal_to_nominal[verbal_syn_name] = nominal_syn_names
            for nominal_syn_name in nominal_syn_names:
                nominal_to_verbal.setdefault(nominal_syn_name, []).append(verbal_syn_name)
                
    return verbal_to_nominal, nominal_to_verbal

def generate_extended_verbal_nominal_mapping(gold_dataset_filepath = "../datasets/dataset_nominal_classification/dataset.json"):
    verbal_to_nominal = {}
    nominal_to_verbal = {}
    with open(gold_dataset_filepath, "r") as f:
        dataset = json.load(f)
        verbal_dataset = dataset["verbal_definitions"]
        nominal_dataset = dataset["nominal_definitions"]
        
        for frame_name, verbal_synsets in verbal_dataset.items():
            nominal_synsets = nominal_dataset.get(frame_name, {})
            verbal_synsets = list(verbal_synsets.keys())
            nominal_synsets = list(nominal_synsets.keys())
            
            for verbal_syn in verbal_synsets:
                verbal_to_nominal[verbal_syn] = nominal_synsets
            for nominal_syn in nominal_synsets:
                nominal_to_verbal[nominal_syn] = verbal_synsets
                
    return verbal_to_nominal, nominal_to_verbal

def split_by_predicate(formatted_element):
    ''' We need to process one predicate at a time, so we need to split in multiple equal sentences, one for each predicate '''
    splitted_element = []
    for current_word_group_idx in range( max(formatted_element["word_groups"])+1 ):

        words_predicates_group_i = [formatted_element["words_predicates"][word_idx] if element_word_group_idx == current_word_group_idx else "_" for word_idx, element_word_group_idx in enumerate(formatted_element["word_groups"])]

        if any(s != NO_ELEMENT_SYMBOL for s in words_predicates_group_i):

            p_i = str(next((i for i, s in enumerate(words_predicates_group_i) if s != NO_ELEMENT_SYMBOL), -1)) # taking the index of the first element != NO_ELEMENT_SYMBOL

            e_multi = copy.deepcopy(formatted_element)
            e_multi["id"] = formatted_element["id"] + f"_predicate_{p_i}"
            
            e_multi["words_predicates"] = words_predicates_group_i

            if "words_frames" in formatted_element:
                e_multi["words_frames"] = [formatted_element["words_frames"][j] if p != NO_ELEMENT_SYMBOL else NO_ELEMENT_SYMBOL for j, p in enumerate(words_predicates_group_i)]

            splitted_element.append(e_multi)
            
    return splitted_element

# endregion
# region ###########################   SemCor   ###########################

def preprocess_semcor():
    formatted_data = {}

    progress_bar = tqdm( zip(semcor.sents(), semcor.tagged_sents(tag="both")) )
    discarded_samples = 0
    progress_bar.set_description(f'Discarded samples: {discarded_samples}')
    for idx_sample, (sentence_words, tagged_sentence) in enumerate(progress_bar):

        sentence_synlemma = [NO_ELEMENT_SYMBOL]*len(sentence_words)
        sentence_syn = [NO_ELEMENT_SYMBOL]*len(sentence_words)
        sentence_pos = [NO_ELEMENT_SYMBOL]*len(sentence_words)
        sentence_groups = [NO_ELEMENT_SYMBOL]*len(sentence_words)
        sentence_position = 0

        for i, tagged_word in enumerate(tagged_sentence):
            lemma = tagged_word.label()

            for j, element_pos in enumerate(tagged_word.pos()):

                if element_pos[1] != None:
                    sentence_pos[sentence_position] = element_pos[1]
                if type(lemma) == Lemma:
                    sentence_synlemma[sentence_position] = lemma.name()
                    sentence_syn[sentence_position] = lemma.synset().name()

                sentence_groups[sentence_position] = i
                sentence_position += 1

        assert len(sentence_words) == len(sentence_syn)

        # correcting words in the sentence:
        quote_symbols = ["``", "''"]
        for i in range(len(sentence_words)):
            w = sentence_words[i]
            if w in quote_symbols: sentence_words[i] = '"'

        element_uid = hashlib.sha256((" ".join(sentence_words)+str(idx_sample)).encode()).hexdigest()
        sample = {
            "id": element_uid, # unique id of that sentence
            "sentence": " ".join(sentence_words), # the sentence in string format
            "words": sentence_words, # the list of words in the sentence
            "lemmas": sentence_synlemma, # the list of lemmas in the sentence

            "words_synsets": sentence_syn, # the list of synsets in the sentence (if the word is not a synset, then = "_")
            "words_pos": sentence_pos, # the list of POS in the sentence

            "word_groups": sentence_groups,
        }
        formatted_data[element_uid] = sample

    return formatted_data
    

def filter_formatted_dataset(formatted_dataset_data, verbal_to_nominal, nominal_to_verbal):
    filtered_verbal_dataset_data = {}
    filtered_nominal_dataset_data = {}

    for sample_id, sample in tqdm(formatted_dataset_data.items()):

        possible_verbal_candidate = False
        verbal_event_synsets = [NO_ELEMENT_SYMBOL]*len(sample["words_synsets"])
        verbal_words_predicates = [NO_ELEMENT_SYMBOL]*len(sample["words_synsets"])
        
        possible_nominal_candidate = False
        nominal_event_synsets = [NO_ELEMENT_SYMBOL]*len(sample["words_synsets"])
        nominal_words_predicates = [NO_ELEMENT_SYMBOL]*len(sample["words_synsets"])

        for syn_idx, syn in enumerate(sample["words_synsets"]):
            if syn in verbal_to_nominal:
                verbal_event_synsets[syn_idx] = verbal_to_nominal[syn]
                verbal_words_predicates[syn_idx] = sample["words"][syn_idx]
                possible_verbal_candidate = True
            if syn in nominal_to_verbal:
                nominal_event_synsets[syn_idx] = nominal_to_verbal[syn]
                nominal_words_predicates[syn_idx] = sample["words"][syn_idx]
                possible_nominal_candidate = True

        if possible_verbal_candidate:
            verbal_sample = deepcopy(sample)
            verbal_sample["synsets_event"] = verbal_event_synsets
            verbal_sample["words_predicates"] = verbal_words_predicates
            filtered_verbal_dataset_data[sample_id] = verbal_sample
            
        if possible_nominal_candidate:
            nominal_sample = deepcopy(sample)
            nominal_sample["synsets_event"] = nominal_event_synsets
            nominal_sample["words_predicates"] = nominal_words_predicates
            filtered_nominal_dataset_data[sample_id] = nominal_sample

    return {"verbal":filtered_verbal_dataset_data, "nominal":filtered_nominal_dataset_data}

# endregion