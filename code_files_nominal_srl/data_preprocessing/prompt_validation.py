import os, sys, json, csv
import string, re
from nltk.corpus import wordnet
import nltk

# region Stanza
import stanza
from tqdm import tqdm
# stanza.download('en')
class StanzaProcess:
    def __init__(self) -> None:
        self.nlp_words = stanza.Pipeline(lang='en',processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True, logging_level='ERROR')
        self.nlp_phrase = stanza.Pipeline(lang='en',processors='tokenize,mwt,pos,lemma', logging_level='ERROR')

    def process(self, words_or_sentence):
        """
        Process a list of words using Stanza.
        Args:
            words_or_sentence (str | list): string of the phrase or list of words.
        Returns:
            list of dict: List of dictionaries containing token information.
        """
        stanza_result = self.nlp_words([words_or_sentence]) if type(words_or_sentence) == list else self.nlp_phrase(words_or_sentence)
        result = []
        for s in stanza_result.sentences:
            for word in s.words:
                result.append({
                    "word":                word.text,
                    "lemma":               word.lemma,
                    'pos':                 word.upos,
                    'xpos':                word.xpos,
                    # 'dependency_head':     word.head,
                    # 'dependency_relation': word.deprel,
                })
        return result
    
class NLTKProcess:
    def __init__(self) -> None:
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def process(self, words_or_sentence):
        """
        Process a list of words using NLTK.
        Args:
            words_or_sentence (str | list): string of the phrase or list of words.
        Returns:
            list of dict: List of dictionaries containing token information.
        """
        if type(words_or_sentence) == str:
            words_or_sentence = nltk.word_tokenize(words_or_sentence)
        tagged_words = nltk.pos_tag(words_or_sentence)
        result = []
        for word, pos in tagged_words:
            wordnet_pos = self.get_wordnet_pos(pos)
            result.append({
                "word":                word,
                "lemma":               self.lemmatizer.lemmatize(word, pos=wordnet_pos),
                'pos':                 pos,
                'xpos':                pos,
            })
        return result

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('V'):
            return 'v' # Verb
        elif treebank_tag.startswith('J'):
            return 'a' # Adjective
        elif treebank_tag.startswith('R'):
            return 'r' # Adverb
        elif treebank_tag.startswith('N'):
            return 'n' # Noun
        
        else:
            return 'n' # The default value for the lemmatizer

stanzaProcess = StanzaProcess()
nltkProcess = NLTKProcess()
# endregion


def validate_single_prompt_3_object(formatted_element, nlpProcess: str = "Stanza"):
    
    if nlpProcess == "NLTK":
        nlp = nltkProcess
    else:
        nlp = stanzaProcess

    good_translation = True
    type_of_result = "good_translation"

    target_predicate_indexes = [i for i, predicate in enumerate(formatted_element["words_predicates"]) if predicate != "_"] # to take also composed predicates such as "took place"

    predicate = " ".join(formatted_element["words_predicates"][i] for i in target_predicate_indexes)
    corresponding_syns = formatted_element["synsets_event"][target_predicate_indexes[0]] # all the splitted predicates have the same corresponding_syns, so just look at the first one
    
    corresponding_lemmas = []
    for corresponding_syn in corresponding_syns:
            lemmas = wordnet.synset(corresponding_syn).lemmas()
            for lemma in lemmas:
                corresponding_lemmas.append(lemma.name().replace("_"," ").lower())
    
    try:            
        translated_sentence, deverbal_noun = re.sub(r"\n+","\n", formatted_element["chatgpt_output"].strip()).split("\n")
        # _, deverbal_noun = deverbal_noun[:-1].split('"')
        deverbal_noun = deverbal_noun.split()[-1].strip(string.punctuation)
        translated_sentence = translated_sentence.strip()
        deverbal_noun = deverbal_noun.strip().lower()
        formatted_element["translated_sentence"] = translated_sentence
        formatted_element["deverbal_noun"] = deverbal_noun
    except:
        good_translation = False
        type_of_result = "error_formatting"
        
        formatted_element["validation_result"] = good_translation
        formatted_element["validation_type"] = type_of_result
        return good_translation, type_of_result
            
    translated_tagged_tokens = nlp.process(translated_sentence)
    deverbal_noun_lemma = " ".join([p["lemma"] for p in nlp.process([deverbal_noun])])
    deverbal_noun_tokens = nlp.process(deverbal_noun)
    deverbal_noun_tokens_idx = 0
    is_noun = 0
    deverbal_noun_present = 0
    for word in translated_tagged_tokens:
        if word["word"] == None or word["lemma"] == None: continue
        if (word["word"].lower() == deverbal_noun_tokens[deverbal_noun_tokens_idx]["word"].lower() or word["lemma"].lower() == deverbal_noun_tokens[deverbal_noun_tokens_idx]["lemma"].lower()):
            deverbal_noun_present += 1
            if word["xpos"].startswith("N"):
                is_noun += 1
            deverbal_noun_tokens_idx += 1
            if deverbal_noun_tokens_idx == len(deverbal_noun_tokens): break
    deverbal_noun_present = (deverbal_noun_present == len(deverbal_noun_tokens))
    is_noun = (is_noun == len(deverbal_noun_tokens))

    if not deverbal_noun_present:
        good_translation = False
        type_of_result = "error_translation"          
    elif not is_noun:
        good_translation = False
        type_of_result = "bad_deverbal_noun"
    elif deverbal_noun not in corresponding_lemmas and deverbal_noun_lemma not in corresponding_lemmas:
        good_translation = False
        type_of_result = "bad_translation"
    else:      
        good_translation = True
        type_of_result = "good_translation"
        
    
    formatted_element["validation_result"] = good_translation
    formatted_element["validation_type"] = type_of_result
    return good_translation, type_of_result

def validate_single_prompt_6_object(formatted_element, nlpProcess: str = "Stanza"):
    
    if nlpProcess == "NLTK":
        nlp = nltkProcess
    else:
        nlp = stanzaProcess

    good_translation = True
    type_of_result = "good_translation"

    target_predicate_indexes = [i for i, predicate in enumerate(formatted_element["words_predicates"]) if predicate != "_"] # to take also composed predicates such as "took place"

    predicate = " ".join(formatted_element["words_predicates"][i] for i in target_predicate_indexes)
    corresponding_syns = formatted_element["synsets_event"][target_predicate_indexes[0]] # all the splitted predicates have the same corresponding_syns, so just look at the first one
    
    corresponding_lemmas = []
    for corresponding_syn in corresponding_syns:
            lemmas = wordnet.synset(corresponding_syn).lemmas()
            for lemma in lemmas:
                corresponding_lemmas.append(lemma.name().replace("_"," ").lower())
    
           
    formatted_sentence = re.sub(r"\n+","\n", formatted_element["chatgpt_output"].strip())
    possible_deverbal_nouns = re.findall(r'\*\*(.*?)\*\*', formatted_sentence)
    
    if len(possible_deverbal_nouns) != 1:
        good_translation = False
        type_of_result = "error_formatting"
        
        formatted_element["validation_result"] = good_translation
        formatted_element["validation_type"] = type_of_result
        return good_translation, type_of_result
    else:
        deverbal_noun = possible_deverbal_nouns[0].strip().lower()
        translated_sentence = re.sub(r'\*\*(.*?)\*\*', r'\1', formatted_sentence).strip()
        formatted_element["translated_sentence"] = translated_sentence
        formatted_element["deverbal_noun"] = deverbal_noun    

    translated_tagged_tokens = nlp.process(translated_sentence)
    deverbal_noun_lemma = " ".join([p["lemma"] for p in nlp.process([deverbal_noun])])
    deverbal_noun_tokens = nlp.process(deverbal_noun)
    deverbal_noun_tokens_idx = 0
    is_noun = 0
    deverbal_noun_present = 0
    for word in translated_tagged_tokens:
        if word["word"] == None or word["lemma"] == None: continue
        if (word["word"].lower() == deverbal_noun_tokens[deverbal_noun_tokens_idx]["word"].lower() or word["lemma"].lower() == deverbal_noun_tokens[deverbal_noun_tokens_idx]["lemma"].lower()):
            deverbal_noun_present += 1
            if word["xpos"].startswith("N"):
                is_noun += 1
            deverbal_noun_tokens_idx += 1
            if deverbal_noun_tokens_idx == len(deverbal_noun_tokens): break
    deverbal_noun_present = (deverbal_noun_present == len(deverbal_noun_tokens))
    is_noun = (is_noun == len(deverbal_noun_tokens))

    if not deverbal_noun_present:
        good_translation = False
        type_of_result = "error_translation"          
    elif not is_noun:
        good_translation = False
        type_of_result = "bad_deverbal_noun"
    elif deverbal_noun not in corresponding_lemmas and deverbal_noun_lemma not in corresponding_lemmas:
        good_translation = False
        type_of_result = "bad_translation"
    else:      
        good_translation = True
        type_of_result = "good_translation"
        
    
    formatted_element["validation_result"] = good_translation
    formatted_element["validation_type"] = type_of_result
    return good_translation, type_of_result


def validate_prompt_results(prompt_results_path, validate_single_prompt_object_fn = validate_single_prompt_3_object, limit_samples = -1, nlpProcess: str = "Stanza"):

    if type(prompt_results_path) == str:
        with open(prompt_results_path, "r") as json_file:
            semcor_prompts = json.load(json_file)
    else:
        semcor_prompts = prompt_results_path
        
    print('len of semcor_prompts:',len(semcor_prompts))
    
    if limit_samples > 0:
        limited_semcor_prompts = {}
        for idx, (elem_id, element) in enumerate(semcor_prompts.items()):
            if idx >= limit_samples: break
            limited_semcor_prompts[elem_id] = element
        semcor_prompts = limited_semcor_prompts
        print('len of semcor_prompts limited to:', len(semcor_prompts))

    good_translations = {}
    error_formatting = {}
    error_translations = {}
    not_noun_translations = {}
    bad_translations = {}

    for elem_id, formatted_element in tqdm(semcor_prompts.items()):

        good_translation, type_of_result  = validate_single_prompt_object_fn(formatted_element=formatted_element, nlpProcess=nlpProcess)
        
        if good_translation:
            good_translations[elem_id] = formatted_element
        elif type_of_result == "error_formatting":
            error_formatting[elem_id] = formatted_element
        elif type_of_result == "error_translation":
            error_translations[elem_id] = formatted_element
        elif type_of_result == "bad_deverbal_noun":
            not_noun_translations[elem_id] = formatted_element
        elif type_of_result == "bad_translation":
            bad_translations[elem_id] = formatted_element

    print(f"Number of good translation: {len(good_translations)}")
    print(f"Number of error in output format (not respecting a parsable format): {len(error_formatting)}")
    print(f"Number of error translation (reported deverbal noun not present in the sentence): {len(error_translations)}")
    print(f"Number of error in deverbal POS (deverbal noun isn't actually a noun): {len(not_noun_translations)}")
    print(f"Number of bad translation (deverbal noun not in the acceptable set): {len(bad_translations)}")

    # return good_translations, error_formatting, error_translations, not_noun_translations, bad_translations
    return {
        "good_translations": good_translations,
        "error_formatting": error_formatting,
        "error_translations": error_translations,
        "not_noun_translations": not_noun_translations,
        "bad_translations": bad_translations,
        "length": len(semcor_prompts)
    }
    
def quick_validate_prompt_results(prompt_results_path, limit_samples = -1):

    if type(prompt_results_path) == str:
        with open(prompt_results_path, "r") as json_file:
            semcor_prompts = json.load(json_file)
    else:
        semcor_prompts = prompt_results_path
        
    print('len of semcor_prompts:',len(semcor_prompts))
    
    if limit_samples > 0:
        limited_semcor_prompts = {}
        for idx, (elem_id, element) in enumerate(semcor_prompts.items()):
            if idx >= limit_samples: break
            limited_semcor_prompts[elem_id] = element
        semcor_prompts = limited_semcor_prompts
        print('len of semcor_prompts limited to:', len(semcor_prompts))

    good_translations = {}
    error_formatting = {}
    error_translations = {}
    not_noun_translations = {}
    bad_translations = {}

    for elem_id, formatted_element in tqdm(semcor_prompts.items()):

        good_translation, type_of_result  = formatted_element["validation_result"], formatted_element["validation_type"]
        
        if good_translation:
            good_translations[elem_id] = formatted_element
        elif type_of_result == "error_formatting":
            error_formatting[elem_id] = formatted_element
        elif type_of_result == "error_translation":
            error_translations[elem_id] = formatted_element
        elif type_of_result == "bad_deverbal_noun":
            not_noun_translations[elem_id] = formatted_element
        elif type_of_result == "bad_translation":
            bad_translations[elem_id] = formatted_element

    print(f"Number of good translation: {len(good_translations)}")
    print(f"Number of error in output format (not respecting a parsable format): {len(error_formatting)}")
    print(f"Number of error translation (reported deverbal noun not present in the sentence): {len(error_translations)}")
    print(f"Number of error in deverbal POS (deverbal noun isn't actually a noun): {len(not_noun_translations)}")
    print(f"Number of bad translation (deverbal noun not in the acceptable set): {len(bad_translations)}")

    # return good_translations, error_formatting, error_translations, not_noun_translations, bad_translations
    return {
        "good_translations": good_translations,
        "error_formatting": error_formatting,
        "error_translations": error_translations,
        "not_noun_translations": not_noun_translations,
        "bad_translations": bad_translations,
        "length": len(semcor_prompts)
    }