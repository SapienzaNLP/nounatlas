from nltk.corpus import wordnet
import random, copy

def add_examples_to_prompt():
    examples = [
        ("Marco is eating fish very loudly", "Marco's very loud eating of fish"),
    ]

    if len(examples) == 1:
        phrase = f' For example, a possible nominalization of the phrase "{examples[0][0]}" is "{examples[0][1]}".'

    elif len(examples) > 1:
        phrase = ' Here are some possible examples: '
        examples_formatted = [f'the phrase "{e[0]}" can be nominalized as "{e[1]}"' for e in examples]
        phrase = phrase + "; ".join(examples_formatted) + "."
    else:
        phrase = ''

    return phrase

def prompt_format_1(formatted_element):
    """ output example: 
        Change the sentence by nominalizing the verb "X": "She X to travel to Greece for business.".
    """
    sentence = formatted_element["sentence"]
    resulting_predicate = " ".join([p for p in formatted_element["words_predicates"] if p != "_"])
    prelude = "Change the sentence by nominalizing the verb"
    
    return f'{prelude} "{resulting_predicate}": "{sentence}"'
    
def prompt_format_2(formatted_element):
    """ output example: 
        Nominalize this sentence and do not add new words or remove them, giving as output only the nominalized sentence: "She X to travel to Greece for business.".
    """
    sentence = formatted_element["sentence"]
    prelude = "Nominalize this sentence and do not add new words or remove them, giving as output only the nominalized sentence"
    input_phrase = f'{prelude}: "{sentence}"'
    return input_phrase

# Following prompt format 3
def prompt_format_3(formatted_element):
    """ output example: 
        Change the sentence by nominalizing the verb "produced". Afterward leave an empty line and indicate the chosen deverbal noun: The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced "no evidence" that any irregularities took place.
    """
    sentence = formatted_element["sentence"]
    resulting_predicate = " ".join([p for p in formatted_element["words_predicates"] if p != "_"])
    prelude = "Change the sentence by nominalizing the verb"

    prompt = f'{prelude} "{resulting_predicate}". Afterward leave an empty line and indicate the chosen deverbal noun: "{sentence}"'
    # prompt = f'{prelude} "{resulting_predicate}". Afterward indicate the chosen deverbal noun: "{sentence}"'
    
    return prompt

# Following prompt format 1
def prompt_format_4(formatted_element):
    """ output example: 
        Change the sentence by nominalizing the verb "produced" into its deverbal noun "production": The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced "no evidence" that any irregularities took place.
    """
    sentence = formatted_element["sentence"]
    resulting_predicate = " ".join([p for p in formatted_element["words_predicates"] if p != "_"])
    resulting_corresponding_syns = [corresponding_syns for predicate, corresponding_syns  in zip(formatted_element["words_predicates"],formatted_element["synsets_event"]) if predicate != "_"][0]

    corresponding_lemmas = []
    for corresponding_syn in resulting_corresponding_syns:
        lemmas = wordnet.synset(corresponding_syn).lemmas()
        corresponding_lemmas.append(lemmas[0].name().replace("_", " "))
        
    prelude = "Change the sentence by nominalizing the verb"
    prompt = f'{prelude} "{resulting_predicate}" into its deverbal noun "{corresponding_lemmas[0]}": "{sentence}"'

    return prompt

# Following prompt format 3
def prompt_format_5(formatted_element):
    """ output example: 
        Change the sentence by nominalizing the verb "produced". Use exactly one of these deverbal nouns: "production". Afterward leave an empty line and indicate the chosen deverbal noun: The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced "no evidence" that any irregularities took place.
    """
    sentence = formatted_element["sentence"]
    resulting_predicate = " ".join([p for p in formatted_element["words_predicates"] if p != "_"])
    resulting_corresponding_syns = [corresponding_syns for predicate, corresponding_syns  in zip(formatted_element["words_predicates"],formatted_element["synsets_event"]) if predicate != "_"][0]

    corresponding_lemmas = set()
    for corresponding_syn in resulting_corresponding_syns:
        lemmas = wordnet.synset(corresponding_syn).lemmas()
        lemmas = [f'"{lemma.name().replace("_", " ")}"' for lemma in lemmas]
        corresponding_lemmas.update(lemmas)
        
    deverbal_lemmas = ", ".join(list(corresponding_lemmas))
    prompt = f'Change the sentence by nominalizing the verb "{resulting_predicate}". Use exactly one of these deverbal nouns: {deverbal_lemmas}. Afterward leave an empty line and indicate the chosen deverbal noun: "{sentence}"'

    return prompt

# Following prompt format 6
def prompt_format_6(formatted_element):
    """ output example: 
        Change the sentence by nominalizing the verb "produced". Use exactly one of these deverbal nouns: "production". Indicate the chosen deverbal noun with **: "The Fulton County Grand Jury said Friday an investigation of Atlanta 's recent primary election **produced** " no evidence " that any irregularities took place ."
    """
    
    resulting_predicate = " ".join([p for p in formatted_element["words_predicates"] if p != "_"])
    predicate_indexes = [i for i,p in enumerate(formatted_element["words_predicates"]) if p != "_"]
    words = copy.deepcopy(formatted_element["words"])
    words[predicate_indexes[0]] = "**" + words[predicate_indexes[0]]
    words[predicate_indexes[-1]] = words[predicate_indexes[-1]] + "**"
    sentence = " ".join(words)
    resulting_corresponding_syns = [corresponding_syns for predicate, corresponding_syns  in zip(formatted_element["words_predicates"],formatted_element["synsets_event"]) if predicate != "_"][0]

    corresponding_lemmas = set()
    for corresponding_syn in resulting_corresponding_syns:
        lemmas = wordnet.synset(corresponding_syn).lemmas()
        lemmas = [f'"{lemma.name().replace("_", " ")}"' for lemma in lemmas]
        corresponding_lemmas.update(lemmas)
        
    deverbal_lemmas = ", ".join(list(corresponding_lemmas))
    prompt = f'Change the sentence by nominalizing the verb "{resulting_predicate}" indicated by **. Use exactly one of these deverbal nouns: {deverbal_lemmas}. Indicate the chosen deverbal noun with **: "{sentence}"'

    return prompt

# Following prompt format 3
def prompt_format_7(formatted_element):
    """ output example: 
        Change the sentence by nominalizing the verb "produced". Use exactly one of these deverbal nouns: "production". Afterward leave an empty line and indicate the chosen deverbal noun: The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced "no evidence" that any irregularities took place.
    """
    sentence = formatted_element["sentence"]
    predicate_indexes = [i for i,p in enumerate(formatted_element["words_predicates"]) if p != "_"]
    resulting_predicate = " ".join([p for p in formatted_element["words_predicates"] if p != "_"])
    resulting_corresponding_syns = [corresponding_syns for predicate, corresponding_syns  in zip(formatted_element["words_predicates"],formatted_element["synsets_event"]) if predicate != "_"][0]

    predicate_lemma = " ".join([formatted_element["lemmas"][i] for i in predicate_indexes])
    predicate_synset = wordnet.synset(formatted_element["words_synsets"][predicate_indexes[0]])
    corresponding_lemmas = []
    for lemma in predicate_synset.lemmas(): # Finding the specific lemma of this predicate
        if lemma.name().replace("_", " ") == predicate_lemma:
            for drf in lemma.derivationally_related_forms():
                if drf.synset().name() in resulting_corresponding_syns: # Check if the synset of this drf is of the allowed synsets (so the ones in the same frame)
                    corresponding_lemmas.append(f'"{drf.name().replace("_", " ")}"')
    
    # Fallback in case there are no direct DRFs
    if len(corresponding_lemmas) == 0:
        return prompt_format_5(formatted_element)
        
    deverbal_lemmas = ", ".join(list(corresponding_lemmas))
    prompt = f'Change the sentence by nominalizing the verb "{resulting_predicate}". Use exactly one of these deverbal nouns: {deverbal_lemmas}. Afterward leave an empty line and indicate the chosen deverbal noun: "{sentence}"'

    return prompt

# Following prompt format 6
def prompt_format_8(formatted_element):
    """ output example: 
        Change the sentence by nominalizing the verb "produced". Use exactly one of these deverbal nouns: "production". Indicate the chosen deverbal noun with **: "The Fulton County Grand Jury said Friday an investigation of Atlanta 's recent primary election **produced** " no evidence " that any irregularities took place ."
    """
    
    predicate_indexes = [i for i,p in enumerate(formatted_element["words_predicates"]) if p != "_"]
    resulting_predicate = " ".join([p for p in formatted_element["words_predicates"] if p != "_"])    
    
    words = copy.deepcopy(formatted_element["words"])
    words[predicate_indexes[0]] = "**" + words[predicate_indexes[0]]
    words[predicate_indexes[-1]] = words[predicate_indexes[-1]] + "**"
    sentence = " ".join(words)
    resulting_corresponding_syns = [corresponding_syns for predicate, corresponding_syns in zip(formatted_element["words_predicates"],formatted_element["synsets_event"]) if predicate != "_"][0]

    predicate_lemma = " ".join([formatted_element["lemmas"][i] for i in predicate_indexes])
    predicate_synset = wordnet.synset(formatted_element["words_synsets"][predicate_indexes[0]])
    corresponding_lemmas = []
    for lemma in predicate_synset.lemmas(): # Finding the specific lemma of this predicate
        if lemma.name().replace("_", " ") == predicate_lemma:
            for drf in lemma.derivationally_related_forms():
                if drf.synset().name() in resulting_corresponding_syns: # Check if the synset of this drf is of the allowed synsets (so the ones in the same frame)
                    corresponding_lemmas.append(f'"{drf.name().replace("_", " ")}"')
    
    # Fallback in case there are no direct DRFs
    if len(corresponding_lemmas) == 0:
        return prompt_format_6(formatted_element)
        
    deverbal_lemmas = ", ".join(list(corresponding_lemmas))
    prompt = f'Change the sentence by nominalizing the verb "{resulting_predicate}" indicated by **. Use exactly one of these deverbal nouns: {deverbal_lemmas}. Indicate the chosen deverbal noun with **: "{sentence}"'

    return prompt


