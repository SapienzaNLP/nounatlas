import sys, os, json, csv
import random
import argparse
from tqdm import tqdm

DIR = os.path.realpath(os.path.dirname(__file__))
dataset_dir = os.path.join(DIR, '../../outputs_nominal_srl/dataset/')

import stanza
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
    
nlp = StanzaProcess()

def split_dictionary(original_dict, percentages):
    total = sum(percentages)

    if not 0.99 < total < 1.01:
        raise ValueError("Percentages must sum to 1.0")

    split_points = [int(len(original_dict) * p) for p in percentages]
    if sum(split_points) < len(original_dict):
        split_points[0] = len(original_dict) - sum(split_points[1:])
    split_dicts = [{} for _ in range(len(percentages))]

    items = list(original_dict.items())
    current_idx = 0

    for i, count in enumerate(split_points):
        split_dicts[i] = dict(items[current_idx:current_idx + count])
        current_idx += count

    return split_dicts

def format_to_dataset(
        mapped_infos_filepath,  output_dir_name,
        keep_verbal_sample = False, keep_verbal_predicates = True, 
        keep_nominal_sample = True, keep_nominal_predicates = True,
        datset_name = "dataset"):
    
    NO_ROLE_SYMBOL = "_"
    result = {}

    dataset_infos = {
        "keep_verbal_sample": keep_verbal_sample,
        "keep_verbal_predicates": keep_verbal_predicates,
        "keep_nominal_sample": keep_nominal_sample,
        "keep_nominal_predicates": keep_nominal_predicates,
    }
    for info, info_value in dataset_infos.items():
        print(f"{info}: {info_value}")
    dataset_infos["number_of_samples"] = 0
    dataset_infos["number_of_verbal_predicates"] = 0
    dataset_infos["number_of_nominal_predicates"] = 0

    with open(mapped_infos_filepath, "r") as json_file:
        mapped_infos = json.load(json_file)

    mapped_infos_keys = list(mapped_infos.keys())
    random.shuffle(mapped_infos_keys)

    pbar = tqdm(mapped_infos_keys)
    for sample_id in pbar:
        # continue
        for sample_type in ["verbal", "nominal"]:
            if sample_type == "verbal" and not keep_verbal_sample: continue
            if sample_type == "nominal" and not keep_nominal_sample: continue
            sample = mapped_infos[sample_id][sample_type]

            nlp_result = nlp.process(sample["words"])

            sample_converted = {
                "annotations": {}, # predicate_idx: {"predicate": "name", "roles": [...]}
                "lemmas": [w["lemma"] for w in nlp_result],
                "words": sample["words"],
            }

            assert len(sample_converted["lemmas"]) == len(sample_converted["words"])

            for predicate_idx, predicate_infos in sample["predictions"].items():
                predicate_idx = int(predicate_idx)

                if predicate_infos["predicate"] == "_": continue # Removing predicates not in phrase

                if sample_type == "nominal" and not keep_verbal_predicates and predicate_idx != int(mapped_infos[sample_id]["nominal_predicate_target_idx"]):
                    continue

                if sample_type == "nominal" and not keep_nominal_predicates and predicate_idx == int(mapped_infos[sample_id]["nominal_predicate_target_idx"]):
                    continue

                roles = [NO_ROLE_SYMBOL]*len(sample["words"])

                for [r_i, r_f, r_type] in predicate_infos["roles"]:
                    role = r_type.title() # the dataset uses first character upper
                    roles[r_i] = f'B-{role}'
                    for r_j in range(r_i+1, r_f):
                        roles[r_j] = f'I-{role}'

                # Adding the predicate POS:
                pos_pred_type = "V"
                # if predicate_idx == int(mapped_infos[sample_id]["nominal_predicate_target_idx"]): pos_pred_type = "N" # Add this line if we want B-N
                roles[predicate_idx] = f"B-{pos_pred_type}"

                sample_converted["annotations"][str(predicate_idx)] = {
                    "predicate": predicate_infos["predicate"],
                    "roles": roles
                }

                if sample_type == "nominal" and predicate_idx == int(mapped_infos[sample_id]["nominal_predicate_target_idx"]):
                    dataset_infos["number_of_nominal_predicates"] += 1
                else:
                    dataset_infos["number_of_verbal_predicates"] += 1
        
            result[sample_id] = sample_converted
            dataset_infos["number_of_samples"] += 1

            pbar.set_description( '|'.join( [f'{k}: {dataset_infos[k]}' for k in ["number_of_samples", "number_of_verbal_predicates", "number_of_nominal_predicates"]] ))

    # file_name = os.path.splitext(os.path.basename(mapped_infos_filepath))[0] + f'@{keep_verbal_sample}_{keep_verbal_predicates}_{keep_nominal_sample}_{keep_nominal_predicates}'
    dataset_save_dir = os.path.join(dataset_dir, output_dir_name)
    os.makedirs(dataset_save_dir, exist_ok=True)

    with open(os.path.join(dataset_save_dir, datset_name + ".json"), "w") as f:
        json.dump(result,f, indent=4)

    with open(os.path.join(dataset_save_dir, f"{datset_name}_infos" + ".json"), "w") as f:
        json.dump(dataset_infos,f, indent=4)
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass a mapped infos file path to convert it.")
    parser.add_argument('-p', '--path',  type=str, help='The path')
    parser.add_argument('-o', '--output_dir_name',  type=str, help='The output directory name')
    parser.add_argument('-kv', '--keep_verbal_sample', default=False, help='Keep verbal samples (default: False)')
    parser.add_argument('-kvp', '--keep_verbal_predicates', default=True, help='Keep verbal predicates in the nominal sample (default: True)')
    parser.add_argument('-kn', '--keep_nominal_sample', default=True, help='Keep nominal samples (default: True)')
    parser.add_argument('-knp', '--keep_nominal_predicates', default=True, help='Keep nominal predicates in the nominal sample (default: True)')
    args = parser.parse_args()

    # combinations = [
    #     # [False, True, True, True], # verbal + nominal predicates (without verbal phrases, using only the nominal phrases)
    #     # [False, True, True, False], # verbal predicates (without verbal phrases, using only the nominal phrases)
    #     [False, False, True, True], # nominal predicates (without verbal phrases, using only the nominal phrases)
    #     # [True, True, True, True], # verbal + nominal predicates (with nominal AND verbal phrases)
    # ]
    # args.path = os.path.join(
    #     DIR, 
    #     '../../outputs_nominal_srl/mapped_infos/', 
    #     "semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True_NN.json"
    # )
    format_to_dataset(
        args.path, args.output_dir_name,
        keep_verbal_sample = args.keep_verbal_sample, keep_verbal_predicates = args.keep_verbal_predicates,
        keep_nominal_sample = args.keep_nominal_sample, keep_nominal_predicates = args.keep_nominal_predicates)

    print("All done.")