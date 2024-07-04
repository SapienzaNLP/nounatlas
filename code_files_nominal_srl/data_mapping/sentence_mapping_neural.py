import sys, os, json, csv, requests

import matplotlib.pyplot as plt
import spacy
from spacy import displacy

from tqdm import tqdm
import argparse

from copy import deepcopy

from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from sentence_mapping_common import *

# region functions

import multiprocessing

def parallelize_mapping(validated_samples_data_path, num_of_threads = 1):

    with open(validated_samples_data_path, "r") as json_file:
        validated_samples = json.load(json_file)

    if num_of_threads == 1:
        return map_nominal_sentences(validated_samples)
    
    queue = multiprocessing.Queue()

    mapped_infos_list = [{}]*num_of_threads
    debug_infos_list = [{}]*num_of_threads
    splitted_validated_samples = split_dict(validated_samples, num_of_threads)
    processes = []

    for i in range(len(splitted_validated_samples)):
        processes.append( multiprocessing.Process(target=map_nominal_sentences, args=(splitted_validated_samples[i], queue, i)) )
        processes[i].start()

    for i in range(len(splitted_validated_samples)):
        processes[i].join()

    for i in range(len(splitted_validated_samples)):
        [mapped_infos_list_i, debug_infos_list_i] = queue.get()
        mapped_infos_list[i] = mapped_infos_list_i
        debug_infos_list[i] = debug_infos_list_i

    merged_mapped_infos = {k: v for d in mapped_infos_list for k, v in d.items()}
    merged_debug_infos = merge_debug_infos(debug_infos_list)

    return merged_mapped_infos, merged_debug_infos

# endregion

# region hyperparameters

DIR = os.path.realpath(os.path.dirname(__file__))

va_frame_pas_path = os.path.join(DIR, '../../resources/VerbAtlas/VA_frame_pas.tsv')
va_frame_pas_path = os.path.join(DIR, '../../resources/VerbAtlas/VA_frame_pas.tsv')

mapped_infos_dir = os.path.join(DIR, '../../outputs_nominal_srl/mapped_infos/')
html_outputs_dir = os.path.join(DIR, '../../outputs_nominal_srl/html_outputs/')

# nominal_classification_dataset_json = os.path.join(DIR, '../../datasets/dataset_nominal_classification/dataset.json')

# endregion

def map_nominal_sentences(validated_samples, queue = None, process_idx = None):

    # # Opening the file for mapping (verbal) synsets to frames
    # with open(nominal_classification_dataset_json, "r") as json_file:
    #     dataset_json = json.load(json_file)
    #     verbal_syns = {s:f_name for f_name,f_infos in dataset_json["verbal_definitions"].items() for s in f_infos.keys()}

    sentence_alignment_model = SentenceSimilarityModel()

    if process_idx is not None: print("PROCESS IDX", process_idx)

    mapped_infos = {}

    debug_infos = {
        # Validation pipeline:
        "total_samples": 0,
        "valid_samples": 0,
        "cost": 0,
        # the errors are acquired at runtime...
        # Mapping pipeline:
        "good_samples": 0,
        "invero_verbal_errors": 0,
        "invero_nominal_errors": 0,
        "predicate_not_found": 0, # predicate not found from the VerbAtlas splitting result of the verbal phrase
        "standard_role_error": 0, # one or more of the verbatlas roles was not found in the nominal phrase
        "special_role_error": 0, # one or more of the "special" roles was not found in the nominal phrase
        "no_roles_error": 0,
        # "predicate_different_error": 0, # frame different from the VerbAtlas prediction
    }

    va_roles = get_possible_role_names(va_frame_pas_path)

    progress_bar = tqdm( enumerate(validated_samples.items()), total=len(validated_samples) )
    progress_bar.update_description = lambda: progress_bar.set_description(f'total: {debug_infos["total_samples"]} valid: {debug_infos["valid_samples"]}, good: {debug_infos["good_samples"]}, preds404: {debug_infos["predicate_not_found"]}, st_roles404: {debug_infos["standard_role_error"]}, sp_roles404: {debug_infos["special_role_error"]}, invero_v_err: {debug_infos["invero_verbal_errors"]}, invero_n_err: {debug_infos["invero_nominal_errors"]}')
    progress_bar.update_description()

    for idx, (sample_id, sample) in progress_bar:

        # Checking if it's a good sample:

        debug_infos["total_samples"] += 1
        debug_infos["cost"] += sample["cost"]
        if sample["validation_result"]:
            debug_infos["valid_samples"] += 1
        else:
            debug_infos.setdefault(sample["validation_type"], 0)
            debug_infos[sample["validation_type"]] += 1
            continue
        
        sentence_nominal = sample["translated_sentence"]
        deverbal_noun = sample["deverbal_noun"]

        # Computing the roles infos from VerbAtlas:

        res_verbal = invero_request(' '.join(sample["words"]))
        if res_verbal is None: 
            debug_infos["invero_verbal_errors"] += 1
            continue
        res_verbal = parse_invero_result(res_verbal)[0]

        # Finding the predicate in the VerbAtlas result:

        predicate_semcor_idx = [i for i,p in enumerate(sample["words_predicates"]) if p != "_"][0] # in this case, "took place" is just "took" (VerbAtlas takes only one)
        predicate_semcor = sample["words_predicates"][predicate_semcor_idx] # The predicate word that we have from SemCor

        possible_candidates_va_idx = sorted([
            candidate_va_idx 
            for candidate_va_idx in res_verbal["predictions"].keys() 
            if res_verbal["words"][candidate_va_idx] == predicate_semcor
        ], key=lambda x: abs(x - predicate_semcor_idx)) # sorting the candidates by their distance w.r.t the semocor index

        if len(possible_candidates_va_idx) == 0:
            # print(f'[WARN] {sample_id}: Predicate not found!')
            debug_infos["predicate_not_found"] += 1
            continue

        predicate_va_idx = possible_candidates_va_idx[0] # The index of the predicate that it's in the VerbAtlas prediction (might be different from the split of SemCor!)
        predicate_frame_va = res_verbal["predictions"][predicate_va_idx]["predicate"]

        # Splitting the words using the same splitter of VerbAtlas:

        res_nominal = invero_request(sentence_nominal)
        if res_nominal is None: 
            debug_infos["invero_nominal_errors"] += 1
            continue
        res_nominal = parse_invero_result(res_nominal)[0]

        # Finding the deverbal noun index in thes output sentence from res_nominal:
        try:
            possible_predicate_nominal_idxs = []
            for w_i, w in enumerate(res_nominal["words"]):
                if w == deverbal_noun: # if the deverbal noun word is found in the phrase
                    possible_predicate_nominal_idxs.append(w_i)
            predicate_nominal_idx = min(possible_predicate_nominal_idxs, key=lambda x: abs(x - predicate_semcor_idx)) # find nearest index w.r.t predicate_semcor_idx
        except:
            debug_infos["invero_nominal_errors"] += 1
            continue
        
        # Mapping the roles:

        predicate_roles_va = res_verbal["predictions"][predicate_va_idx]["roles"]
        res_nominal_predictions = {} # if we want to discard verbal predictions from the VerbAtlas model
        res_nominal_predictions[predicate_nominal_idx] = {"predicate": predicate_frame_va, "roles": []}

        sample_standard_roles_not_found = 0 # number of standard roles not found in the sample
        sample_special_roles_not_found = 0 # number of special roles not found in the sample

        # Neural network for phrases:
        verbal_token_embeddings, verbal_encoded_input = sentence_alignment_model.encode_no_pooling(res_verbal["words"])
        nominal_token_embeddings, nominal_encoded_input = sentence_alignment_model.encode_no_pooling(res_nominal["words"])

        for role_infos in predicate_roles_va:
            role_span, role_name = (role_infos[0], role_infos[1]), role_infos[2]

            # Neural network for role:
            verbal_token_embeddings_cutted, verbal_encoded_input_cutted = sentence_alignment_model.cut_token_embeddings(
                verbal_token_embeddings, verbal_encoded_input, span = role_span
            )
            verbal_role_encode = sentence_alignment_model.pooling(verbal_token_embeddings_cutted, verbal_encoded_input_cutted)
            # print("Verbal span:", res_verbal["words"][role_span[0]:role_span[1]], "Verbal decoded:",sentence_alignment_model.decode(verbal_encoded_input_cutted["input_ids"][0]))

            role_inventory = [e.lower() for e in res_verbal["words"][ role_span[0]:role_span[1] ]]

            best_window_score = 0
            best_window_position = 0
            best_window_correspondence = None
            current_window_size = len(role_inventory)

            for current_window_position in range( 0, len(res_nominal["words"]) - len(role_inventory) + 1 ):

                current_nominal_window = res_nominal["words"][current_window_position:current_window_position+current_window_size]

                # Neural Network score:
                nominal_token_embeddings_cutted, nominal_encoded_input_cutted = sentence_alignment_model.cut_token_embeddings(
                    nominal_token_embeddings, nominal_encoded_input, span = [current_window_position, current_window_position+current_window_size]
                )
                nominal_window_encode = sentence_alignment_model.pooling(nominal_token_embeddings_cutted, nominal_encoded_input_cutted)

                model_score = sentence_alignment_model.compute_cosine_similarity(verbal_role_encode, nominal_window_encode)

                # The final heuristic score adopted
                current_window_score = model_score # + current_window_words_in_role/len(current_nominal_window)

                if best_window_score < current_window_score:
                    best_window_score = current_window_score
                    best_window_position = current_window_position
                    best_window_correspondence = True

            if best_window_correspondence is not None:
                
                # Check that the predicate index is NOT inside the window! If so, strip the role:
                if predicate_nominal_idx < best_window_position+current_window_size and predicate_nominal_idx >= best_window_position:
                    min_distance_from_predicate = min([best_window_position, best_window_position+current_window_size], key=lambda x: abs(x - predicate_nominal_idx))
                    if min_distance_from_predicate == best_window_position:
                        current_window_size -= predicate_nominal_idx+1 - best_window_position
                        best_window_position = predicate_nominal_idx+1
                    elif min_distance_from_predicate == best_window_position+current_window_size:
                        current_window_size = predicate_nominal_idx - best_window_position
                # Save it
                res_nominal_predictions[predicate_nominal_idx]["roles"].append( [best_window_position, best_window_position+current_window_size, role_name] )
            else:

                print("Error on sentence with role:")
                print(res_nominal["words"])
                print(role_inventory)

                if role_name not in va_roles: 
                    sample_special_roles_not_found += 1 # This role isn't one of the roles listed in the VerbAtlas-roles-list file
                else:
                    sample_standard_roles_not_found += 1 # A matching role wasn't found in the nominal sentence (The role is listed in the VerbAtlas-roles-list file)
                # print(f'[WARN] {sample_id}: No correspondence for a role! (or part of a role) {"BUT NOT A STANDARD ROLE!" if role_name not in va_roles else ""}')
                
        # Stripping overlapping roles:

        res_nominal_predictions[predicate_nominal_idx]["roles"] = sorted(res_nominal_predictions[predicate_nominal_idx]["roles"], key=lambda x: x[0])
        current_ordered_roles = res_nominal_predictions[predicate_nominal_idx]["roles"]

        # Checking if at least one role exists:
        if len(current_ordered_roles) == 0:
            debug_infos["no_roles_error"] += 1
            continue

        for i in range(len(current_ordered_roles) - 1):
            if current_ordered_roles[i][1] > current_ordered_roles[i+1][0]:
                if (current_ordered_roles[i][1] - current_ordered_roles[i][0]) > (current_ordered_roles[i+1][1] - current_ordered_roles[i+1][0]):
                    current_ordered_roles[i][1] = current_ordered_roles[i+1][0]
                else:
                    current_ordered_roles[i+1][0] = current_ordered_roles[i][1]

        # Removing roles that have the same words:
        current_filtered_ordered_roles = [current_ordered_roles[0]]
        for i in range(1, len(current_ordered_roles)):
            if not (
                (
                    (current_ordered_roles[i][0] == current_ordered_roles[i-1][0]) and 
                    (current_ordered_roles[i][1] == current_ordered_roles[i-1][1])
                ) or 
                current_ordered_roles[i][0] == current_ordered_roles[i][1]
            ):
                current_filtered_ordered_roles.append(current_ordered_roles[i])

        res_nominal_predictions[predicate_nominal_idx]["roles"] = current_filtered_ordered_roles

        # Saving infos:

        if sample_standard_roles_not_found > 0: debug_infos["standard_role_error"] += 1
        if sample_special_roles_not_found > 0: debug_infos["special_role_error"] += 1
        
        # res_nominal["predictions"] = res_nominal_predictions # if we want to keep only nominal predictions
        for key, value in res_nominal_predictions.items():
            res_nominal["predictions"][key] = value # if we want to keep nominal and verbal predictions

        mapped_infos[sample_id] = {
            "verbal":res_verbal, "nominal":res_nominal, 
            "verbal_predicate_target_idx": predicate_va_idx,
            "nominal_predicate_target_idx": predicate_nominal_idx,
            "standard_roles_not_found": sample_standard_roles_not_found,
            "special_roles_not_found": sample_special_roles_not_found,
        }

        if sample_special_roles_not_found == 0 and sample_standard_roles_not_found == 0:
            debug_infos["good_samples"] += 1

        progress_bar.update_description()

    print(debug_infos)

    # Finally, we return the processed data:

    if queue is not None:
        queue.put([mapped_infos, debug_infos])
    
    return mapped_infos, debug_infos
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass a list of validated samples paths to map it.")
    parser.add_argument('-p', '--paths', nargs='+', type=str, help='A list of paths (separated by space)')
    args = parser.parse_args()
    
    args.paths = [os.path.join(DIR, '../../datasets/dataset_nominal_srl/semcor', f) for f in [
        "semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True.json"
    ]]

    for validated_samples_data_path in args.paths:

        mapped_infos, debug_infos = parallelize_mapping(validated_samples_data_path, num_of_threads=1)

        file_name = os.path.splitext(os.path.basename(validated_samples_data_path))[0]

        with open(os.path.join(mapped_infos_dir, file_name + "_NN.json"), "w") as f:
            json.dump(mapped_infos,f, indent=4)

        with open(os.path.join(html_outputs_dir, file_name + "_NN.html"), "w") as f:
            f.write(
                spacy_visualize_sentences(mapped_infos, debug_infos = debug_infos, title = file_name)
            )

    print("All done.")