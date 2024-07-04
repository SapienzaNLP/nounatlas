import os, sys, json
from data_mapping.format_to_dataset import format_to_dataset, split_dictionary
from collections import Counter
from collections import OrderedDict
import random
# Params

# THE SMALLEST DATASET TO THE LARGEST, put them in that order!
mapped_infos_datasets_names = [
    "semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True.json",
    "semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True_NN.json"
]
selected_combination_type = 2
formatted_datasets_dir_names = [
    f"semcorgemini_{selected_combination_type}_rule",
    f"semcorgemini_{selected_combination_type}_nn"
]
manual_dataset_whitelist = [
    "rule.txt",
    "nn.txt"
]
split_percentages = [0.8, 0.1, 0.1]
SEED = 1746911

# Not params

split_set_target = 2

DIR = os.path.realpath(os.path.dirname(__file__))
mapped_infos_dir = os.path.join(DIR, '../outputs_nominal_srl/mapped_infos/')
dataset_dir = os.path.join(DIR, '../outputs_nominal_srl/dataset/')
manual_id_selection_dir = os.path.join(DIR, '../outputs_nominal_srl/manual_id_selection/')
combinations = [
    [False, True, True, True], # verbal + nominal predicates (without verbal phrases, using only the nominal phrases)
    [False, True, True, False], # verbal predicates (without verbal phrases, using only the nominal phrases)
    [False, False, True, True], # nominal predicates (without verbal phrases, using only the nominal phrases)
    [True, True, True, True], # verbal + nominal predicates (with nominal AND verbal phrases)
]
combination = combinations[selected_combination_type]
mapped_infos_datasets_filepaths = [os.path.join(mapped_infos_dir, p) for p in mapped_infos_datasets_names]
manual_datasets_whitelists_filepaths = [os.path.join(manual_id_selection_dir, p) if p is not None else None for p in manual_dataset_whitelist]
formatted_datasets_dir_paths = [os.path.join(dataset_dir, p) for p in formatted_datasets_dir_names]

# Code pipeline

# Generating/Opening dataset.json

results = []

for idx, mapped_infos_filepath in enumerate(mapped_infos_datasets_filepaths):
    if os.path.exists(os.path.join(formatted_datasets_dir_paths[idx], "dataset.json")):
        print(f"Skipping {formatted_datasets_dir_names[idx]} because dataset.json already exists...")
        with open(os.path.join(formatted_datasets_dir_paths[idx], "dataset.json"), "r") as json_file:
            result = json.load(json_file)
    else:
        result = format_to_dataset(mapped_infos_filepath = mapped_infos_filepath, output_dir_name=formatted_datasets_dir_names[idx], 
                    keep_verbal_sample = combination[0], keep_verbal_predicates = combination[1],
                    keep_nominal_sample = combination[2], keep_nominal_predicates = combination[3])
        
    results.append(result)

# Loading ids of the manual annotations

manual_datasets_whitelists = []
for idx, fp in enumerate(manual_datasets_whitelists_filepaths):
    print(f"###### Processing manual annotated for {formatted_datasets_dir_names[idx]} ######")
    if fp is None:
        print("Skipping... (No whitelist provided)")
    with open(fp, 'r') as file:
        manual_ids_in_dataset = [line.strip() for line in file.readlines()]

        not_found = 0
        for id in manual_ids_in_dataset:
            if id not in results[idx].keys():
                print(f"WARN: {id} not found ({formatted_datasets_dir_names[idx]})")
                not_found += 1
            else:
                manual_datasets_whitelists.append(id)

        print("Samples len:", len(manual_ids_in_dataset))
        print("Not found:", not_found)

        # Statistics:
        counter_frames = Counter([pred_infos["predicate"] for s_id, sample in results[idx].items() for p_id, pred_infos in sample["annotations"].items() if s_id in manual_datasets_whitelists])
        print("Number of different frames:", len(counter_frames))
        print("Frames count:", counter_frames.most_common(), "The frames are:")
        frames_used = list(counter_frames.keys())
        print(frames_used)

manual_datasets_whitelists = list(OrderedDict.fromkeys(manual_datasets_whitelists))
print(manual_datasets_whitelists)

# Splitting accordingly

resulting_split_set_ids = manual_datasets_whitelists
result_split_sets = []
for idx, result in enumerate(results):

    random.seed(SEED)
    result_keys = list(result.keys())
    random.shuffle(result_keys)

    split_percentage = split_percentages[split_set_target]
    dataset_split_len = len(result)*split_percentage
    if dataset_split_len < len(manual_datasets_whitelists):
        print("WARN: the dataset split set is smaller than the number of ids in the whitelists!")

    print(f"||||| Generating split for {formatted_datasets_dir_names[idx]} |||||")
    print("DEBUG: len of the total:",len(result))
    dataset_splits = split_dictionary(result, split_percentages)
    
    for dataset_split_idx in range(len(dataset_splits)):
        if dataset_split_idx == split_set_target: continue
        for target_sample_id in resulting_split_set_ids:
            if target_sample_id in dataset_splits[dataset_split_idx]:
                # Move the ID to the target dictionary
                dataset_splits[split_set_target][target_sample_id] = dataset_splits[dataset_split_idx].pop(target_sample_id)
                # Select a random sample from the target dictionary
                selected_giveaway_sample_id = None
                for possible_giveaway_sample_id in dataset_splits[split_set_target].keys():
                    if possible_giveaway_sample_id not in resulting_split_set_ids:
                        selected_giveaway_sample_id = possible_giveaway_sample_id
                        break
                if selected_giveaway_sample_id == None:
                    print("FATAL ERROR: Unexpected error, the split target is too small?")
                    sys.exit()
                # Move the random sample to the dictionary where the ID was taken from
                dataset_splits[dataset_split_idx][selected_giveaway_sample_id] = dataset_splits[split_set_target].pop(selected_giveaway_sample_id)
    
    # This is to ensure that the datasets share similar target sets
    resulting_split_set_ids = list(OrderedDict.fromkeys(resulting_split_set_ids) | OrderedDict.fromkeys(dataset_splits[split_set_target].keys()))
    print("DEBUG: len of the total:",len(dataset_splits[0])+len(dataset_splits[1])+len(dataset_splits[2]))
    # Save the splits
    for split_name, split_data in zip(["train", "dev", "test"], dataset_splits):
        with open(os.path.join(formatted_datasets_dir_paths[idx], split_name + ".json"), "w") as f:
            json.dump(split_data,f, indent=4)

    result_split_sets.append(dataset_splits[split_set_target])

# Get the keys of the first dictionary
intersection_keys = set(result_split_sets[0].keys())

# Find the intersection of keys among all dictionaries
for dictionary in result_split_sets[1:]:
    intersection_keys = intersection_keys.intersection(dictionary.keys())

print("Number of equal keys:", len(intersection_keys))

# Check if the values associated with the intersection keys are equal across all dictionaries
intersection_values = {key: result_split_sets[0][key] for key in intersection_keys}
blacklist = []
for dictionary in result_split_sets[1:]:
    for key in intersection_keys:
        if dictionary[key] != intersection_values[key]:
            blacklist.append(key)
            
intersection_values = {k:v for k,v in intersection_values.items() if k not in blacklist}

print("Number of equal keys with equal samples:", len(intersection_values))

print("All done.")


    
        
    

