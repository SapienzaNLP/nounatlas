import os, sys, json, csv, random
from typing import Union
import torch
import lightning.pytorch as pl
import copy
from models.PretrainedBiencoder import PretrainedBiencoder
from tqdm import tqdm

from datasets.DynamicParametrizedDataModule import DynamicDataModule, DynamicDataset

def dicts_equal_except_keys(dict1, dict2, keys_to_ignore):
    # Create a copy of the dictionaries without the keys to ignore
    filtered_dict1 = {key: value for key, value in dict1.items() if key not in keys_to_ignore}
    filtered_dict2 = {key: value for key, value in dict2.items() if key not in keys_to_ignore}
    # Check if the filtered dictionaries are equal
    return filtered_dict1 == filtered_dict2

def generate_dataset_type(
        dataset_hyps: dict,
        dataset_dir_path: str,
        division_type: Union["default", "comparison"],
        train_types: Union[list[str], None] = ["verbals.json", "non_ambiguous.json", "disambiguated.json", "manual_classified.json"],
        dev_types: Union[list[str], None] = None,
        test_types: Union[list[str], None] = None
    ):
    
    if division_type == "default" and dev_types == None and test_types == None:
        if train_types == None:
            dataset_dir_path_info = dataset_dir_path
        else:
            dataset_dir_path_info = [os.path.join(dataset_dir_path, part_type) for part_type in train_types]
        dataset_hyps_default = copy.deepcopy(dataset_hyps)
        dataset_hyps_default["dataset_dir_path"] = dataset_dir_path_info
        data_module = DynamicDataModule(**dataset_hyps_default)
        return data_module
    
    elif division_type == "comparison":
        if dev_types == None: dev_types = train_types
        if test_types == None: test_types = dev_types
        generated_types = []
        for split_types in [train_types, dev_types, test_types]:
            dataset_hyps_split = copy.deepcopy(dataset_hyps)
            dataset_hyps_split["dataset_dir_path"] = [
                os.path.join(dataset_dir_path, part_type) for part_type in split_types
            ]
            generated_types.append( DynamicDataModule(**dataset_hyps_split) )

        generated_types[0].delete_dataset("dev").delete_dataset("test")
        generated_types[1].delete_dataset("train").delete_dataset("test")
        generated_types[2].delete_dataset("train").delete_dataset("dev")

        data_module = generated_types[0] + generated_types[1] + generated_types[2]
        return data_module
    
    # Implementing the various tests for the paper:
    else:
        data_module = None

        if division_type == "nonamb_test" or division_type == "manual_test" or division_type == "manual_in_train_test": # Test1: train, then dev and test scores (binary)

            dataset_hyperparameters = copy.deepcopy(dataset_hyps)
            dataset_hyperparameters["train_percentage"] = 0.8
            dataset_hyperparameters["dev_percentage"] = 0.1
            dataset_hyperparameters["test_percentage"] = 0.1
            dataset_hyperparameters["dataset_dir_path"] = [
                os.path.join(dataset_dir_path, part_type) for part_type in ["verbals.json", "non_ambiguous.json", "disambiguated.json"]
            ]
            data_module = DynamicDataModule(**dataset_hyperparameters)
            
        
        if division_type == "manual_test": # Test2: use the trained model to predict "manual_classified", do predict_test (classification) and maybe also test (binary)

            data_module_test2_1 = data_module.clone()
            data_module_test2_1.delete_dataset("test")

            dataset_hyperparameters = copy.deepcopy(dataset_hyps)
            dataset_hyperparameters["dataset_dir_path"] = [
                os.path.join(dataset_dir_path, part_type) for part_type in ["verbals.json", "manual_classified.json"]
            ]
            dataset_hyperparameters["train_percentage"] = 0.0
            dataset_hyperparameters["dev_percentage"] = 0.0
            dataset_hyperparameters["test_percentage"] = 1.0
            data_module_test2_2 = DynamicDataModule(**dataset_hyperparameters)

            data_module = data_module_test2_1 + data_module_test2_2

        if division_type == "manual_in_train_test": # Test3: show that the model is better with also "manual_classified" in the training

            data_module_test3_1 = data_module.clone()

            dataset_hyperparameters = copy.deepcopy(dataset_hyps)
            dataset_hyperparameters["dataset_dir_path"] = [
                os.path.join(dataset_dir_path, part_type) for part_type in ["verbals.json", "manual_classified.json"]
            ]
            dataset_hyperparameters["train_percentage"] = 0.8
            dataset_hyperparameters["dev_percentage"] = 0.2
            dataset_hyperparameters["test_percentage"] = 0.0
            data_module_test3_2 = DynamicDataModule(**dataset_hyperparameters)

            data_module = data_module_test3_1 + data_module_test3_2


        if data_module == None: raise Exception("Please specify the correct test")
        return data_module
    

pretrainedBiencoder = PretrainedBiencoder()
def negative_embedding_finder_data_construction(self, unformatted_data):
    import os, pickle
    file_path = "embs.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as input_file:
            # Load the dictionary using load function
            embeddings = pickle.load(input_file)
    else:
        # Computing all embeddings
        embeddings = {}
        for syns_type in ["nominal_definitions", "verbal_definitions"]:
            embeddings[syns_type] = {}
            for frame_name, synsets_dict in tqdm(unformatted_data[syns_type].items(), desc="Computing embeddings for frame"):
                embeddings[syns_type][frame_name] = {}
                for syn, syn_values in synsets_dict.items():
                    embeddings[syns_type][frame_name][syn] = pretrainedBiencoder.encode(syn_values["wn_definition"])
        with open(file_path, "wb") as output_file:
            pickle.dump(embeddings, output_file)
    
    similarity_threshold = 0.5
    all_verbals_names = [verbal_syn_name for verbal_frame_name, verbal_frame in unformatted_data["verbal_definitions"].items() for verbal_syn_name in verbal_frame.keys()]
    all_verbals_frames =  [verbal_frame_name for verbal_frame_name, verbal_frame in unformatted_data["verbal_definitions"].items() for verbal_syn_name in verbal_frame.keys()]
    all_verbals_tensors = torch.stack([embeddings["verbal_definitions"][verbal_frame_name][verbal_syn_name] for verbal_frame_name, verbal_frame in unformatted_data["verbal_definitions"].items() for verbal_syn_name in verbal_frame.keys()])
    similarities = {}
    for frame_name in tqdm(embeddings["nominal_definitions"].keys(), desc="Computing similarities for frame"):
        for nominal_syn_name, nominal_emb in embeddings["nominal_definitions"][frame_name].items():
            s = torch.nn.functional.cosine_similarity(nominal_emb.unsqueeze(0),all_verbals_tensors,dim=1).tolist()
            similarities[nominal_syn_name] = sorted(zip(all_verbals_frames,all_verbals_names,s),key=lambda x: x[2],reverse=True)
                    
    data = []
    nominal_definitions = list(unformatted_data["nominal_definitions"].items())
        
    for frame_name, nominal_synsets_dict in tqdm(nominal_definitions):
        for nominal_synset_name, nominal_syn in nominal_synsets_dict.items():
            nominal_synset_def = nominal_syn["wn_definition"]
            # Cleaning the definitions and removing duplicates
            if self.clean_definitions != False:  nominal_synset_def = self.filter_definitions([nominal_synset_def], self.clean_definitions)[0]
            nominal_str = self.format_sentence(nominal_syn,nominal_synset_def,self.use_synset_name,self.use_lemmas)
                
            verbal_synsets_dict = copy.deepcopy(unformatted_data["verbal_definitions"][frame_name])
            # A list of tuples of usable verbal defs. Each tuple is (synset object, chosen definition of the synset)
            verbal_synsets_defs = [(verb_syn,verb_syn["wn_definition"]) for verb_syn in verbal_synsets_dict.values()]
            # Cleaning the definitions and removing duplicates
            if self.clean_definitions != False:  verbal_synsets_defs = self.filter_definitions(verbal_synsets_defs, self.clean_definitions)
            # Counting the total number of usable definitions in this frame
            num_verbal = min(len(verbal_synsets_defs),self.max_num_verbal)
            for i in range(num_verbal):
                verbal_def_tuple = verbal_synsets_defs.pop(random.randrange(0,len(verbal_synsets_defs)))
                verbal_str = self.format_sentence(verbal_def_tuple[0],verbal_def_tuple[1],self.use_synset_name,self.use_lemmas)
                if self.sentence_separator != None:
                    final_input = f"{nominal_str} {self.sentence_separator} {verbal_str}"
                else:
                    final_input = [nominal_str,verbal_str]
                data.append([final_input, 1.0])
            
            closer_verbals = similarities[nominal_syn_name]
            selected = 0
            index = 0
            while(selected < num_verbal):
                negative_frame_name, negative_syn_name, similarity = closer_verbals[index]
                index += 1 
                if negative_frame_name == frame_name:
                    continue
                negative_syn = unformatted_data["verbal_definitions"][negative_frame_name][negative_syn_name]
                negative_syn_def = negative_syn["wn_definition"]
                # Cleaning the definitions and removing duplicates
                if self.clean_definitions != False:  negative_syn_def = self.filter_definitions([negative_syn_def], self.clean_definitions)[0]
                negative_str = self.format_sentence(negative_syn,negative_syn_def, self.use_synset_name,self.use_lemmas)
                if self.sentence_separator != None:
                    final_input = f"{nominal_str} {self.sentence_separator} {negative_str}"
                else:
                    final_input = [nominal_str,negative_str]
                data.append([final_input, 0.0])
                selected += 1
    return data