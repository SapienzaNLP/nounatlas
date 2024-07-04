import os, sys, json, math
import re
import random
import copy
from typing import List, Tuple, Dict, Union, Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch import as_tensor
import torchvision.transforms as T
import lightning.pytorch as pl
import numpy as np
import pickle
from tqdm import tqdm

def merge_dicts(dict1, dict2):
    merged_dict = {}
    for key in dict1.keys():
        if key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merged_dict[key] = merge_dicts(dict1[key], dict2[key])
            elif isinstance(dict1[key], list) and isinstance(dict2[key], list):
                merged_dict[key] = merge_lists(dict1[key], dict2[key])
            else:
                merged_dict[key] = dict1[key]
        else:
            merged_dict[key] = dict1[key]
            
    for key, value in dict2.items():
        if key not in merged_dict:
            merged_dict[key] = value
    return merged_dict

def merge_lists(list1, list2):
    return copy.deepcopy(list1 + [elem for elem in list2 if elem not in list1])
    

DIR = os.path.realpath(os.path.dirname(__file__))
random.seed(1749274)
frame2idx_filepath = os.path.abspath(os.path.join(DIR,"../..", "datasets/base_nominal_classification/frame2idx.json"))
idx2frame_filepath = os.path.abspath(os.path.join(DIR,"../..", "datasets/base_nominal_classification/idx2frame.json"))
general_datasets_path = os.path.abspath(os.path.join(DIR,"../..", "datasets/"))
datasets_paths = []
for directory in os.listdir(general_datasets_path):
    if os.path.isdir(os.path.join(general_datasets_path, directory)) and directory.startswith("dataset"):
        datasets_paths.append(os.path.join(general_datasets_path, directory))
datasets_paths = sorted(datasets_paths)

class DynamicDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_dir_path: Union[str, List[str]] = None, # Path of the pickle or json dataset file(s) generated using a "build_dataset.py" script
            # dataset_files_paths: Union[List[str], List[List[str]]] = None, # A list of path 
            built_dataset_save_path: Union[str,List[str]] = None, # If a string, should be a folder path to save train.tsv and dev.tsv. If a list, should be 2 full file paths
            train_percentage = 0.8,
            dev_percentage = 0.2,
            test_percentage = 0.0,
            split_verbal = False, # Wheter to split the verbal or use all of them in each set
            use_only_test_frames = False, # 
            batch_size: int = 8,
            num_workers: int = 1,
            **kwargs
        ):
        super().__init__()
        # if dataset_dir_path == None and dataset_files_paths == None:
        #     raise Exception("At least one of 'dataset_dir_path' or 'dataset_files_paths' must be provided with a valid value")
        self.save_hyperparameters()
        setattr(self,"ready",False)

    def prepare_data(self, stage: Optional[str] = None):
        if getattr(self, "ready", False) == True: return
                
        dataset_args = {key:val for key,val in self.hparams.items()}
        
        dataset_dir_path: Union[str, List[str]] = dataset_args.pop("dataset_dir_path")
        unformatted_data = {}
        if isinstance(dataset_dir_path,str):
            if (dataset_dir_path.endswith("pkl") or dataset_dir_path.endswith("json")):
                unformatted_data = self.read_unformatted_data(dataset_dir_path)
            else:
                unformatted_data = self.read_unformatted_data(os.path.join(dataset_dir_path,"dataset.json"))
        elif isinstance(dataset_dir_path, list) and len(dataset_dir_path)>0:
                unformatted_data = {}
                for dataset_filename in dataset_dir_path:
                    other_unformatted_data = self.read_unformatted_data(dataset_filename)
                    unformatted_data = merge_dicts(unformatted_data,other_unformatted_data)
        else:
            raise Exception("Unsupported value for dataset_dir_path parameters: Supported: str | List[str]")        
                
        # Splitting the arguments per dataset
        train_dataset_args = {
            "dataset_type": "train"
        }
        dev_dataset_args = {
            "dataset_type": "dev"
        }
        test_dataset_args = {
            "dataset_type": "test"
        }
        for key,val in dataset_args.items():
            if isinstance(val,list):
                train_dataset_args[key] = val[0]
                dev_dataset_args[key] = val[1] if len(val) > 1 else val[0]
                test_dataset_args[key] = val[2] if len(val) > 2 else val[0]
            else:
                train_dataset_args[key] = val
                dev_dataset_args[key] = val
                test_dataset_args[key] = val
                
        # Fixing split percentage values
        remaining = 1 - (dataset_args["train_percentage"] + dataset_args["dev_percentage"] + dataset_args["test_percentage"])
        dataset_args["train_percentage"] += remaining
        
        # Nominal split
        nominal_frames = unformatted_data["nominal_definitions"]
        nominal_train_data = {}
        nominal_dev_data = {}
        nominal_test_data = {}
        selected_dev_set = 0
        selected_test_set = int(not selected_dev_set)
        
        # Setting up the debug statistics object
        with open(frame2idx_filepath, "r") as f:
            frame2idx = json.load(f)
        debug_stats = { 
            frame_name: {
                "total": 0,
                "train": {"precise":0, "final":0, "percentage":0},
                "dev": {"precise":0, "final":0, "percentage":0},
                "test": {"precise":0, "final":0, "percentage":0},
            } 
            for frame_name in frame2idx.keys()
        }
        total_samples = 0
        total_train_sample = 0
        total_dev_sample = 0
        total_test_sample = 0
        
        # Looping over the nominal synsets to create the splits
        for frame_name, synsets_dict in nominal_frames.items():
            tot_num_defs = len(synsets_dict)
            max_num_train_defs = tot_num_defs*dataset_args["train_percentage"]
            max_num_dev_defs = tot_num_defs*dataset_args["dev_percentage"]
            max_num_test_defs = tot_num_defs*dataset_args["test_percentage"]
            
            rounded_tot = round(max_num_train_defs)+round(max_num_dev_defs)+round(max_num_test_defs)
            train_diff = 0
            dev_diff = 0
            test_diff = 0
            if rounded_tot != tot_num_defs:
            # if rounded_tot < tot_num_defs:
                diff = tot_num_defs-rounded_tot
                selected_dev_set = (selected_dev_set+1)%2
                selected_test_set = (selected_test_set+1)%2
                dev_diff = diff*selected_dev_set
                test_diff = diff*selected_test_set
            # elif rounded_tot > tot_num_defs:
            #     train_diff = tot_num_defs-rounded_tot
            
            num_train_defs = 0
            num_dev_defs = 0
            num_test_defs = 0
            syn_list = list(synsets_dict.items())
            # random.shuffle(syn_list)
            for synset_name, syn in syn_list:
                # Adding synset to the training set
                if num_train_defs < round(max_num_train_defs)+train_diff:
                    nominal_train_data.setdefault(frame_name,{})[synset_name] = syn
                    num_train_defs += 1
                # Adding synset to the test set    
                elif num_test_defs < round(max_num_test_defs) + test_diff:
                    nominal_test_data.setdefault(frame_name,{})[synset_name] = syn
                    num_test_defs += 1
                # Adding synset to the dev set                     
                elif num_dev_defs < round(max_num_dev_defs) + dev_diff:
                    nominal_dev_data.setdefault(frame_name,{})[synset_name] = syn
                    num_dev_defs += 1
                
            debug_stats[frame_name]["total"] = tot_num_defs
            debug_stats[frame_name]["train"] = {"precise":tot_num_defs*dataset_args["train_percentage"], "final":num_train_defs, "percentage": num_train_defs/tot_num_defs}
            debug_stats[frame_name]["dev"] = {"precise":tot_num_defs*dataset_args["dev_percentage"], "final":num_dev_defs, "percentage": num_dev_defs/tot_num_defs}
            debug_stats[frame_name]["test"] = {"precise":tot_num_defs*dataset_args["test_percentage"], "final":num_test_defs, "percentage": num_test_defs/tot_num_defs}
            total_samples += num_train_defs+num_dev_defs+num_test_defs
            total_train_sample += num_train_defs
            total_dev_sample += num_dev_defs
            total_test_sample += num_test_defs
            
        built_dataset_save_path = train_dataset_args["built_dataset_save_path"]
        if built_dataset_save_path != None:
            os.makedirs(built_dataset_save_path, exist_ok=True)
            with open(os.path.join(built_dataset_save_path,"debug_stats.json"), "w") as f:
                debug_stats["$DEBUG"] = {
                    "total_samples": total_samples,
                    "total_train_sample": total_train_sample,
                    "total_dev_sample": total_dev_sample,
                    "total_test_sample": total_test_sample,
                    "total_train_percentage": total_train_sample/total_samples,
                    "total_dev_percentage": total_dev_sample/total_samples,
                    "total_test_percentage": total_test_sample/total_samples,
                }
                json.dump(debug_stats, f, indent=4)
                
        # Verbal split
        if train_dataset_args["split_verbal"]:
            verbal_frames = unformatted_data["verbal_definitions"]
            verbal_train_data = {}
            verbal_dev_data = {}
            verbal_test_data = {}
            # Counting the total number of definitions for each frame, to compute the split threshold
            for frame_name, synsets_dict in verbal_frames.items():
                tot_num_defs = len(synsets_dict)
                max_num_train_defs = tot_num_defs*dataset_args["train_percentage"]
                max_num_dev_defs = tot_num_defs*dataset_args["dev_percentage"]
                max_num_test_defs = tot_num_defs*dataset_args["test_percentage"]
                
                num_train_defs = 0
                num_dev_defs = 0
                num_test_defs = 0
                syn_list = list(synsets_dict.items())
                # random.shuffle(syn_list)
                for synset_name, syn in syn_list:
                    # Adding synset to the training set
                    if num_train_defs < max_num_train_defs:
                        verbal_train_data.setdefault(frame_name,{})[synset_name] = syn
                        num_train_defs += 1
                    # Adding synset to the dev set    
                    elif num_dev_defs < max_num_dev_defs and dev_dataset_args["split_verbal"]:
                        verbal_dev_data.setdefault(frame_name,{})[synset_name] = syn
                        num_dev_defs += 1
                    # Adding synset to the test set                            
                    elif num_test_defs < max_num_test_defs and test_dataset_args["split_verbal"]:
                        verbal_test_data.setdefault(frame_name,{})[synset_name] = syn
                        num_test_defs += 1
                    else:
                        break
                    
            if not dev_dataset_args["split_verbal"]:
                verbal_dev_data = verbal_frames
            if not test_dataset_args["split_verbal"]:
                verbal_test_data = verbal_frames
        else:
            # Using all the verbal synset in each set
            verbal_frames = unformatted_data["verbal_definitions"]
            verbal_train_data = verbal_frames
            verbal_dev_data = verbal_frames
            verbal_test_data = verbal_frames
        
        
        if dataset_args["use_only_test_frames"]:
            frames_to_use = list(nominal_test_data.keys())
            [nominal_train_data.pop(frame_name) for frame_name in list(nominal_train_data.keys()) if frame_name not in frames_to_use]
            [verbal_train_data.pop(frame_name) for frame_name in list(verbal_train_data.keys()) if frame_name not in frames_to_use]
            
            [nominal_dev_data.pop(frame_name) for frame_name in list(nominal_dev_data.keys()) if frame_name not in frames_to_use]
            [verbal_dev_data.pop(frame_name) for frame_name in list(verbal_dev_data.keys()) if frame_name not in frames_to_use]
            
            [verbal_test_data.pop(frame_name) for frame_name in list(verbal_test_data.keys()) if frame_name not in frames_to_use]
        
        train_unformatted_data = {
            "nominal_definitions": nominal_train_data,
            "verbal_definitions": verbal_train_data
        }
        dev_unformatted_data = {
            "nominal_definitions": nominal_dev_data,
            "verbal_definitions": verbal_dev_data
        }
        test_unformatted_data = {
            "nominal_definitions": nominal_test_data,
            "verbal_definitions": verbal_test_data
        }
        
        if dataset_args["train_percentage"] > 0:      
            self.train_dataset = DynamicDataset(unformatted_data = train_unformatted_data, **train_dataset_args)
        else:
            self.train_dataset = DynamicDataset(unformatted_data = {}, prebuilt_data = [], **train_dataset_args)
            
        if dataset_args["dev_percentage"] > 0:
            self.dev_dataset = DynamicDataset(unformatted_data = dev_unformatted_data, **dev_dataset_args)
        else:
            self.dev_dataset = DynamicDataset(unformatted_data = {}, prebuilt_data = [], **dev_dataset_args)
        
        if dataset_args["test_percentage"] > 0:
            self.test_dataset = DynamicDataset(unformatted_data = test_unformatted_data, **test_dataset_args)
        else:
            self.test_dataset = DynamicDataset(unformatted_data={}, prebuilt_data = [], **test_dataset_args)
        
        # This flag is needed to not execute the prepare_data again if already done
        self.ready = True

    def __add__(self, other: "DynamicDataModule") -> "DynamicDataModule":
        if getattr(self, "ready", False) == False: self.prepare_data()
        if getattr(other, "ready", False) == False: other.prepare_data()
        
        new_datamodule = self.__class__(**self.hparams)
        new_datamodule.ready = True
        
        new_datamodule.train_dataset = self.train_dataset+other.train_dataset
        new_datamodule.dev_dataset = self.dev_dataset+other.dev_dataset
        new_datamodule.test_dataset = self.test_dataset+other.test_dataset
        return new_datamodule
    
    def delete_dataset(self,dataset_type: str):
        if dataset_type == "train":
            self.train_dataset.delete_dataset_data()
        elif dataset_type == "dev":
            self.dev_dataset.delete_dataset_data()
        elif dataset_type == "test":
            self.test_dataset.delete_dataset_data()
        else:
            raise Exception(f"Dataset type {dataset_type} unsupported. Supporting only: train, dev, test")
        return self
    
    def clone(self):
        if getattr(self, "ready", False) == False: self.prepare_data()
        
        new_datamodule = self.__class__(**self.hparams)
        new_datamodule.ready = True
        new_datamodule.train_dataset = self.train_dataset.clone()
        new_datamodule.dev_dataset = self.dev_dataset.clone()
        new_datamodule.test_dataset = self.test_dataset.clone()
        return new_datamodule

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate,
        )

    def val_dataloader(self) :
        return DataLoader(
            self.dev_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate,
        )
        
    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                collate_fn=self.collate,
            )
        else: return super().test_dataloader()
    
    def collate(self, batch: List[Tuple[str,str]]):
        # If the batch returned by the dataset is a list of lists, rebuild it into a dictionary of lists
        if type(batch[0]) == list:
            batch_out = {
            "input": [d[0] for d in batch],
            "target": self.list_to_tensor([d[1] for d in batch])
            # "target": torch.Tensor([d[1] for d in batch]).to(torch.long if not self.hparams.binary_dataset else torch.float)
        }
        # Else, if the batch is a list of dicts, rebuild it into a dictionary with keys equal to the list items, but the single keys are aggregated into lists
        elif type(batch[0]) == dict:
            batch_out = {}
            for key in batch[0].keys():
                batch_out[key] = self.list_to_tensor([sample[key] for sample in batch])
        else:
            batch_out = batch_out
        
        return batch_out
    
    @staticmethod
    def list_to_tensor(lst):
        first_elem = lst[0]
        if isinstance(first_elem, int):
            return torch.tensor(lst, dtype=torch.long)
        elif isinstance(first_elem, float):
            return torch.tensor(lst, dtype=torch.float)
        else:
            return lst
    
    @staticmethod
    def read_unformatted_data(filepath):
        unformatted_data = {}
        if filepath.endswith("pkl"):
            with open(filepath, "rb") as f:
                unformatted_data = pickle.load(f)
        elif filepath.endswith("json"):
            with open(filepath, "r") as f:
                unformatted_data = json.load(f)
        else:
            raise Exception(f"{filepath} is not a supported file. Pass a pickle of json file")

        if "verbal_definitions" not in unformatted_data: unformatted_data["verbal_definitions"] = {}
        if "nominal_definitions" not in unformatted_data: unformatted_data["nominal_definitions"] = {}
        return unformatted_data
    
class DynamicDataset(Dataset):
    def __init__(self, unformatted_data: Dict, prebuilt_data: List = None,
                dataset_type: str = "train",
                use_synset_name = False,
                use_lemmas = False,
                max_num_verbal = -1, # How many verbal definitions to pair with the nominal definition. -1 to take all the verbal definitions in the frame
                sentence_separator: str = "[SEP]", # The separator to use to create the single input sentence. If equal to None, the dataset will be a list of tuples [nominal sentence, verbal sentence]
                clean_definitions = False, # Removes punctuation and capital letters to normalize even more the sentences
                max_num_samples = -1, # -1 to take all the samples in the dataset
                custom_data_construction_fn = None, # Custom function to build the dataset
                built_dataset_save_path = None, # path to save a tsv version of the built dataset
                seed: int = None,
                **kwargs):
        super().__init__()
        self.dataset_type = dataset_type
        self.use_synset_name = use_synset_name
        self.use_lemmas = use_lemmas
        self.max_num_verbal = max_num_verbal if max_num_verbal != -1 else 1e1000000
        self.sentence_separator = sentence_separator
        self.clean_definitions = clean_definitions
        self.max_num_samples = max_num_samples if max_num_samples != -1 else 1e1000000
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Binary dataset requires pairing nominals with verbals
        if self.binary_dataset: self.pair_with_verbal = True
        
        self.unformatted_data = unformatted_data
        self.data = []
        
        if isinstance(prebuilt_data, list):
            self.data = prebuilt_data
        else:
            pl.seed_everything(seed)
            if custom_data_construction_fn:
                self.data = custom_data_construction_fn(self, unformatted_data)
            else:
                self.data = self.default_data_construction(unformatted_data)
                        
            if len(self.data) > self.max_num_samples:
                self.data = self.data[:self.max_num_samples]
            
            if built_dataset_save_path != None:
                os.makedirs(os.path.dirname(built_dataset_save_path), exist_ok=True)
                with open(idx2frame_filepath, "r") as f:
                    idx2frame = json.load(f)
                # Saving final formatted tsv
                with open(os.path.join(built_dataset_save_path,self.dataset_type+".tsv"), "w") as f:
                    for input_data, target in self.data:
                        target = str(target)
                        f.write(f"{input_data}\t{idx2frame[target] if not self.binary_dataset else target}\n")
                # Saving unformatted json data
                with open(os.path.join(built_dataset_save_path,self.dataset_type+".json"), "w") as f:
                    json.dump(self.unformatted_data, f, indent=4)
        
        
        print(f"Dataset len: {len(self.data)}")
        
    def default_data_construction(self, unformatted_data):
        data = []
        nominal_definitions = list(unformatted_data["nominal_definitions"].items())
            
        for frame_name, nominal_synsets_dict in tqdm(nominal_definitions):
            for nominal_synset_name, nominal_syn in nominal_synsets_dict.items():
                nominal_synset_defs = [nominal_syn["wn_definition"]]
                # Cleaning the definitions and removing duplicates
                if self.clean_definitions != False:  nominal_synset_defs = self.filter_definitions(nominal_synset_defs, self.clean_definitions)
                for nominal_synset_def in nominal_synset_defs:
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
                        # If this should be a binary dataset, this is a positive couple, the target should be 1.0, not the frame class index
                        data.append([final_input, 1.0])
                        
                        # Create the negative couple for the positive just created
                        negative_frames = list(unformatted_data["verbal_definitions"].keys())
                        negative_frames.remove(frame_name)
                        random_frame = random.choice(negative_frames)
                        negative_verbal_syn = random.choice(list(unformatted_data["verbal_definitions"][random_frame].values()))
                        negative_verbal_defs = [negative_verbal_syn["wn_definition"]]
                        # Cleaning the definitions and removing duplicates
                        if self.clean_definitions != False:  negative_verbal_defs = self.filter_definitions(negative_verbal_defs, self.clean_definitions)
                        random_verbal_def =  random.choice(negative_verbal_defs)
                        negative_verbal_str = self.format_sentence(negative_verbal_syn,random_verbal_def, self.use_synset_name,self.use_lemmas)
                        if self.sentence_separator != None:
                            final_negative_input = f"{nominal_str} {self.sentence_separator} {negative_verbal_str}"
                        else:
                            final_negative_input = [nominal_str,negative_verbal_str]
                    
                        data.append([final_negative_input, 0.0])
        return data
    
    @staticmethod
    def format_sentence(obj, definition, use_synset_name = False, use_lemmas = False, **kwargs):
        str_start = ""
        if use_lemmas:
            str_start = ", ".join(obj["lemmas"]) + ", "
        elif use_synset_name:
            str_start = f"{obj['lemmas'][0]}, "
            
        s = f"{str_start}{definition}"
        return s.strip()
    
    @staticmethod
    def clean_sentence(s, clean_params = None, **kwargs):
        s = s.lower().strip()
        s = re.sub(r"[^a-zA-Z0-9_\s]","",s)
        s = re.sub(r"\s+"," ",s)
        return s
    
    @staticmethod
    def filter_definitions(definitions, clean_params = None, **kwargs):
        cleaned_definitions = []
        contained_def = lambda x,y: False # Check if x is contained in y
        for definition in definitions:
            if isinstance(definition, str):
                # Definitions is a list of strings
                contained_def = lambda x,y: x in y
                clean_def = DynamicDataset.clean_sentence(definition, clean_params)
                cleaned_definitions.append(clean_def)
            else:
                # Definitions is a list of tuples. Each tuple is (synset object, chosen definition of the synset)
                contained_def = lambda x,y: x[1] in y[1] # TODO: Should check if the defs are of the same synset in order to not remove identical definitions from different synsets?
                clean_def = DynamicDataset.clean_sentence(definition[1], clean_params)
                cleaned_definitions.append((definition[0],clean_def))
                
        usable_definitions = []
        for i in range(len(cleaned_definitions)):
            if cleaned_definitions[i] == None: continue # Don't check an already removed element (this should never happen, here just for safety)
            for c in range(len(cleaned_definitions)):
                if c == i: continue # Don't check with itself
                if cleaned_definitions[c] == None: continue # Don't check against an already removed element
                if contained_def(cleaned_definitions[i], cleaned_definitions[c]):
                    cleaned_definitions[i] = None
                    break
            if cleaned_definitions[i] != None:
                usable_definitions.append(cleaned_definitions[i])
        return usable_definitions
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __add__(self, other: Dataset) -> Dataset:
        self_params = dict(self.__dict__)
        self_data = self_params.pop("data")
        self_unformatted_data = self_params.pop("unformatted_data")
        
        other_params = dict(other.__dict__)
        other_data = other_params.pop("data")
        other_unformatted_data = other_params.pop("unformatted_data")
        
        data = copy.deepcopy([*self_data, *other_data])
        unformatted_data = merge_dicts(self_unformatted_data, other_unformatted_data)
        
        return self.__class__(unformatted_data=unformatted_data, prebuilt_data=data, **self_params)
    
    def clone(self):
        self_params = dict(self.__dict__)
        self_data = copy.deepcopy(self_params.pop("data"))
        self_unformatted_data = copy.deepcopy(self_params.pop("unformatted_data"))
        return self.__class__(unformatted_data=self_unformatted_data, prebuilt_data=self_data, **self_params)
        
    def delete_dataset_data(self):
        self.data = []
        self.unformatted_data = {}
        return self
