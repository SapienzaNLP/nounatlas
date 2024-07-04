import wordnet_utils as wnutils
from wordnet_utils import wn

import os
from collections import Counter
import random
import pickle
import json
import copy
from tqdm import tqdm

DIR = os.path.realpath(os.path.dirname(__file__))
random.seed(1749274123)

# HYPERPARAMETERS
frame_connection_frequency_threshold = 2 # minimum number of additional connections to the most frequent frame compared to the other frames
split_percentage = 0.8 # train and dev split percentage

# Writing base dataset files
dataset_folder_path = f"{DIR}/../../datasets/"
base_path = dataset_folder_path+ "base_nominal_classification/"
without_manually_classified_path = base_path+ "without_manually_classified/"

classified_nominal_filename = "classified_nominal.tsv"
verbal_to_drfs_filename = "verbal_to_derivationally_related_forms.tsv"
non_ambiguous_nominal_filename = "non_ambiguous_nominal.tsv"
disambiguated_nominal_filename = "disambiguated_nominal.tsv"
ambiguous_nominal_filename = "ambiguous_nominal.tsv"
unclassified_nominal_filename = "unclassified_nominal.tsv"
manually_classified_nominal_filename = "manual_classified_nominal.tsv"

# Frame to index reference
frame2idx_path = base_path+"frame2idx.json"
frame2idx = {}
with open(frame2idx_path, "w") as f:
    for index, frame_name in enumerate(wnutils.inventory.va_dict.keys()):
        frame2idx[frame_name] = index
    json.dump(frame2idx, f, indent=4)

# Index to frame reference
idx2frame_path = base_path+"idx2frame.json"
idx2frame = {}
with open(idx2frame_path, "w") as f:
    for frame_name, index in frame2idx.items():
        idx2frame[index] = frame_name
    json.dump(idx2frame, f, indent=4)

e_list = [
    "event.n.01",
    "event.n.02",
    "event.n.03",
    "event.n.04",
    "act.n.02",
    "process.n.01",
    "process.n.02",
    "process.n.06",
    "action.n.01",
]
event_hypernyms = set([wn.synset(n) for n in e_list])

# verbal_synsets = [wn.synset(syn) for syn in wnutils.inventory.get_wordnet_synsets()]

nominal_synsets = {}
for wn_name in wnutils.inventory.get_wordnet_synsets():
    verbal_synset = wn.synset(wn_name)
    drfs = wnutils.get_derivetionally_releted_synsets(verbal_synset)
    for drf in drfs:
        hyps = set(wnutils.get_all_hypernyms(drf)+[drf])
        if len(hyps & event_hypernyms) == 0:
            continue # Skipping non event synsets
        nominal_synsets.setdefault(drf.name(),Counter()).update([wnutils.inventory.get_verbatlas_frame(wn_name)])
        
unclassified_nominal_event_synsets = set()
for hypernym in event_hypernyms:
    for synset in wnutils.get_all_hyponyms(hypernym):
        unclassified_nominal_event_synsets.add(synset.name())
unclassified_nominal_event_synsets.difference_update(nominal_synsets.keys())
unclassified_nominal_event_synsets = sorted(list(unclassified_nominal_event_synsets))
    
non_ambiguous_nominal_synsets = {}
ambiguous_nominal_synsets = {}
disambiguated_nominal_synsets = {}

for drf,frames_counter in nominal_synsets.items():
    moc = frames_counter.most_common()
    if len(moc) == 1:
        non_ambiguous_nominal_synsets[drf] = moc[0][0]
    else:
        ambiguous_nominal_synsets[drf] = moc
        
 
to_remove = []
for synset_name, moc in ambiguous_nominal_synsets.items():
    under_threshold = True
    max_frequency = moc[0][1] - frame_connection_frequency_threshold
    if max_frequency < 0: continue # Skipping frames that have multiple low frequency connections to frames (the highest frequency connection is lower than the threshold)
    frequency = moc[1][1] # The next highest frequency
    if frequency < max_frequency:
        disambiguated_nominal_synsets[synset_name] = moc[0][0]
        to_remove.append(synset_name)
        
for rem in to_remove:
    ambiguous_nominal_synsets.pop(rem)

# Generating the file with the correspondance between verbals and their nominal classified drfs
verbal_to_drfs = {}
classified_nominal_synsets = dict()
classified_nominal_synsets.update(non_ambiguous_nominal_synsets)
classified_nominal_synsets.update(disambiguated_nominal_synsets)
for verbal_name in wnutils.inventory.get_wordnet_synsets():
    verbal_synset = wn.synset(verbal_name)
    verbal_to_drfs[verbal_name] = []
    drfs = wnutils.get_derivetionally_releted_synsets(verbal_synset)
    for drf in drfs:
        drf_name = drf.name()
        frame_name = wnutils.inventory.get_verbatlas_frame(verbal_name)
        if drf_name in classified_nominal_synsets and classified_nominal_synsets[drf_name] == frame_name:
            verbal_to_drfs[verbal_name].append(drf_name)
            
print("WITHOUT THE MANUALLY CLASSIFIED SYNSETS:")
print(f"{len(non_ambiguous_nominal_synsets)} non ambiguous nominal synsets")
print(f"{len(disambiguated_nominal_synsets)} disambiguated nominal synsets")
print(f"{len(ambiguous_nominal_synsets)} ambiguous nominal synsets")
print(f"{len(unclassified_nominal_event_synsets)} unclassified nominal synsets")

os.makedirs(without_manually_classified_path,exist_ok=True)

with open(without_manually_classified_path+classified_nominal_filename, "w") as f:
    for synset_name in nominal_synsets.keys():
        synset = wn.synset(synset_name)
        f.write(f"{synset.name()}\t{synset.definition()}\n")
        
with open(without_manually_classified_path+verbal_to_drfs_filename, "w") as f:
    for synset_name, drfs in verbal_to_drfs.items():
        drfs_str = "\t".join(drfs)
        f.write(f"{synset_name}\t{drfs_str}\n")

with open(without_manually_classified_path+unclassified_nominal_filename, "w") as f:
    for synset_name in unclassified_nominal_event_synsets:
        synset = wn.synset(synset_name)
        f.write(f"{synset.name()}\t{synset.definition()}\n")

with open(without_manually_classified_path+non_ambiguous_nominal_filename, "w") as f:
    for synset_name, frame_name in non_ambiguous_nominal_synsets.items():
        synset = wn.synset(synset_name)
        f.write(f"{synset.name()}\t{synset.definition()}\t{frame_name}\n")

with open(without_manually_classified_path+disambiguated_nominal_filename, "w") as f:
    for synset_name, frame_name in disambiguated_nominal_synsets.items():
        synset = wn.synset(synset_name)
        f.write(f"{synset.name()}\t{synset.definition()}\t{frame_name}\n")
        
with open(without_manually_classified_path+ambiguous_nominal_filename, "w") as f:
    for synset_name, frame_frequencies in ambiguous_nominal_synsets.items():
        synset = wn.synset(synset_name)
        freq_str = "\t".join([str(e) for freq_tuple in frame_frequencies for e in freq_tuple])
        f.write(f"{synset.name()}\t{synset.definition()}\t{freq_str}\n")
        

# Adding the manually classified synsets
# The manually classified synsets file is manually created. It's composed of all the ambiguous synsets and some synset we reclassified 
manually_classified_nominal_synsets = {}
with open(base_path+manually_classified_nominal_filename, "r") as f:
    for line in f:
        cols = line.strip().split("\t")
        if len(cols) != 3: continue
        synset_name = cols[0].strip()
        frame_name = cols[2].strip()
        
        # Sanity check
        try:
            wnutils.wn.synset(synset_name)
        except:
            raise Exception(f"{synset_name} doesn't exist as a synset")
        
        if frame_name not in frame2idx:
            raise Exception(f"{frame_name} doesn't exist as a frame")
        
        manually_classified_nominal_synsets[synset_name] = frame_name
        
        nominal_synsets.setdefault(synset_name,Counter()).update([frame_name])
        # Removing the manually classified synset to the non ambiguous and disambiguated (N.B. this way the manual classification has precedence!)
        if synset_name in non_ambiguous_nominal_synsets: non_ambiguous_nominal_synsets.pop(synset_name)
        if synset_name in disambiguated_nominal_synsets: disambiguated_nominal_synsets.pop(synset_name)
        # Removing the manually classified synset from the other sets
        if synset_name in ambiguous_nominal_synsets: ambiguous_nominal_synsets.pop(synset_name)
        if synset_name in unclassified_nominal_event_synsets: unclassified_nominal_event_synsets.remove(synset_name)
        
# Generating the file with the correspondance between verbals and their nominal classified drfs
verbal_to_drfs = {}
classified_nominal_synsets = dict()
classified_nominal_synsets.update(non_ambiguous_nominal_synsets)
classified_nominal_synsets.update(disambiguated_nominal_synsets)
classified_nominal_synsets.update(manually_classified_nominal_synsets)
for verbal_name in wnutils.inventory.get_wordnet_synsets():
    verbal_synset = wn.synset(verbal_name)
    verbal_to_drfs[verbal_name] = []
    drfs = wnutils.get_derivetionally_releted_synsets(verbal_synset)
    for drf in drfs:
        drf_name = drf.name()
        frame_name = wnutils.inventory.get_verbatlas_frame(verbal_name)
        if drf_name in classified_nominal_synsets and classified_nominal_synsets[drf_name] == frame_name:
            verbal_to_drfs[verbal_name].append(drf_name)
        
print()
print(f"WITH THE MANUALLY CLASSIFIED SYNSETS ({len(manually_classified_nominal_synsets)} manually classified):")
print(f"{len(non_ambiguous_nominal_synsets)} non ambiguous nominal synsets")
print(f"{len(disambiguated_nominal_synsets)} disambiguated nominal synsets")
print(f"{len(ambiguous_nominal_synsets)} ambiguous nominal synsets")
print(f"{len(unclassified_nominal_event_synsets)} unclassified nominal synsets")      

os.makedirs(base_path,exist_ok=True)

with open(base_path+classified_nominal_filename, "w") as f:
    for synset_name in nominal_synsets.keys():
        synset = wn.synset(synset_name)
        f.write(f"{synset.name()}\t{synset.definition()}\n")
        
with open(base_path+verbal_to_drfs_filename, "w") as f:
    for synset_name, drfs in verbal_to_drfs.items():
        drfs_str = "\t".join(drfs)
        f.write(f"{synset_name}\t{drfs_str}\n")

with open(base_path+unclassified_nominal_filename, "w") as f:
    for synset_name in unclassified_nominal_event_synsets:
        synset = wn.synset(synset_name)
        f.write(f"{synset.name()}\t{synset.definition()}\n")

with open(base_path+non_ambiguous_nominal_filename, "w") as f:
    for synset_name, frame_name in non_ambiguous_nominal_synsets.items():
        synset = wn.synset(synset_name)
        f.write(f"{synset.name()}\t{synset.definition()}\t{frame_name}\n")

with open(base_path+disambiguated_nominal_filename, "w") as f:
    for synset_name, frame_name in disambiguated_nominal_synsets.items():
        synset = wn.synset(synset_name)
        f.write(f"{synset.name()}\t{synset.definition()}\t{frame_name}\n")
        
with open(base_path+ambiguous_nominal_filename, "w") as f:
    for synset_name, frame_frequencies in ambiguous_nominal_synsets.items():
        synset = wn.synset(synset_name)
        freq_str = "\t".join([str(e) for freq_tuple in frame_frequencies for e in freq_tuple])
        f.write(f"{synset.name()}\t{synset.definition()}\t{freq_str}\n")

# Check if there are some frames with less than 2 synsets in them. Those frames can't be split in training set and validation
# print()
# min_len = 100000
# for frame in wnutils.inventory.get_verbatlas_frame():
#     synsets = wnutils.inventory.get_wordnet_synsets(frame)
#     min_len = min(min_len, len(synsets))
#     if len(synsets) < 2:
#         print(f"{frame} has {len(synsets)}")
# print(f"The least amount of synsets in a frame is {min_len}")
# print()

#region Creating a dictionary indexed by frame of the verbal synset definitions
verbal_definitions = {}
for frame_name in tqdm(wnutils.inventory.get_verbatlas_frame()):
    verbal_definitions[frame_name] = {}
    for synset_name in wnutils.inventory.get_wordnet_synsets(frame_name):
        lemmas = [l.name().replace("_"," ") for l in wnutils.wn.synset(synset_name).lemmas()]
        wn_def = wnutils.wn.synset(synset_name).definition()
        verbal_definitions[frame_name][synset_name] = {"synset_name":synset_name, "lemmas":lemmas,"wn_definition":wn_def, "frame_name":frame_name, "frame_index":frame2idx[frame_name]}
#endregion

#region Creating the dictionaries indexed by frame of nominal synset definitions
non_ambiguous_nominal_definitions = {}
for synset_name, frame_name in tqdm(non_ambiguous_nominal_synsets.items()):
    lemmas = [l.name().replace("_"," ") for l in wnutils.wn.synset(synset_name).lemmas()]
    wn_def = wnutils.wn.synset(synset_name).definition()
    non_ambiguous_nominal_definitions.setdefault(frame_name,{})[synset_name] = {"synset_name":synset_name, "lemmas":lemmas,"wn_definition":wn_def, "frame_name":frame_name, "frame_index":frame2idx[frame_name]}
    
disambiguated_nominal_definitions = {}
for synset_name, frame_name in tqdm(disambiguated_nominal_synsets.items()):
    lemmas = [l.name().replace("_"," ") for l in wnutils.wn.synset(synset_name).lemmas()]
    wn_def = wnutils.wn.synset(synset_name).definition()
    disambiguated_nominal_definitions.setdefault(frame_name,{})[synset_name] = {"synset_name":synset_name, "lemmas":lemmas,"wn_definition":wn_def, "frame_name":frame_name, "frame_index":frame2idx[frame_name]}
    
manually_classified_nominal_definitions = {}
for synset_name, frame_name in tqdm(manually_classified_nominal_synsets.items()):
    lemmas = [l.name().replace("_"," ") for l in wnutils.wn.synset(synset_name).lemmas()]
    wn_def = wnutils.wn.synset(synset_name).definition()
    manually_classified_nominal_definitions.setdefault(frame_name,{})[synset_name] = {"synset_name":synset_name, "lemmas":lemmas,"wn_definition":wn_def, "frame_name":frame_name, "frame_index":frame2idx[frame_name]}
    
nominal_definitions = {}
all_items = list(non_ambiguous_nominal_synsets.items()) + list(disambiguated_nominal_synsets.items()) + list(manually_classified_nominal_synsets.items())
for synset_name, frame_name in tqdm(all_items):
    lemmas = [l.name().replace("_"," ") for l in wnutils.wn.synset(synset_name).lemmas()]
    wn_def = wnutils.wn.synset(synset_name).definition()
    nominal_definitions.setdefault(frame_name,{})[synset_name] = {"synset_name":synset_name, "lemmas":lemmas,"wn_definition":wn_def, "frame_name":frame_name, "frame_index":frame2idx[frame_name]}
#endregion

#region Creating the dictionary of synsets to test
ambiguous_nominal_definitions = {}
all_items = list(ambiguous_nominal_synsets.items())
for synset_name, moc in tqdm(all_items):
    frame_names = [f[0] for f in moc]
    frame_indices = [frame2idx[f] for f in frame_names]
    lemmas = [l.name().replace("_"," ") for l in wnutils.wn.synset(synset_name).lemmas()]
    wn_def = wnutils.wn.synset(synset_name).definition()
    ambiguous_nominal_definitions[synset_name] = {"synset_name":synset_name, "lemmas":lemmas,"wn_definition":wn_def,"bn_definitions": bn_defs, "frame_name": frame_names, "frame_index": frame_indices}

unclassified_nominal_definitions = {}
all_items = list(unclassified_nominal_event_synsets)
for synset_name in tqdm(all_items):
    lemmas = [l.name().replace("_"," ") for l in wnutils.wn.synset(synset_name).lemmas()]
    unclassified_nominal_definitions[synset_name] = {"synset_name":synset_name, "lemmas":lemmas,"wn_definition":wnutils.wn.synset(synset_name).definition().lower() ,"bn_definitions": [], "frame_name": None, "frame_index": None}
#endregion

#region Dataset single definitions
dataset_path = dataset_folder_path+"dataset_nominal_classification/"
os.makedirs(dataset_path, exist_ok=True)

all_definitions = {
    "nominal_definitions": nominal_definitions,
    "verbal_definitions": verbal_definitions
}
with open(dataset_path+"dataset.json", "w") as tf:
    json.dump(all_definitions, tf, indent="\t")

verbal_definitions_filename = "verbals.json"
non_ambiguous_nominal_filename = "non_ambiguous.json"
disambiguated_nominal_filename = "disambiguated.json"
manually_classified_nominal_filename = "manual_classified.json"
ambiguous_nominal_filename = "ambiguous.json"
unclassified_nominal_filename = "unclassified.json"

with open(dataset_path+verbal_definitions_filename, "w") as tf:
    json.dump({
        "verbal_definitions": verbal_definitions,
    }, tf, indent="\t")

with open(dataset_path+non_ambiguous_nominal_filename, "w") as tf:
    json.dump({
        "nominal_definitions": non_ambiguous_nominal_definitions,
    }, tf, indent="\t")

with open(dataset_path+disambiguated_nominal_filename, "w") as tf:
    json.dump({
        "nominal_definitions": disambiguated_nominal_definitions,
    }, tf, indent="\t")

with open(dataset_path+manually_classified_nominal_filename, "w") as tf:
    json.dump({
        "nominal_definitions": manually_classified_nominal_definitions,
    }, tf, indent="\t")

with open(dataset_path+ambiguous_nominal_filename, "w") as tf:
    json.dump(ambiguous_nominal_definitions, tf, indent="\t")

with open(dataset_path+unclassified_nominal_filename, "w") as tf:
    json.dump(unclassified_nominal_definitions, tf, indent="\t")
    

print("Dynamic dataset definitions files created")
#endregion