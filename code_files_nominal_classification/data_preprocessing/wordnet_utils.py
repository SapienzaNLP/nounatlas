from typing import Union
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn

import os
DIR = os.path.realpath(os.path.dirname(__file__))
bn2wn_path = f"{DIR}/../../resources/VerbAtlas/bn2wn.tsv"
bn2va_path = f"{DIR}/../../resources/VerbAtlas/VA_bn2va.tsv"
va_info_path = f"{DIR}/../../resources/VerbAtlas/VA_frame_info.tsv"

# Class that groups together VerbAtlas, BabelNet and WordNet info taken from the provided files
class Inventory():

    def __init__(self, bn2wn_path, bn2va_path, va_info_path):        
        self.wn_dict = {}
        self.va_dict = {}
        
        va_id2name = {}
        va_name2id = {}
        bn2wn_dict = {}
        bn2va_dict = {}
        
        va2name_list = self.read_tsv(va_info_path, True)
        for d in va2name_list:
            va_id, va_name = d[0], d[1]
            va_id2name[va_id] = va_name
            va_name2id[va_name] = va_id

        bn2wn_list = self.read_tsv(bn2wn_path, True)
        for bn, wn_id in bn2wn_list:
            wn_name = wn.synset_from_pos_and_offset(wn_id[-1],int(wn_id[3:-1])).name()
            bn2wn_dict[bn] = wn_name

        bn2va_list = self.read_tsv(bn2va_path, True)
        for bn, va_id in bn2va_list:
            va_name = va_id2name[va_id]
            bn2va_dict[bn] = va_name

        for bn, wn_name in bn2wn_dict.items():
            va_name = bn2va_dict[bn]
            self.wn_dict[wn_name] = va_name
            self.va_dict.setdefault(va_name,[]).append(wn_name)        

    @staticmethod
    def read_tsv(file_path, header = False):
        ret = []
        with open(file_path) as file:
            if header == True: h = file.readline()
            for line in file:
                line = line.strip()
                if len(line) == 0: continue
                if line.startswith("#"): continue
                split = line.split("\t")
                # if header != None and len(split) < 2: continue
                ret.append(split)
        return ret

    def get_verbatlas_frame(self, wn_name: Union[str, None] = None):
        """Given a wordnet verbal synset name, returns its verbatlas frame
           If the synset name is None, all the verbatlas frames are returned

        Args:
            wn_name (Union[str, None]): wordnet verbal synset name in the format lemma.pos.index. Defaults to None.

        Returns:
            Union[str, List[str]]: Verbatlas frame name
        """
        if wn_name == None:
            return [frame_name for frame_name in self.va_dict.keys()]
        if wn_name in self.wn_dict:
            return self.wn_dict[wn_name]
        return None

    def get_wordnet_synsets(self,frame: Union[str,None] = None):
        """Given a frame name, returns all the verbal synsets in the frame
           If the frame is None, returns all the verbal synsets in the inventory

        Args:
            frame (Union[str,None], optional): The frame name. Defaults to None.

        Returns:
            List[str]: List of verbal synset names
        """
        if frame == None:
            return [wn_name for wn_name in self.wn_dict.keys()]
        return [*self.va_dict[frame]]

def get_derivetionally_releted_synsets(synset, accepted_pos=["n"]):
    """Gets all the derivationally related forms of a synset, filtered by pos type, just nouns are returned

    Args:
        synset (wn.Synset): The starting synset
        accepted_pos (list(str)): A list of all the accepted pos for the synsets

    Returns:
        list(wn.Synset): the list of derivationally related synsets
    """
    return sorted(list(set([drf.synset() for lemma in synset.lemmas() for drf in lemma.derivationally_related_forms() if drf.synset().pos() in accepted_pos])), key=lambda syn: syn.name())

def get_all_hypernyms(synset):
    """Gets all the hypernyms of the given synset

    Args:
        synset (wn.Synset): The starting synset

    Returns:
        list(wn.Synset): the list of hypernym synsets
    """
    hyp_list = []
    out_hyps = []
    out_hyps.extend(synset.hypernyms())
    while len(out_hyps) > 0:
        hyp = out_hyps.pop(0)
        if hyp not in hyp_list:
            hyp_list.append(hyp)
            out_hyps.extend(hyp.hypernyms())
    return hyp_list

def get_all_hypernyms_with_distances(synset):
    """Gets all the hypernyms of the given synset with their relative distance

    Args:
        synset (wn.Synset): The starting synset

    Returns:
        list((wn.Synset,int)): the list of hypernym synsets with their distance
    """
    hyp_list = []
    out_hyps = []
    out_hyps.extend([[hyp,1] for hyp in synset.hypernyms()])
    while len(out_hyps) > 0:
        current_hyp, current_distance = out_hyps.pop(0)
        if [current_hyp,current_distance] not in hyp_list:
            hyp_list.append([current_hyp,current_distance])
            out_hyps.extend([[hyp,current_distance+1] for hyp in current_hyp.hypernyms()])
    return hyp_list

def get_all_hyponyms(synset):
    """Gets all the hyponyms of the given synset

    Args:
        synset (wn.Synset): The starting synset

    Returns:
        list(wn.Synset): the list of hyponyms synsets
    """
    hypos = set()
    tree = [synset]
    while len(tree)>0:
        synset = tree.pop(0) 
        hypos.add(synset)
        tree.extend(synset.hyponyms())
    return list(hypos)

def get_all_hyponyms_with_distances(synset):
    """Gets all the hyponyms of the given synset with their relative distance

    Args:
        synset (wn.Synset): The starting synset

    Returns:
        list((wn.Synset,int)): the list of hyponyms synsets with their distance
    """
    hypos = set()
    tree = [[synset,0]]
    while len(tree)>0:
        synset, current_distance = tree.pop(0) 
        hypos.add([synset, current_distance])
        tree.extend([[hypo,current_distance+1] for hypo in synset.hyponyms()])
    return list(hypos)

inventory = Inventory(bn2wn_path, bn2va_path, va_info_path)
read_tsv = inventory.read_tsv