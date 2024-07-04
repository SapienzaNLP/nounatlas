import sys, os, json, csv, requests

from spacy import displacy

from copy import deepcopy

from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# region functions

class SentenceSimilarityModel():
    def __init__(self, model_name = 'sentence-transformers/all-mpnet-base-v2', device="cpu") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)

    def mean_pooling(self, token_embeddings, encoded_input):
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_no_pooling(self, sentences, is_split_into_words = True, **kwargs):
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, 
            is_split_into_words=is_split_into_words,
            return_tensors='pt',
            **kwargs
        )
        encoded_input.to(self.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        token_embeddings = model_output[0]
        # attention_mask = encoded_input['attention_mask']
        return token_embeddings, encoded_input
    
    def pooling(self, token_embeddings, encoded_input):
        # Perform pooling
        sentence_embeddings = self.mean_pooling(token_embeddings, encoded_input)
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
    def encode(self, sentences, is_split_into_words = True, **kwargs):
        token_embeddings, attention_mask = self.encode_no_pooling(sentences, is_split_into_words = is_split_into_words, **kwargs)
        return self.pooling(token_embeddings, attention_mask)
    
    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids)
    
    def compute_cosine_similarity(self, source_embedding, compare_embedding):
        similarities = util.pytorch_cos_sim(source_embedding, compare_embedding).diag()
        similarities = similarities.tolist()
        if len(similarities) == 1:
            return similarities[0]
        return similarities 
    
    def cut_token_embeddings(self, token_embeddings, encoded_input, span):
        encoded_input_cutted = encoded_input.copy()

        span_start, span_end = span
        word_ids = encoded_input_cutted.word_ids(batch_index=0)
        span_start = word_ids.index(span_start) if span_start in word_ids else word_ids.index(span_start + 1)
        # span_end = max((i for i, x in enumerate(word_ids) if x == span_end), default=-1) + 1
        if span_end+1 > max([x for x in word_ids if x != None]):
            span_end = len(word_ids) - 1
        else:
            if span_end+1 in word_ids:
                span_end_t = word_ids.index(span_end+1)
                if span_end in word_ids:
                    span_end = span_end_t - len([i for i in word_ids if i == word_ids[span_end_t-1]])
                else:
                    span_end = span_end_t
            else: 
                span_end = word_ids.index(span_end)

        encoded_input_cutted["attention_mask"] = encoded_input["attention_mask"][ : , span_start : span_end ]
        encoded_input_cutted["input_ids"] = encoded_input["input_ids"][ : , span_start : span_end ]
        return token_embeddings[ : , span_start : span_end , : ], encoded_input_cutted
    

def invero_request(text = "Marco is eating an apple.", lang = "EN", invero_url = 'http://127.0.0.1:3001/api/model'):
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

def parse_invero_result(invero_res_obj):
    res = []
    for e in invero_res_obj:
        words = [t["rawText"] for t in e["tokens"]]
        frame_roles = {}
        for ann in e["annotations"]:
            index = ann["tokenIndex"] # index of the predicate
            va_frame = ann["verbatlas"]["frameName"]
            va_roles = ann["verbatlas"]["roles"]
            roles = []
            for role in va_roles:
                roles.append( [role["span"][0], role["span"][1], role["role"].lower()] )
                
            frame_roles[index] = {"predicate": va_frame, "roles": roles}
        res.append({
            "words": words,
            "predictions": frame_roles
        })
    return res

def get_possible_role_names(va_frame_pas_path):
    roles = []
    with open(os.path.join(va_frame_pas_path), mode="r", newline="") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter="\t") 
        for row in tsv_reader:
            row_roles = row[1:]
            roles += [r.lower() for r in row_roles]
    return sorted(list(set(roles)))

def format_multi_srl_result(res, predicate_idx = None):
    results = []
    for p_idx, p_values in res["predictions"].items():
        if predicate_idx is not None and str(predicate_idx) != str(p_idx): continue
        words = deepcopy(res["words"])
        words[int(p_idx)] = (words[int(p_idx)], p_values["predicate"])
        
        for role_infos in sorted(p_values["roles"], key=lambda x: x[0], reverse=True):
            role_span, role_name = (role_infos[0], role_infos[1]), role_infos[2]
            words[role_span[0]:role_span[1]] = [(words[role_span[0]:role_span[1]], role_name)]

        results.append(words)
    return results

def check_no_false_between_trues(bool_list):
    state = 0
    for b in bool_list:
        if state == 0 and not b: continue
        elif state == 0 and b: state = 1
        elif state == 1 and not b: state = 2
        elif state == 2 and b: return False
    return True

def spacy_get_span(words, span_start, span_end):
    start = len(" ".join(words[:span_start+1])) - len(words[span_start])
    end = start + len(" ".join(words[span_start:span_end]))
    return start, end

def spacy_compute_element(res_type, predicate_va_idx, title_type = "Verbal", color_predicate = "#ff9200", color_role = "#ff9200"):
    colors = {}
    type_element = {
        "text": " ".join(res_type["words"]),
        "ents": [],
        "title": title_type,
    }
    span_start, span_end = spacy_get_span(res_type["words"], int(predicate_va_idx), int(predicate_va_idx)+1)
    try:
        predictions = res_type["predictions"][str(predicate_va_idx)]
    except:
        predictions = res_type["predictions"][int(predicate_va_idx)]
    predicate = predictions["predicate"]
    type_element["ents"].append({"start": span_start, "end": span_end, "label": predicate})
    colors[predicate] = color_predicate
    for role in predictions["roles"]:
        span_start, span_end = spacy_get_span(res_type["words"], role[0], role[1])
        type_element["ents"].append({"start": span_start, "end": span_end, "label": role[2]})
        colors[role[2]] = color_role

    return type_element, colors

def spacy_visualize_sentences(mapped_infos: dict, debug_infos: dict = dict(), title: str = "", jupyter = False):
    html_content = ""
    color_predicate = '#ff9200'
    color_role = '#8bc34a'

    if title != "": html_content += f'<h3 style="text-align: center">{title}</h3>'
    if len(debug_infos) > 0:
        for key, val in debug_infos.items():
            html_content += f'<p>{key}: {val}</p>'
        html_content += '<hr>'

    for element_id, element_res in mapped_infos.items():
        dic_ents = []
        res_verbal, res_nominal, predicate_va_idx_verbal, predicate_va_idx_nominal = element_res["verbal"], element_res["nominal"], element_res["verbal_predicate_target_idx"], element_res["nominal_predicate_target_idx"]
        
        verbal_element, verbal_colors = spacy_compute_element(res_verbal, predicate_va_idx_verbal, "Verbal", color_predicate=color_predicate, color_role=color_role)
        dic_ents.append(verbal_element)

        nominal_element, nominal_colors = spacy_compute_element(res_nominal, predicate_va_idx_nominal, "Nominal", color_predicate=color_predicate, color_role=color_role)
        dic_ents.append(nominal_element)

        colors = {**verbal_colors, **nominal_colors}

        rendered_element = displacy.render(dic_ents, manual=True, style="ent", jupyter=jupyter, options={"colors": colors})
        if not jupyter:
            html_content += f'<h3 style="text-align: center">{element_id}</h3>'
            html_content += rendered_element
            html_content += '<hr>'
    return html_content

# Additional roles problem (he/she -> his/her):
additional_roles_list = [
    ["i", "mine", "my"],
    ["you", "yours"],
    ["he", "his"],
    ["she", "her", "hers"],
    ["it", "its"],
    ["they", "them", "their", "theirs"],
]
def get_additional_roles(role_inventory):
    results = []
    for r in role_inventory:
        for l in additional_roles_list:
            if r.lower() in l:
                results += [e for e in l if e not in role_inventory]
    return results

def merge_debug_infos(dicts):
    result = dicts[0]
    for idx in range(1, len(dicts)):
        for k, v in dicts[idx].items():
            if k not in result.keys(): result[k] = v
            else: result[k] += v
    return dict(result)

def split_dict(input_dict, num_splits):
    dict_splits = [{} for _ in range(num_splits)]

    keys = list(input_dict.keys())
    num_keys = len(keys)

    for i, key in enumerate(keys):
        split_index = int(i / (num_keys/num_splits))
        dict_splits[split_index][key] = input_dict[key]

    return dict_splits

# endregion