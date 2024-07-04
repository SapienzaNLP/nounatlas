import os, sys, json
import re
from data_mapping.sentence_mapping_common import spacy_visualize_sentences
from data_mapping.format_to_dataset import format_to_dataset
import argparse

DIR = os.path.realpath(os.path.dirname(__file__))

mapped_infos_dataset_name = "semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True_NN.json"
formatted_datasets_dir_name = f"semcorgemini_2_nn"
split_type = "test"

dataset_dir = os.path.join(DIR, '../outputs_nominal_srl/dataset/')
mapped_infos_dir = os.path.join(DIR, '../outputs_nominal_srl/mapped_infos/')
formatted_datasets_dir_path = os.path.join(dataset_dir, formatted_datasets_dir_name)
out_txt_path = os.path.join(DIR, "../outputs_nominal_srl/manual_annotation")

OPEN_ROLE = "{"
CLOSE_ROLE = "}"
ROLE_SEP = "|"
FRAME_SEP = "||"
file_subnames = ["data"]
combination = [False, False, True, True]

def generate_manual_sample(sample, predicate_target_idx):
    predicate_target_idx = str(predicate_target_idx)

    result = []
    words = sample["words"]
    predictions = sample["predictions"][predicate_target_idx]
    predicate_frame = sample["predictions"][predicate_target_idx]["predicate"]
    roles = predictions["roles"]
    roles = sorted(roles, key=lambda x: x[0])
    current_role = 0

    idx = 0
    while idx < len(words):
        if current_role < len(roles) and roles[current_role][0] == idx:
            result += [OPEN_ROLE, words[idx]]
            idx += 1
            while idx < roles[current_role][1]:
                result += [words[idx]]
                idx += 1
            result += [ROLE_SEP, roles[current_role][2].upper(), CLOSE_ROLE]
            current_role += 1
        elif idx == int(predicate_target_idx):
            result += [OPEN_ROLE,  f"**{words[idx]}**", FRAME_SEP, predicate_frame, CLOSE_ROLE]
            idx += 1
        else:
            result += [words[idx]]
            idx += 1

    return result

def parse_manual_sample(text_str):
    words = []
    roles = []
    predicate_frame_name = None
    predicate_frame_idx = None

    text_list = re.sub(r'\s+', ' ', text_str.strip()).split(" ")
    idx = 0
    while idx < len(text_list):
        if text_list[idx] == OPEN_ROLE:
            idx += 1
            start_idx = len(words)
            while text_list[idx] != ROLE_SEP and text_list[idx] != FRAME_SEP:
                words.append(text_list[idx])
                idx += 1
            if text_list[idx] == ROLE_SEP or text_list[idx] == FRAME_SEP:
                end_idx = len(words)
                if text_list[idx] == ROLE_SEP:
                    roles.append([start_idx, end_idx, text_list[idx+1].lower()])
                else:
                    predicate_frame_name = text_list[idx+1]
                    predicate_frame_idx = start_idx
                idx += 3
        else:
            words.append(text_list[idx])
            idx += 1

    words[predicate_frame_idx] = words[predicate_frame_idx][2:-2]

    result = {
        "words": words,
        "predictions": {
            predicate_frame_idx: {
                "predicate": predicate_frame_name,
                "roles": roles
            }
        }
    }

    return result, predicate_frame_idx

def create_manual_samples():
    with open(os.path.join(formatted_datasets_dir_path, split_type + ".json"), "r") as f:
        split_data = json.load(f)

    with open(os.path.join(mapped_infos_dir, mapped_infos_dataset_name), "r") as f:
        mapped_infos = json.load(f)

    for file_i, file_name in enumerate(file_subnames):

        file_lines = []

        for k in list(split_data.keys())[700*(file_i):700*(file_i+1)]:
            if k not in mapped_infos.keys():
                print("ERROR: No corresponding id. Did you check mapped_infos_dataset_name and formatted_datasets_dir_name ?")
                return

            verbal_sample = mapped_infos[k]["verbal"]
            verbal_predicate_target_idx = mapped_infos[k]["verbal_predicate_target_idx"]
            verbal_manual = generate_manual_sample(verbal_sample, verbal_predicate_target_idx)
            verbal_manual_str = " ".join(verbal_manual)

            nominal_sample = mapped_infos[k]["nominal"]
            nominal_predicate_target_idx = mapped_infos[k]["nominal_predicate_target_idx"]
            nominal_manual = generate_manual_sample(nominal_sample, nominal_predicate_target_idx)
            nominal_manual_str = " ".join(nominal_manual)

            file_lines += [k, "\n\n", verbal_manual_str, "\n\n", nominal_manual_str, "\n\n\n\n"]

        with open(os.path.join(out_txt_path, split_type + f"_toannotate_{file_name}.txt"), "w") as f:
            for item in file_lines:
                f.write(item)

    print("All done.")

def parse_manual_samples(file_name):
    with open(os.path.join(out_txt_path, file_name), "r") as f:
        file_lines = [line.strip() for line in f.readlines()]

    with open(os.path.join(mapped_infos_dir, mapped_infos_dataset_name), "r") as f:
        mapped_infos = json.load(f)

    result = {}

    unchanged_sentences = 0
    total_sentences = 0

    total_frames = 0
    unchanged_frames = 0

    total_roles = 0
    unchanged_roles = 0
    unchanged_roles_spans = 0

    for i in range(0, len(file_lines), 8):
        sample_id = file_lines[i].strip()
        verbal_manual_str = file_lines[i+2].strip()
        nominal_manual_str = file_lines[i+4].strip()

        if len(sample_id.split(" ")) > 1:
            print("ERROR in formatting?")

        # print(f"Parsing: {sample_id}")

        verbal_sample, verbal_predicate_target_idx = parse_manual_sample(verbal_manual_str)
        nominal_sample, nominal_predicate_target_idx = parse_manual_sample(nominal_manual_str)

        current_predicate_frame = nominal_sample["predictions"][nominal_predicate_target_idx]["predicate"]
        current_roles = nominal_sample["predictions"][nominal_predicate_target_idx]["roles"]

        previous_nominal_predicate_target_idx = mapped_infos[sample_id]["nominal_predicate_target_idx"]
        previous_predicate_frame = mapped_infos[sample_id]["nominal"]["predictions"][str(previous_nominal_predicate_target_idx)]["predicate"]
        previous_roles = mapped_infos[sample_id]["nominal"]["predictions"][str(previous_nominal_predicate_target_idx)]["roles"]

        if mapped_infos[sample_id]["nominal"]["words"] == nominal_sample["words"]:
            unchanged_sentences += 1
        total_sentences += 1

        if current_predicate_frame == previous_predicate_frame and int(nominal_predicate_target_idx) == int(previous_nominal_predicate_target_idx):
            unchanged_frames += 1
            
        for prev_role in previous_roles:
            is_unchanged_role = len([
                curr_role for curr_role in current_roles
                if curr_role[2] == prev_role[2]
            ])
            is_unchanged_role_and_span = len([
                curr_role for curr_role in current_roles
                if curr_role[2] == prev_role[2] and
                curr_role[:2] == prev_role[:2]
            ])
            if is_unchanged_role_and_span > 0:
                unchanged_roles_spans += 1
                unchanged_roles += 1
            elif is_unchanged_role > 0:
                unchanged_roles += 1

            total_roles += 1

        total_frames += 1
        
        result[sample_id] = {
            "verbal": verbal_sample, "nominal": nominal_sample,
            "verbal_predicate_target_idx": verbal_predicate_target_idx,
            "nominal_predicate_target_idx": nominal_predicate_target_idx,
        }

    print(f"Unchanged sentences: {unchanged_sentences}/{total_sentences}")
    print(f"Unchanged frames: {unchanged_frames}/{total_frames}")
    print(f"Unchanged roles: {unchanged_roles}/{total_roles}")
    print(f"Unchanged roles spans: {unchanged_roles_spans}/{total_roles}")

    return result, {
        "unchanged_sentences": unchanged_sentences, 
        "total_sentences": total_sentences, 
        "unchanged_frames": unchanged_frames, 
        "total_frames": total_frames, 
        "unchanged_roles": unchanged_roles, 
        "unchanged_roles_spans": unchanged_roles_spans, 
        "total_roles": total_roles
    }

def sum_dictionaries(dict1, dict2):
    sum_dict = {}
    for key in set(dict1.keys()) | set(dict2.keys()):
        sum_dict[key] = dict1.get(key, 0) + dict2.get(key, 0)
    return sum_dict
    

pipeline_phase = "generate"
if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description="This script manages different phases of the semi-automatic annotation done by the linguist (see Section 3.1.4 of the original paper).")
    parser.add_argument(
        "--pipeline_phase",
        type=str,
        default="generate",
        required=True,
        choices=['generate','parse'],
        help="generate -> creates the file in human-readable format to be manually edited ; parse -> converts the manually-edited file in json format"
    )

    args = parser.parse_args()
    pipeline_phase = args.pipeline_phase


if pipeline_phase == "generate":
    create_manual_samples()

elif pipeline_phase == "parse":
    final_res = {}

    final_metrics = {}

    for file_i, file_name in enumerate(file_subnames):
        file_path_current = os.path.join(out_txt_path, split_type + f"_toannotate_{file_name}_result.txt")
        if not os.path.exists( file_path_current ):
            print("The file", file_name, "does not exist")
            continue
        print(f"processing file name", file_name)

        res, current_file_metrics = parse_manual_samples(split_type + f"_toannotate_{file_name}_result.txt")
        final_metrics = sum_dictionaries(current_file_metrics, final_metrics)

        final_res.update(res)
        res_html = spacy_visualize_sentences(res, title=file_name)
        with open(os.path.join(formatted_datasets_dir_path, file_name + ".html"), "w") as f:
            f.write(res_html)

    print(" ########### Final metrics: ##############")
    print(f"Unchanged sentences: {final_metrics['unchanged_sentences']}/{final_metrics['total_sentences']}, {(100*final_metrics['unchanged_sentences']/final_metrics['total_sentences']):.3f}%")
    print(f"Unchanged frames: {final_metrics['unchanged_frames']}/{final_metrics['total_frames']}, {(100*final_metrics['unchanged_frames']/final_metrics['total_frames']):.3f}%")
    print(f"Unchanged roles: {final_metrics['unchanged_roles']}/{final_metrics['total_roles']}, {(100*final_metrics['unchanged_roles']/final_metrics['total_roles']):.3f}%")
    print(f"Unchanged roles spans: {final_metrics['unchanged_roles_spans']}/{final_metrics['total_roles']}, {(100*final_metrics['unchanged_roles_spans']/final_metrics['total_roles']):.3f}%")

    save_result = os.path.join(formatted_datasets_dir_path, "mapped_infos_manual.json")

    with open(save_result, "w") as f:
        print("Len dataset:", len(final_res))
        json.dump(final_res, f)

    format_to_dataset(
        save_result, formatted_datasets_dir_path, 
        combination[0], combination[1], combination[2], combination[3],
        datset_name="dataset_manual")