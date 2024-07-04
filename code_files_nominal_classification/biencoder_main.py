import data_preprocessing.wordnet_utils as wnutils # import here, otherwise LAPACK breaks

import os, sys
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import json
import torch
from tqdm import tqdm
import copy

from pipeline_utils import generate_dataset_type, dicts_equal_except_keys
from models.TransformerBiEncoder import TransformerBiEncoder as ModelClassifier


from datasets.DynamicParametrizedDataModule import DynamicDataModule, DynamicDataset

import random
import itertools
from datetime import datetime

import argparse

# Script hyperparameters

pipeline_phase = "finetune_model"
version_name = "version_0"
use_best_checkpoint = True
checkpoint_path = None

dataset_division_type = "manual_test"

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description="This script manages different phases of the ML pipeline, allowing the user to specify the phase, dataset type, version, and checkpoint handling.")
    parser.add_argument(
        "--pipeline_phase", 
        type=str, 
        default="train", 
        required=True,
        choices=["finetune_dataset", "finetune_model", "train", "valid", "test", "predict_test", "predict", "visualize_biencoder"],
        help="The phase of the pipeline."
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="manual_test",
        required=False,
        choices=["nonamb_test", "manual_test", "manual_in_train_test"],
        help="Defines the type of dataset to be used. Default is 'manual_test', which replicates the results from the original paper."
    )
    parser.add_argument(
        "--version_name",
        type=str,
        default="version_0",
        required=True,
        help="Specifies the version name in the lightning_logs folders, useful for tracking different experiments. Default is 'version_0'."
    )
    parser.add_argument(
        "--use_best_checkpoint",
        type=str,
        default=True,
        required=False,
        help="Determines whether to automatically use the best checkpoint based on highest accuracy. Set to True by default. When True, no need to specify --checkpoint_path."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="Path to a specific checkpoint to load. Default is None, which implies no checkpoint unless overridden by use_best_checkpoint being True."
    )

    args = parser.parse_args()
    pipeline_phase = args.pipeline_phase
    dataset_division_type = args.dataset_type
    version_name = args.version_name
    use_best_checkpoint = args.use_best_checkpoint
    checkpoint_path = args.checkpoint_path

os.environ["TOKENIZERS_PARALLELISM"] = "true"
DIR = os.path.realpath(os.path.dirname(__file__))
SEED = 1749274
pl.seed_everything(SEED)
model_type = ModelClassifier.__name__

# region Hyperparameters tuning (to find the best dataset and model)
dataset_hyperparameters_tuning = {
    "dataset_dir_path" : [os.path.abspath(os.path.join(DIR,"..","datasets/dataset_nominal_classification/"))],
    "train_percentage": [0.8],
    "dev_percentage": [0.1],
    "test_percentage": [0.1],
    "split_verbal": [False],
    "use_synset_name" : [False],
    "use_lemmas" : [True],
    "pair_with_verbal" : [True],
    "max_num_verbal" : [[10, 3, 3], [20, 3, 3], [30, 3, 3], [40, 3, 3]],
    "sentence_separator": [None],
    "binary_dataset" : [True],
    "use_nominal_bn_definitions": [False],
    "use_verbal_bn_definitions": [False],
    "clean_definitions": [True],
    "max_num_samples" : [-1],
    "batch_size" : [32],
    "num_workers": [max(os.cpu_count()-2, 2)],
    "seed": [SEED],
    # "preprocessing_hparams": [None],
}
model_hyperparameters_tuning = {
    "num_classes" : [len(wnutils.inventory.va_dict.keys())],
    # Model hyperparams:
    "transformer_name" : ["sentence-transformers/all-mpnet-base-v2", "roberta-base", "BAAI/bge-m3"],
    "fine_tune_transformer" : [True],
    "mlp_num_hidden" : [None, 512, [512,1024,512]],
    "dropout" : [0.2],
    # Training hyperparams:
    "learning_rate" : [3e-5],
    "adam_eps" : [1e-8],
    "adam_weight_decay" : [0.0],
    "accuracy_k" : [4],
    "seed": [SEED],
}
# endregion

# region Saving model hyperparameters

if pipeline_phase.startswith("finetune"): 
    model_type = f'{pipeline_phase}_{model_type}'

checkpoint_folder = os.path.abspath(os.path.join(DIR,"..",f"checkpoints_nominal_classification/{model_type}"))
checkpoint_version_folder = os.path.join(checkpoint_folder, f"lightning_logs/{version_name}/checkpoints/")

def get_checkpoint_val_accuracy(checkpoint_string):
    try: return float(checkpoint_string.split("val_accuracy=")[1].split(".ckpt")[0])
    except: return 0.0

if use_best_checkpoint and pipeline_phase != "train" and not pipeline_phase.startswith("finetune"):
    checkpoint_files = os.listdir(checkpoint_version_folder)
    best_checkpoints_names = sorted(checkpoint_files, key=get_checkpoint_val_accuracy, reverse=True)
    if len(best_checkpoints_names) > 0:
        checkpoint_path = os.path.join(checkpoint_version_folder, best_checkpoints_names[0])
    else:
        use_best_checkpoint = False # fallback to manual path

print(f"Checkpoint path: {checkpoint_path}")

# endregion

# region Dataset hyperparameters
dataset_basename = "version_"
datasets_folder = os.path.abspath(os.path.join(checkpoint_folder,"datasets/"))
os.makedirs(datasets_folder, exist_ok=True)
os.makedirs(os.path.join(checkpoint_folder,"lightning_logs/"), exist_ok=True)
v_num = len([f for f in os.listdir(checkpoint_folder+"/lightning_logs") if f.startswith("version_")])

dataset_nominal_classification_path = os.path.abspath(os.path.join(DIR,"..","datasets/dataset_nominal_classification/"))
dataset_hyperparameters = {
    "dataset_dir_path" : dataset_nominal_classification_path,
    "train_percentage": 0.8,
    "dev_percentage": 0.1,
    "test_percentage": 0.1,
    "split_verbal": False,
    "use_synset_name" : False,
    "use_lemmas" : True,
    "pair_with_verbal" : True,
    "max_num_verbal" : [30, 3, 3],
    "sentence_separator": None,
    "binary_dataset" : True,
    "use_nominal_bn_definitions": False,
    "use_verbal_bn_definitions": False,
    "clean_definitions": True,
    "max_num_samples" : -1,
    "batch_size" : 32,
    "num_workers": max(os.cpu_count()-2, 2),
    "seed": SEED,
    # "use_verbals_for_empty_frames": [4, False, False],
    # "preprocessing_hparams": dataset_preprocessing_hyperparameters if use_custom_data_construction_fn else None,
    "built_dataset_save_path": datasets_folder+f"/{dataset_basename}{v_num}",
    # "custom_data_construction_fn" : custom_data_construction_fn if use_custom_data_construction_fn else None,
}

generate_dataset_hyperparameters = {
    "dataset_dir_path": dataset_nominal_classification_path,
    "division_type": dataset_division_type,
}
# endregion

# region Model hyperparameters
model_hyperparameters = {
    "num_classes" : len(wnutils.inventory.va_dict.keys()),
    # Model hyperparams:
    "transformer_name" : "sentence-transformers/all-mpnet-base-v2",
    # "transformer_name" : "roberta-base",
    # "transformer_name" : "jinaai/jina-embeddings-v2-base-en",
    "fine_tune_transformer" : True,
    "mlp_num_hidden" : None,
    "dropout" : 0.2,
    # Training hyperparams:
    "learning_rate" : 3e-5,
    "adam_eps" : 1e-8,
    "adam_weight_decay" : 0.0,
    "accuracy_k" : 4,
    "seed": SEED,
}
# endregion

# region Prediction hyperparameters
prediction_hyperparameters = {
    "max_num_nominal_definitions": 1,
    "num_verbal_per_nominal": 10,
    "top_k": 5,
}

# Get the current timestamp
current_timestamp = datetime.now()
# Format the timestamp as YYYY-MM-DD_HH-MM-SS
formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")

prediction_results_save_dir = os.path.join(DIR,"..",f"outputs_nominal_classification/results_{model_type}_{formatted_timestamp}")
grid_search_save_dir = os.path.join(DIR,"..",f"outputs_nominal_classification/", f"finetuning/")
os.makedirs(grid_search_save_dir, exist_ok=True)

# endregion

# region finetuning
if pipeline_phase == "finetune_dataset" or pipeline_phase == "finetune_model":
    # Grid search
    if pipeline_phase == "finetune_dataset":
        hyperparameters_tuning = dataset_hyperparameters_tuning
    elif pipeline_phase == "finetune_model":
        hyperparameters_tuning = model_hyperparameters_tuning

    finetune_name = pipeline_phase.split("_")[1]

    saving_hyperparameters_path = os.path.join(grid_search_save_dir, f'{model_type}_grid_search.json')
    combinations = list(itertools.product(*hyperparameters_tuning.values()))
    keys = list(hyperparameters_tuning.keys())
    finetuning_hyperparameters_grid = [dict(zip(keys, combination)) for combination in combinations]
    
    grid_results = []
    if os.path.exists(saving_hyperparameters_path):
        with open(saving_hyperparameters_path, "r") as f:
            grid_results = json.load(f)

    for finetuning_hyperparameters in finetuning_hyperparameters_grid:
        skip_test = False
        for grid_test in grid_results:
            ignore_keys = ["dataset_dir_path", "best_model_path", "built_dataset_save_path", "custom_data_construction_fn"]
            if dicts_equal_except_keys(grid_test[f"{finetune_name}_hyperparameters"], finetuning_hyperparameters, ignore_keys):
                skip_test = True
                break
        if skip_test:
            print("Skipped test already computed...")
            continue
        
        if pipeline_phase == "finetune_model":
            dataset_hyperparameters_selected = dataset_hyperparameters
            model_hyperparameters_selected = finetuning_hyperparameters
            print(model_hyperparameters_selected)
        elif pipeline_phase == "finetune_dataset":
            dataset_hyperparameters_selected = finetuning_hyperparameters
            model_hyperparameters_selected = model_hyperparameters
            print(dataset_hyperparameters_selected)

        data_module = DynamicDataModule(**dataset_hyperparameters_selected)
        model = ModelClassifier(**model_hyperparameters_selected)

        early_stop_accuracy = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=3, verbose=False, mode="max", stopping_threshold = 1.0, strict=False)
        early_stop_loss = EarlyStopping(monitor="val_loss", min_delta=0.00001, patience=3, verbose=False, mode="min", stopping_threshold = 0.0, strict=False)
        checkpoint = ModelCheckpoint(save_top_k=2, mode="max", monitor="val_accuracy", save_last=True, filename='{epoch}-{val_accuracy:.4f}')

        logger = CSVLogger(checkpoint_folder)

        pl.seed_everything(SEED)
        trainer = pl.Trainer(deterministic=True,
                        accelerator="auto",
                        default_root_dir=checkpoint_folder,
                        callbacks=[
                            early_stop_accuracy, 
                            early_stop_loss, 
                            checkpoint
                        ],
                        min_epochs=2, max_epochs=20, log_every_n_steps=1,
                        logger=logger)
        trainer.fit(model = model, datamodule = data_module)
        
        print(f"Model trained path: {checkpoint.best_model_path}. Score: {checkpoint.best_model_score}\n\n")
        grid_results.append({
            "dataset_hyperparameters": dataset_hyperparameters_selected,
            "model_hyperparameters": model_hyperparameters_selected,
            "best_model_path": checkpoint.best_model_path,
            "best_model_score": model.best_scores
        })
        
        with open(saving_hyperparameters_path, "w") as f:
            json.dump(grid_results,f,indent="\t")
# endregion

# region train
elif pipeline_phase == "train":
    data_module = generate_dataset_type(dataset_hyperparameters, **generate_dataset_hyperparameters)
    model = ModelClassifier(**model_hyperparameters)

    early_stop_accuracy = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=3, verbose=False, mode="max", stopping_threshold = 1.0, strict=False)
    early_stop_loss = EarlyStopping(monitor="val_loss", min_delta=0.00001, patience=3, verbose=False, mode="min", stopping_threshold = 0.0, strict=False)
    checkpoint = ModelCheckpoint(save_top_k=2, mode="max", monitor="val_accuracy", save_last=True, filename='{epoch}-{val_accuracy:.4f}')

    logger = CSVLogger(checkpoint_folder)
    
    trainer = pl.Trainer(deterministic=True,
                    accelerator="auto",
                    default_root_dir=checkpoint_folder,
                    callbacks=[
                        early_stop_accuracy, 
                        early_stop_loss, 
                        checkpoint
                    ],
                    min_epochs=2, max_epochs=20, log_every_n_steps=1,
                    logger=logger)
    trainer.fit(model = model, datamodule = data_module)
    print(f"Best model's path: {checkpoint.best_model_path}. Score: {checkpoint.best_model_score}")
# endregion
# region valid
elif pipeline_phase == "valid":
    data_module = generate_dataset_type(dataset_hyperparameters, **generate_dataset_hyperparameters)
    model = ModelClassifier.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer(deterministic=True,
                    accelerator="auto",
                    default_root_dir=checkpoint_folder,
                    min_epochs=2, max_epochs=20, log_every_n_steps=1, enable_checkpointing=False)
    trainer.validate(model=model, datamodule=data_module, verbose=True)
# endregion
# region test
elif pipeline_phase == "test":
    data_module = generate_dataset_type(dataset_hyperparameters, **generate_dataset_hyperparameters)
    model = ModelClassifier.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer(deterministic=True,
                    accelerator="auto",
                    default_root_dir=checkpoint_folder,
                    min_epochs=2, max_epochs=20, log_every_n_steps=1, enable_checkpointing=False)
    trainer.test(model=model, datamodule=data_module, verbose=True)
# endregion

# region predict (on test)
elif pipeline_phase == "predict_test":
    model = ModelClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Loading the dev dataset (it contains all the verbal synsets and the classified nominal ones)
    dataset_hyperparameters_x = copy.deepcopy(dataset_hyperparameters)
    dataset_hyperparameters_x["split_verbal"] = False
    data_module = generate_dataset_type(dataset_hyperparameters, **generate_dataset_hyperparameters)
    data_module.prepare_data()

    # Creating the verbal dictionary: each frame is a list of verbal definitions
    verbal_dict = {}
    for frame_name, verbal_frame_synsets in data_module.train_dataset.unformatted_data["verbal_definitions"].items():
        for verbal_frame_synset in verbal_frame_synsets.values():
            verbal_synset_name = verbal_frame_synset["synset_name"]
            verbal_sentence = DynamicDataset.clean_sentence(verbal_frame_synset["wn_definition"], clean_params=dataset_hyperparameters["clean_definitions"]) 
            verbal_synset_sentence = DynamicDataset.format_sentence(verbal_frame_synset, verbal_sentence, **dataset_hyperparameters)
            verbal_dict.setdefault(frame_name, []).append(verbal_synset_sentence)

        if len(verbal_dict[frame_name]) < prediction_hyperparameters["num_verbal_per_nominal"]:
            print(f"[WARN] Frame {frame_name} has {len(verbal_dict[frame_name])} verbal samples, but requested {prediction_hyperparameters['num_verbal_per_nominal']}")

    # Open test set split:
    data_module = generate_dataset_type(dataset_hyperparameters, **generate_dataset_hyperparameters)
    data_module.prepare_data()
    annotated_data = data_module.test_dataset.unformatted_data["nominal_definitions"]
    annotated_data = {k:v for frame_name, frame_synsets in annotated_data.items() for k,v in frame_synsets.items()}

    # Creating the data dict that contains the definitions for the nominal synsets:
    for data_key, synset_object in annotated_data.items():
        nominal_synset_name = synset_object["synset_name"]
        target_frames = synset_object["frame_name"]
        sentence = DynamicDataset.clean_sentence(synset_object["wn_definition"], clean_params=dataset_hyperparameters["clean_definitions"]) 
        synset_sentence = DynamicDataset.format_sentence(synset_object, sentence, **dataset_hyperparameters)
        annotated_data[data_key]["definitions"] = [synset_sentence]

    # Obtaining the results from the prediction
    results = []
    no_intersection_synsets = set()
    top_k = prediction_hyperparameters["top_k"]
    pbar = tqdm(annotated_data.items())
    
    for nominal_synset_name, nominal_synset_object in pbar:
        for nominal_definition in nominal_synset_object["definitions"][:prediction_hyperparameters["max_num_nominal_definitions"]]:

            result_object = {
                "synset_name": nominal_synset_name,
                "definition": nominal_definition,
                "target_frame": nominal_synset_object["frame_name"],
                "frame_scores": {},
                "processed_frame_scores": []
            }

            for frame_name, frame_defs_list in verbal_dict.items():

                # Heuristic adopted: sample at most N randomly chosen verbal definitions from each frame
                verbal_definitions_len = min(len(frame_defs_list), prediction_hyperparameters["num_verbal_per_nominal"])
                verbal_definitions = random.sample(frame_defs_list, verbal_definitions_len)
                samples = [[nominal_definition, vdef] for vdef in verbal_definitions]

                with torch.no_grad():
                    logits = torch.sigmoid(model(samples)["logits"]).squeeze()
                result_object["frame_scores"][frame_name] = sorted(logits.tolist(), reverse=True)
                result_object["processed_frame_scores"].append({"frame_name": frame_name, "mean": torch.mean(logits).item()})

            # Heuristic adopted: sort frames by the mean of the logits
            result_object["processed_frame_scores"] = sorted(result_object["processed_frame_scores"], key=lambda x: x["mean"], reverse=True)

            results.append(result_object)
    
    # Saving results variable
    os.makedirs(prediction_results_save_dir, exist_ok=True)
    with open(os.path.join(prediction_results_save_dir, "results_test.json"), 'w') as json_file:
        json.dump(results, json_file)

# elif pipeline_phase == "parse_predict_test":
#     with open(os.path.join(prediction_results_save_dir, "results_test.json"), 'r') as json_file:
#         results = json.load(json_file)

    # Open the TSV file for writing
    top_k = prediction_hyperparameters["top_k"]
    top_k_accuracy = [0 for _ in range(prediction_hyperparameters["top_k"])]
    os.makedirs(prediction_results_save_dir, exist_ok=True)
    with open(os.path.join(prediction_results_save_dir, "results_test.tsv"), 'w') as tsv_file:
        # Write the header row
        top_k = prediction_hyperparameters["top_k"]
        header = "synset_name\tsynset_definition\ttarget_frame\t" + "\t".join([f"frame_{i}" for i in range(top_k)]) # Get the keys from the first dictionary
        tsv_file.write(header + "\n")

        # Write the data rows
        for result_obj in results:
            top_k_results = [r["frame_name"] for r in result_obj["processed_frame_scores"][:top_k]]
            row_values = f"{result_obj['synset_name']}\t{result_obj['definition']}\t{result_obj['target_frame']}\t" + "\t".join(top_k_results)
            tsv_file.write(row_values + "\n")

            for i in range(len(top_k_accuracy)):
                if result_obj['target_frame'] in top_k_results[:i+1]: top_k_accuracy[i] += 1

    for i,v in enumerate(top_k_accuracy):
        print(f"{i+1}-Accuracy", v/len(results))
    
    print("Finished")

# endregion

# region predict
elif pipeline_phase == "predict":

    model = ModelClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Loading the dev dataset (it contains all the verbal synsets and the classified nominal ones)
    with open(os.path.join(dataset_hyperparameters["dataset_dir_path"], "dataset.json"), 'r') as json_file:
        json_data = json.load(json_file)

    # Creating the verbal dictionary: each frame is a list of verbal definitions
    verbal_dict = {}
    for frame_name, verbal_frame_synsets in json_data["verbal_definitions"].items():
        for verbal_frame_synset in verbal_frame_synsets.values():
            verbal_synset_name = verbal_frame_synset["synset_name"]
            verbal_sentence = DynamicDataset.clean_sentence(verbal_frame_synset["wn_definition"], clean_params=dataset_hyperparameters["clean_definitions"]) 
            verbal_synset_sentence = DynamicDataset.format_sentence(verbal_frame_synset, verbal_sentence, **dataset_hyperparameters)
            verbal_dict.setdefault(frame_name, []).append(verbal_synset_sentence)

        if len(verbal_dict[frame_name]) < prediction_hyperparameters["num_verbal_per_nominal"]:
            print(f"[WARN] Frame {frame_name} has {len(verbal_dict[frame_name])} verbal samples, but requested {prediction_hyperparameters['num_verbal_per_nominal']}")

    # Open ambiguous and unclassified files:
    with open(os.path.join(dataset_hyperparameters["dataset_dir_path"], "ambiguous.json"), 'r') as json_file:
        ambiguous_data = json.load(json_file)
    with open(os.path.join(dataset_hyperparameters["dataset_dir_path"], "unclassified.json"), 'r') as json_file:
        unclassified_data = json.load(json_file)

    uncertain_data = {**ambiguous_data, **unclassified_data}

    # Creating the data dict that contains the definitions for the nominal synsets and the target frames (if ambiguous)
    for data_key, synset_object in uncertain_data.items():
        nominal_synset_name = synset_object["synset_name"]
        target_frames = synset_object["frame_name"]
        sentence = DynamicDataset.clean_sentence(synset_object["wn_definition"], clean_params=dataset_hyperparameters["clean_definitions"]) 
        synset_sentence = DynamicDataset.format_sentence(synset_object, sentence, **dataset_hyperparameters)

        uncertain_data[data_key]["definitions"] = [synset_sentence]
    
    # Obtaining the results from the prediction
    results = []
    no_intersection_synsets = set()
    top_k = prediction_hyperparameters["top_k"]
    pbar = tqdm(uncertain_data.items())
    pbar.set_description(f'Ambiguous synsets with no intersection: {len(no_intersection_synsets)}')
    for nominal_synset_name, nominal_synset_object in pbar:
        for nominal_definition in nominal_synset_object["definitions"][:prediction_hyperparameters["max_num_nominal_definitions"]]:

            result_object = {
                "synset_name": nominal_synset_name,
                "definition": nominal_definition,
                "ambiguous_frames": nominal_synset_object["frame_name"],
                "frame_scores": {},
                "processed_frame_scores": []
            }

            for frame_name, frame_defs_list in verbal_dict.items():

                # Heuristic adopted: sample at most N randomly chosen verbal definitions from each frame
                verbal_definitions_len = min(len(frame_defs_list), prediction_hyperparameters["num_verbal_per_nominal"])
                verbal_definitions = random.sample(frame_defs_list, verbal_definitions_len)
                samples = [[nominal_definition, vdef] for vdef in verbal_definitions]

                with torch.no_grad():
                    logits = torch.sigmoid(model(samples)["logits"]).squeeze()
                result_object["frame_scores"][frame_name] = sorted(logits.tolist(), reverse=True)
                result_object["processed_frame_scores"].append({"frame_name": frame_name, "mean": torch.mean(logits).item()})

            # Heuristic adopted: sort frames by the mean of the logits
            result_object["processed_frame_scores"] = sorted(result_object["processed_frame_scores"], key=lambda x: x["mean"], reverse=True)

            top_k_results = [r["frame_name"] for r in result_object["processed_frame_scores"][:top_k]]
            if result_object["ambiguous_frames"] is not None and len( list(set(top_k_results) & set(result_object["ambiguous_frames"])) ) == 0:
                no_intersection_synsets.add(nominal_synset_name)
                pbar.set_description(f'Ambiguous synsets with no intersection: {len(no_intersection_synsets)}')
            results.append(result_object)
    
    # Saving results variable
    results = {"data":results, "no_intersection_synsets":list(no_intersection_synsets)}
    os.makedirs(prediction_results_save_dir, exist_ok=True)
    with open(os.path.join(prediction_results_save_dir, "results.json"), 'w') as json_file:
        json.dump(results, json_file)

# # endregion

# # region parse prediction (for the expert)
# elif pipeline_phase == "parse_predict":

#     with open(os.path.join(prediction_results_save_dir, "results.json"), 'r') as json_file:
#         results = json.load(json_file)

    top_k = prediction_hyperparameters["top_k"]
    # Open the TSV file for writing
    os.makedirs(prediction_results_save_dir, exist_ok=True)
    with open(os.path.join(prediction_results_save_dir, "results.tsv"), 'w') as tsv_file:
        # Write the header row
        header = "synset_name\tsynset_definition\t" + "\t".join([f"frame_{i}" for i in range(top_k)]) # Get the keys from the first dictionary
        tsv_file.write(header + "\n")

        # Write the data rows
        for result_obj in tqdm(results["data"]):
            top_k_results = [r["frame_name"] for r in result_obj["processed_frame_scores"][:top_k]]
            row_values = f"{result_obj['synset_name']}\t{result_obj['definition']}\t" + "\t".join(top_k_results)
            tsv_file.write(row_values + "\n")
    
    print("Ambiguous synsets:", len(
        set([e["synset_name"] for e in results["data"] if e["ambiguous_frames"] is not None and len(e["ambiguous_frames"]) > 1])
    ))
    print("Ambiguous synsets that have no intersection with top-k predictions:", len(results["no_intersection_synsets"]))
    
    print("Finished")


elif pipeline_phase == "visualize_biencoder":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sentence_transformers import SentenceTransformer

    model = ModelClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    num_frames_to_visualize =  5
    max_num_synsets_per_frame_to_visualize = 10

    sentences = []
    embeddings = []
    synsets_names = []
    frames_names = []

    # data_module = generate_dataset_type(dataset_hyperparameters, **generate_dataset_hyperparameters)
    # data_module.prepare_data()
    # split_to_consider = data_module.train_dataset.unformatted_data["nominal_definitions"]
    with open(os.path.join(dataset_hyperparameters["dataset_dir_path"], "dataset.json"), 'r') as f:
        split_to_consider = json.load(f)["nominal_definitions"]

    most_populated_frames = dict(sorted(
        split_to_consider.items(), 
        key=lambda item: len(item[1]),
        reverse=True
    )[:num_frames_to_visualize])

    frames_count = {f:0 for f in most_populated_frames.keys()}
    frames_colors = {f:i for i, f in enumerate(most_populated_frames.keys())}

    for frame_name, frame_synsets in most_populated_frames.items():
        for frame_synset in frame_synsets.values():
            if frames_count[frame_name] >= max_num_synsets_per_frame_to_visualize: continue
            synset_name = frame_synset["synset_name"]
            synset_definition = DynamicDataset.clean_sentence(frame_synset["wn_definition"], clean_params=dataset_hyperparameters["clean_definitions"])
            synset_definition = DynamicDataset.format_sentence(frame_synset, synset_definition, **dataset_hyperparameters)
            
            sentences.append(synset_definition)
            synsets_names.append(synset_name)
            frames_names.append(frame_name)
            
            with torch.no_grad():
                output = model.forward_embedding(synset_definition)["sentence_embeddings"].detach().cpu().tolist() # -> batch size of 2: (batch, word_emb_dim)
                embeddings += output

            frames_count[frame_name] += 1

    print(frames_count)

    n_components = 2

    # Apply PCA/TSNE for dimensionality reduction
    # dim_red = PCA(n_components=n_components)
    dim_red = TSNE(n_components=n_components)

    reduced_embeddings = dim_red.fit_transform(np.array(embeddings, dtype=np.float32))

    if n_components == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], c=[frames_colors[f] for f in frames_names])
        for i,_ in enumerate(reduced_embeddings): plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], synsets_names[i])
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Visualization of Sentence Embeddings")
        plt.show()
    elif n_components == 3:
        fig = plt.figure(figsize=(14,9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], reduced_embeddings[:,2], c=frames_names)
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.title("Visualization of Sentence Embeddings")
        plt.show()

# endregion