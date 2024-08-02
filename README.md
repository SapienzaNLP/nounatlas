<div align="center">
  <img src="https://github.com/SapienzaNLP/nounatlas/blob/main/logo.png" height="200", width="700">
</div>

# NounAtlas: Filling the Gap in Nominal Semantic Role Labeling

[![Conference](https://img.shields.io/badge/ACL-2024-4b44ce
)](https://2024.aclweb.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://2024.aclweb.org/program/main_conference_papers/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<div align='center'>
  <img src="https://github.com/Babelscape/FENICE/blob/master/Sapienza_Babelscape.png" height="70">
</div>


This repository contains the code for the paper **[NounAtlas: Filling the Gap in Nominal Semantic Role Labeling](TODO)**. 

## Website
NounAtlas [website](https://frameatlas.org) is out!
Check out the latest version of our resource in the [Download](https://frameatlas.org/download) section.

## Using the NounAtlas Model

Our model for joint nominal and verbal SRL will be usable soon through the [InVeRo API](https://nlp.uniroma1.it/invero/).

## Dataset for Nominal SRL
Our dataset for nominal SRL is available on Hugging Face 🤗

[https://huggingface.co/datasets/sapienzanlp/nounatlas_srl_corpus](https://huggingface.co/datasets/sapienzanlp/nounatlas_srl_corpus)

## Reproducing the paper results

This section provides instructions to reproduce the experiments and results described in the paper. Each section corresponds to a specific paragraph in the paper, refer to it for further details.


### Step 0:  Setting Up the Environment

1. **Create a conda environment** with Python 3.9 or higher.
2. **Activate the environment**.
3. **Navigate to the root of the project**.
4. **Install required packages**: Open a terminal and run the following command:

```bash
pip install -r requirements.txt
```

### Phase 1: Creating a Large-Scale Nominal Predicate Inventory

**WordNet-based synset-to-frame mapping**

1. **Generate link files**: Run the following command to generate the "Unambiguous", "Manually-curated", and "Non-existing" link files (they're already present in the repo):

```bash
python code_files_nominal_classification/data_preprocessing/build_datasets.py
```

**Ranking frames for unlinked synsets**

1. **Train the Cross-Encoder model**: Train the Cross-Encoder model for the verbal to nominal definition grouping task (Section 3.1.3) using the following command:

```bash
python code_files_nominal_classification/crossencoder_main.py --pipeline_phase train
```

2. **Evaluate the model**: Evaluate the model's performance:

```bash
python code_files_nominal_classification/crossencoder_main.py --pipeline_phase test --version_name version_0
```

Replace "version_0" with the version you want to evaluate in the *_[checkpoints_nominal_classification/CrossEncoderClassifier/lightning_logs](/checkpoints_nominal_classification/CrossEncoderClassifier/lightning_logs)_* folder.

3. **Evaluate top-k accuracy**: Evaluate the model's performance on the top-k frame predictions task (Section 3.1.3):

```bash
python code_files_nominal_classification/crossencoder_main.py --pipeline_phase predict_test --version_name version_0
```

**Manual mapping of unlinked synsets**

1. **Generate predictions for expert annotation**: Generate the TSV file with top-5 predictions for manual annotation by the experts:

```bash
python code_files_nominal_classification/crossencoder_main.py --pipeline_phase predict --version_name version_0
```
The file is saved in the folder */outputs_nominal_classification/results_CrossEncoderClassifier_{current_timestamp}/results.tsv*

2. **Generate files for expert annotation**: Create 2 files for the annotator in the folder *[/outputs_nominal_classification/results_linguist/](/outputs_nominal_classification/results_linguist/)*:
    - *[results_for_expert.xlsx](/outputs_nominal_classification/results_linguist/results_for_expert.xlsx)* containing for each predicate, the definition and the top-5 predictions in a human-readable format
    - *[frames_infos.xlsx](/outputs_nominal_classification/results_linguist/frames_infos.xlsx)* containing additional information to aid the annotation process

The command requires the path of the top-5 prediction tsv file generated in the previous step:

```bash
python code_files_nominal_classification/manual_classification/generate_file_for_linguist.py --pipeline_phase generate --model_results_path <path_to_top5_predictions_tsv_file>
```

Alternatively, to use our results, don't pass the parameter *--model_results_path*:

```bash
python code_files_nominal_classification/manual_classification/generate_file_for_linguist.py --pipeline_phase generate
```
**Important:** Using this command will overwrite the repo's *"results_for_expert.xlsx"* file, that contains the annotations provided by our experts.

3. **Parse manually annotated results**: Parse the annotations from the *"results_for_expert.xlsx"* file:

```bash
python code_files_nominal_classification/manual_classification/generate_file_for_linguist.py --pipeline_phase parse
```

### Phase 2: Creating a Nominal SRL Dataset

**Predicate Nominalization**

1. **Generate data with LLM**: To generate the nominalized sentences, you can select an LLM from one of the following three providers:
    - OpenAI (set the OPENAI_KEY environment variable in the .env file)
    - Google AI Studio (set the GOOGLE_KEY environment variable in the .env file)
    - Fireworks AI (set the FIREWORKS_KEY environment variable in the .env file)
    Google's Gemini-Pro is the model used in the paper and selected by default.

```bash
python code_files_nominal_srl/llm_nominalization.py --model gemini-pro
```

Alternatively, you can skip this step by downloading our already processed data: *[semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True.zip](/datasets/dataset_nominal_srl/semcor/semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True.zip)*

**Verbal-to-Nominal Role Propagation**

1. **Run the mapping scripts**: Execute the following commands to perform verbal-to-nominal role propagation (Section 4.3):

```bash
python code_files_nominal_srl/data_mapping/sentence_mapping_rule.py
python code_files_nominal_srl/data_mapping/sentence_mapping_neural.py
```

or you can skip this part by extracting our already processed files:
&nbsp;  *[semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True.rar](/outputs_nominal_srl/mapped_infos/semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True.rar)*
&nbsp;  *[semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True_NN.rar](/outputs_nominal_srl/mapped_infos/semcor_nominalized_sentences_prompt_format_6@model=gemini-pro@prompt=6@few_shots=10@temperature=0.7@system=1@shuffle_examples=True_NN.rar)*

which are located in: *[outputs_nominal_srl/mapped_infos/](/outputs_nominal_srl/mapped_infos/)*

2. **Create the final SRL dataset**: Generate the dataset to train an SRL model using the following command: 

```bash
python code_files_nominal_srl/datasets_smart_splitting.py
```

or you can skip this step and find the final dataset in: *[outputs_nominal_srl/dataset/semcorgemini_2_nn/](/outputs_nominal_srl/dataset/semcorgemini_2_nn/)*

### Phase 3: Train the SRL model on the generated dataset

We employed a pre-trained "roberta-base" model, finetuned with the training pipeline from the [multi-srl](https://github.com/SapienzaNLP/multi-srl) repository. Follow its instructions to train and evaluate the model. The process is the same as reported for the "Ontonotes" dataset, as our dataset in *[outputs_nominal_srl/dataset/semcorgemini_2_nn/](/outputs_nominal_srl/dataset/semcorgemini_2_nn/)* follows the same format.
To simplify reproducing the results in the paper, we provide various configuration files (in *.yaml format) within the *[resources/roberta_custom_config](/resources/roberta_custom_config)* directory. You can use these files to obtain the same metrics presented in the paper.

---

## License

This work is under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
