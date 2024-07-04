import os, sys, json
import copy
import time, random
from tqdm import tqdm
from dotenv import load_dotenv
import argparse

from data_preprocessing.preprocess_datasets import *
from data_preprocessing.prompt_formats import *
from code_files_nominal_srl.data_preprocessing.llms_utils import *
from data_preprocessing.few_shot_examples import setup_few_shot_examples, prompt_3_outputs, prompt_6_outputs
from data_preprocessing.prompt_validation import *

load_dotenv()
DIR = os.path.realpath(os.path.dirname(__file__))
dataset_name = "semcor" # the name of the dataset and its files

# region Hyperparameters
verbal_to_derivationally_related_forms_path = os.path.join(
    DIR, "../datasets/base_nominal_classification/verbal_to_derivationally_related_forms.tsv" # the file with the verbal -> drf -> nominal mapping
)
full_classified_dataset_path = os.path.join(
    DIR, "../datasets/dataset_nominal_classification/dataset.json" # the file with the entire classified dataset
)
outputs_dir_path = os.path.join(
    DIR, f"../datasets/dataset_nominal_srl/{dataset_name}" # where the generated dataset (and their nominalizations) will be saved
)
os.makedirs(outputs_dir_path, exist_ok=True)

LLM_INPUT_LABEL = "chatgpt_input"
LLM_OUTPUT_LABEL = "chatgpt_output"
LLM_COST_LABEL = "cost"
TRIES_IDX_LABEL = "tries_count"


prompt_data_fn = prompt_format_6 # prompt function to be tested
validate_single_prompt_object_fn = validate_single_prompt_6_object # Validation function for this specific prompt format
verbal_example_sentences = setup_few_shot_examples(prompt_6_outputs)
save_dataset = True # Save it for faster execution next time the script is invoked

OPENAI_KEY = os.getenv("OPENAI_KEY")
FIREWORKS_KEY = os.getenv("FIREWORKS_KEY")
GOOGLE_KEY =  os.getenv("GOOGLE_KEY")


model = "gemini-pro"
# Add code to parse input arguments to select the model
if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description="Nominalizes the sentences of the semcor dataset with the model specified in the command line.")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-pro",
        required=False,
        choices=["gemini-pro", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-instruct", "gpt-4-1106", "gpt-4", "text-davinci-003", "text-davinci-002", "mixtral-8x7b"],
        help="The LLM model to use for the nominalization. Default is 'gemini-pro'."
    )
    args = parser.parse_args()
    model = args.model


ask_model = None
if MODELS[model]["type"] == "gpt_chat":
    ask_model = ask_gpt_chat
    USING_KEY = FIREWORKS_KEY if MODELS[model]["provider"] == "fireworks" else OPENAI_KEY
    client = OpenAI(
        api_key=USING_KEY,
        base_url=MODELS[model]["base_url"] 
    )
elif MODELS[model]["type"] == "gpt_instruct":
    ask_model = ask_gpt_instruct
    USING_KEY = FIREWORKS_KEY if MODELS[model]["provider"] == "fireworks" else OPENAI_KEY
    client = OpenAI(
        api_key=USING_KEY,
        base_url=MODELS[model]["base_url"] 
    )
elif MODELS[model]["type"] == "google_instruct":
    ask_model = ask_google_instruct
    USING_KEY = GOOGLE_KEY
    client = USING_KEY

price_per_prompt_token = MODELS[model]["price_per_prompt_token"]
price_per_response_token = MODELS[model]["price_per_response_token"]
request_cooldown_time_sec = 60/MODELS[model]["requests_per_minute"]

max_samples = None # Max samples per prompt to be tested, set None to infinite
examples_to_use = 10
use_previous_samples_as_examples = False         # Possible values: 
                                                 #                  False -> use just manual examples, the number is set by examples_to_use
                                                 #                  "all" -> use the examples generated during this session, both good and bad (like a real chat), the number is set by examples_to_use
                                                 #                  "good" -> use the examples generated during this session, just the one that pass validation, the number is set by examples_to_use
                                                 #                  "hybrid_good" -> use the examples generated during this session, just the one that pass validation, but also append at the beggining the manual ones, the number is set by examples_to_use
shuffle_examples = True
retries_count = 1 # How many times to retry a failed nominalization
    
use_previous_samples_as_examples = use_previous_samples_as_examples if examples_to_use > 0 else False
examples_to_use = min(examples_to_use, len(verbal_example_sentences)) if not use_previous_samples_as_examples else examples_to_use
retries_count = max(1,retries_count)

system_directives = [
    "You are a linguist that can nominalize sentences. The sentences must be grammatically correct",
    # "Change the sentences and the order of the words as little as possible"
    # "The verb you nominalize must not be present in the sentence anymore"
]
temperature = 0.7 # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.
top_p = 1 # An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.

additional_info = {
    "model": model,
    "prompt": prompt_data_fn.__name__.split("_")[-1],
    "few_shots": examples_to_use,
    "temperature": temperature,
}
if top_p != 1: additional_info["top_p"] = top_p
if len(system_directives) > 0: additional_info["system"] = len(system_directives)
if use_previous_samples_as_examples: additional_info["chat_like"] = use_previous_samples_as_examples
if shuffle_examples: additional_info["shuffle_examples"] = "True"
if retries_count > 1: additional_info["retries_count"] = retries_count

# endregion

# region creating or opening dataset files

# General files nomenclature and structure:
# datasetX_verbal_sentences -> LLM -> datasetX_nominalized_sentences_promptN
result_nominalized_data = {} # datasetX_nominalized_sentences_promptN

# Creating the dicts from verbal -> drf -> nominal mapping file
verbal_to_nominal, nominal_to_verbal = generate_verbal_nominal_mapping(verbal_to_derivationally_related_forms_path)
# verbal_to_nominal, nominal_to_verbal = generate_extended_verbal_nominal_mapping(full_classified_dataset_path)

dataset_verbal_sentences_path = save_dataset_split(None, outputs_dir_path, dataset_name, is_verbal=True, prompt_name=None) # datasetX_verbal_sentences
dataset_nominalized_path = save_dataset_split(None, outputs_dir_path, dataset_name, is_verbal=False, prompt_name=prompt_data_fn.__name__, additional_info=additional_info) # datasetX_nominalized_sentences_promptN

if os.path.exists(dataset_verbal_sentences_path):
    print("Starting dataset already processed and generated, loading it...")
    with open(dataset_verbal_sentences_path, "r") as json_file:
        dataset_verbal_samples_splitted = json.load(json_file)
else:
    print("Creating the processed dataset for the first time...")
    # Generating a common structure for all type of datasets...
    if dataset_name == "semcor":
        formatted_dataset_data = preprocess_semcor()
    filtered_dataset_data = filter_formatted_dataset(formatted_dataset_data, verbal_to_nominal, nominal_to_verbal) # Filter between verbal and nominal sample...
    # Splitting each sample by predicate
    dataset_verbal_samples_splitted = {sample["id"]: sample for element in tqdm(filtered_dataset_data["verbal"].values()) for sample in split_by_predicate(element)}

    print("The number of verbal samples with nominal correspondance is:", len(filtered_dataset_data["verbal"]))
    print("The percentage kept is:", len(filtered_dataset_data["verbal"])/len(formatted_dataset_data))

    # Saving generated samples
    if save_dataset:
        with open(dataset_verbal_sentences_path, "w") as json_file:
            json.dump(dataset_verbal_samples_splitted, json_file, indent=4)

print("The number of verbal samples splitted is:", len(dataset_verbal_samples_splitted))

# Check if output data already exists
if os.path.exists(dataset_nominalized_path):
    print("Opening already existing nominalized sentences...")
    with open(dataset_nominalized_path, 'r') as file:
        diff_data = json.load(file)
        for diff_id, diff_e in diff_data.items():
            if LLM_INPUT_LABEL in diff_e and LLM_OUTPUT_LABEL in diff_e:
                result_nominalized_data[diff_e["id"]] = copy.deepcopy(diff_e)
# endregion

# Variable renaming for convenience
dataset_type_data = dataset_verbal_samples_splitted
result_generated_type_data = result_nominalized_data
result_generated_type_data_path = dataset_nominalized_path

print("Test hyperparameters:")
print(additional_info)

num_processed_samples = 0
num_good_samples = 0
total_cost = 0
dataset_items = list(dataset_type_data.items())
if max_samples != None:
    dataset_items = dataset_items[:max_samples]
progress_bar = tqdm(dataset_items)
progress_bar.set_description(f'Processed samples: {num_processed_samples}')
try:
    for sample_id, e in progress_bar:
        
        num_processed_samples += 1

        e_generated = {} if sample_id not in result_generated_type_data.keys() else result_generated_type_data[sample_id]
        if LLM_INPUT_LABEL in e_generated and LLM_OUTPUT_LABEL in e_generated:
            total_cost += e_generated[LLM_COST_LABEL]
            # good_translation, type_of_result = validate_single_prompt_object_fn(e_generated)
            good_translation = e_generated["validation_result"]
            if good_translation: num_good_samples += 1
            progress_bar.set_description(f'Processed samples: {num_processed_samples}. Good samples: {num_good_samples} ({(num_good_samples/num_processed_samples)*100:.2f}%)')
            continue # element already processed by LLM, skipping it...

        prompt = prompt_data_fn(e) # creating the text prompt input for the LLM model
        
        # Creating the few shot examples using the right prompt
        examples = []
        if not use_previous_samples_as_examples:
            if shuffle_examples:
                random.shuffle(verbal_example_sentences)
            for example_idx in range(examples_to_use):
                example_formatted_element = verbal_example_sentences[example_idx]
                examples.append( [prompt_data_fn(example_formatted_element),example_formatted_element[LLM_OUTPUT_LABEL]] )
        elif use_previous_samples_as_examples == "all":
            previous_examples = list(result_generated_type_data.values())
            previous_examples = previous_examples[max(0,len(previous_examples)-examples_to_use):len(previous_examples)]
            for example_formatted_element in previous_examples:
                examples.append( [prompt_data_fn(example_formatted_element),example_formatted_element[LLM_OUTPUT_LABEL]] )
        elif use_previous_samples_as_examples == "good":
            previous_examples = list(result_generated_type_data.values())
            previous_examples.reverse()
            for example_formatted_element in previous_examples:
                good_translation, type_of_result = validate_single_prompt_object_fn(example_formatted_element)
                if good_translation:
                    examples = [ [prompt_data_fn(example_formatted_element), example_formatted_element[LLM_OUTPUT_LABEL]] ] + examples
                if len(examples) >= examples_to_use: break
        elif use_previous_samples_as_examples == "hybrid_good":
            previous_examples = verbal_example_sentences[:examples_to_use] + list(result_generated_type_data.values())
            previous_examples.reverse()
            for example_formatted_element in previous_examples:
                good_translation, type_of_result = validate_single_prompt_object_fn(example_formatted_element)
                if good_translation:
                    examples = [ [prompt_data_fn(example_formatted_element), example_formatted_element[LLM_OUTPUT_LABEL]] ] + examples
                if len(examples) >= examples_to_use: break
        
        if shuffle_examples:
            random.shuffle(examples)

        e[LLM_INPUT_LABEL] = prompt
        e[LLM_COST_LABEL] = 0
        for tries_idx in range(retries_count):
            start_time = time.time() # Needed because of the strict 3 RPM limit
            response = ask_model(client, prompt, model=model, examples=examples, system_directives=system_directives, temperature=temperature, top_p=top_p)
            e[LLM_OUTPUT_LABEL] = response["response"]
            e[TRIES_IDX_LABEL] = tries_idx + 1
            cost = (response["prompt_tokens"]*price_per_prompt_token) + (response["response_tokens"]*price_per_response_token)
            e[LLM_COST_LABEL] += cost
            total_cost += cost
            good_translation, type_of_result = validate_single_prompt_object_fn(e)
            if good_translation:
                num_good_samples += 1
                break
            else:
                # Every provider sets a request per minute limit
                remaining_time = request_cooldown_time_sec - (time.time() - start_time)
                if remaining_time > 0: time.sleep(remaining_time)
        
        result_generated_type_data[sample_id] = e
        progress_bar.set_description(f'Processed samples: {num_processed_samples}. Good samples: {num_good_samples} ({(num_good_samples/num_processed_samples)*100:.2f}%)')

        if num_processed_samples % 5 == 0: # Write the results every 10 requests not not overload the drive
            with open(result_generated_type_data_path, 'w') as file:
                json.dump(result_generated_type_data, file, indent=4)

        # Every provider sets a request per minute limit
        remaining_time = request_cooldown_time_sec - (time.time() - start_time)
        if remaining_time > 0: time.sleep(remaining_time)
except Exception as e:
    print("Gracefully closing the script...")
    with open(result_generated_type_data_path, 'w') as file:
        json.dump(result_generated_type_data, file, indent=4)
    raise e

print("Processed samples:", num_processed_samples)
print("Total cost: ", total_cost)
print(f"Everything done.")


# validate_prompt_results(result_generated_type_data_path, validate_single_prompt_object_fn=validate_single_prompt_object_fn)
    
    