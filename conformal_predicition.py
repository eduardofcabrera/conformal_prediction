import yaml
from tqdm import tqdm
import torch
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from prompt_templates import *

def conformal_predicition(dataset, device, token_map, letter_to_index_map, model, tokenizer, prompt_template_boolean, conformal_range):
    results = []
    for i in tqdm(range(conformal_range[0], conformal_range[1])):
        question = dataset[0][i]
        answer = dataset[1][i]
        answer_index  = letter_to_index_map[answer]
        prompt = prompt_template_boolean.format(question_text=question)
        
        prompt_token = tokenizer.encode(prompt, return_tensors="pt").to(device)
        model_return = model(prompt_token, return_dict=True)
        logits = model_return.logits[0, -1, :]
        probs = softmax(logits, dim=0)
        probs = torch.tensor([probs[token_map[letter]].item() for letter in "AB"])
        probs = probs / probs.sum()
        results.append(1 - probs[answer_index].item())
    df = pd.DataFrame(results, columns=["conformal_scores"])
    df.to_csv("conformal_scores.csv", index=False)
    
def inference_cot(dataset, device, token_map, letter_to_index_map, model, tokenizer, prompt_template_boolean, cot_template, val_range):
    results = []
    is_probs = True
    token_map = {"True": tokenizer.encode("True")[-1], "False": tokenizer.encode("False")[-1]}
    for i in tqdm(range(val_range[0], val_range[1])):
        question = dataset[0][i]
        answer = dataset[1][i]
        answer_index  = letter_to_index_map[answer]
        cot_prompt = cot_template.format(question_text=question)
        cot_prompt_token = tokenizer.encode(cot_prompt, return_tensors="pt").to(device)
        model_return = model.generate(cot_prompt_token, max_length=300, temperature=0.0)
        model_return = tokenizer.decode(*model_return)
        model_return_ = "The answer is ".join(model_return.split("The answer is ")[:-1]) + "The answer is "
        model_return_token = tokenizer.encode(model_return_, return_tensors="pt").to(device)
        model_return = model(model_return_token, return_dict=True)
        logits = model_return.logits[0, -1, :]
        probs = softmax(logits, dim=0)
        probs = torch.tensor([probs[token_map[letter]].item() for letter in ["True", "False"]])       
        probs = probs / probs.sum() 
        results.append(1 - probs[answer_index].item())

    df = pd.DataFrame(results, columns=["conformal_scores"])
    df.to_csv("conformal_scores_cot.csv", index=False)
    
    

if __name__ == "__main__":
    
    config_file = "config.yaml"
    
    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    dataset = [[], []]
    with open(config["dataset_file"], "r") as file:
        for line in file:
            line_ = line.split("\n")[0]
            question, answer = line_.split("-")
            dataset[0].append(question)
            dataset[1].append(answer == "True")
            
    dataset[0] = dataset[0][1:] # first element used to one shot in chain of thoughts
    dataset[1] = dataset[1][1:]
    
    torch.random.manual_seed(0)
    device = "cuda:1"

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct", 
        device_map=device, 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    
    token_map = {letter: tokenizer.encode(letter)[-1] for letter in "AB"}
    letter_to_index_map = {letter: index for index, letter in enumerate([True, False])}
    
    if config["inference_mode"] == "vanilla":
        conformal_predicition(dataset, device, token_map, letter_to_index_map, model, tokenizer, prompt_template_boolean, config["conformal_range"])
    elif config["inference_mode"] == "cot":
        inference_cot(dataset, device, token_map, letter_to_index_map, model, tokenizer, prompt_template_boolean, cot_template, config["val_range"])