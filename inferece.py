import yaml
from tqdm import tqdm
import torch
import pandas as pd
import torch
from torch.nn.functional import softmax
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from prompt_templates import *


def inference_vanilla(dataset, device, token_map, letter_to_index_map, model, tokenizer, prompt_template_boolean, val_range):
    results = []
    is_probs = True
    for i in tqdm(range(val_range[0], val_range[1])):
        question = dataset[0][i]
        answer = dataset[1][i]
        answer_index  = letter_to_index_map[answer]
        prompt = prompt_template_boolean.format(question_text=question)
        
        prompt_token = tokenizer.encode(prompt, return_tensors="pt").to(device)
        model_return = model(prompt_token, return_dict=True)
        logits = model_return.logits[0, -1, :]
        if is_probs:
            probs = softmax(logits, dim=0)
            probs = torch.tensor([probs[token_map[letter]].item() for letter in "AB"])
            probs = probs / probs.sum()
            probs = probs.tolist()
            probs.append(answer_index)
            results.append(probs)
        else:
            logits_letter = torch.tensor([logits[token_map[letter]].item() for letter in "AB"])
            results.append([answer_index, logits_letter.argmax().item()])
    if is_probs:
        df = pd.DataFrame(results, columns=["prob_1", "prob_2", "answer"])
        df.to_csv("results_vanilla_probs.csv", index=False)
    else:
        df = pd.DataFrame(results, columns=["answer", "prediction"])
        df.to_csv("results_vanilla.csv", index=False)
    
def inference_cot(dataset, device, token_map, letter_to_index_map, model, tokenizer, prompt_template_boolean, cot_template, val_range):
    results = []
    is_probs = True
    for i in tqdm(range(val_range[0], val_range[1])):
        question = dataset[0][i]
        answer = dataset[1][i]
        answer_index  = letter_to_index_map[answer]
        cot_prompt = cot_template.format(question_text=question)
        cot_prompt_token = tokenizer.encode(cot_prompt, return_tensors="pt").to(device)
        model_return = model.generate(cot_prompt_token, max_length=300, temperature=0.0)
        model_return = tokenizer.decode(*model_return)
        
        prompt = prompt_template_boolean.format(question_text=model_return)
        
        prompt_token = tokenizer.encode(prompt, return_tensors="pt").to(device)
        model_return = model(prompt_token, return_dict=True)
        logits = model_return.logits[0, -1, :]
        if is_probs:
            probs = softmax(logits, dim=0)
            probs = torch.tensor([probs[token_map[letter]].item() for letter in "AB"])
            probs = probs / probs.sum()
            probs = probs.tolist()
            probs.append(answer_index)
            results.append(probs)
        else:
            logits_letter = torch.tensor([logits[token_map[letter]].item() for letter in "AB"])
            results.append([answer_index, logits_letter.argmax().item()])
    if is_probs:
        df = pd.DataFrame(results, columns=["prob_1", "prob_2", "answer"])
        df.to_csv("results_cot_probs.csv", index=False)
    else:
        df = pd.DataFrame(results, columns=["answer", "prediction"])
        df.to_csv("results_cot.csv", index=False)

def inference_conformal_prediction(dataset, device, token_map, letter_to_index_map, model, tokenizer, prompt_template_boolean, cot_template, val_range, conformal_scores_file, alpha):
    
    conformal_scores = pd.read_csv(conformal_scores_file)
    n = conformal_scores.shape[0]
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(conformal_scores["calibrating_scores"].to_numpy(), q_level, method='higher')
    
    
    results = []
    is_probs = True
    for i in tqdm(range(val_range[0], val_range[1])):
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
        
        max_prob = probs.max().item()
        if max_prob > qhat:
            if is_probs:
                probs = probs.tolist()
                probs.append(answer_index)
                results.append(probs)
            else:
                results.append([answer_index, probs.argmax().item()])
        else:
            cot_prompt = cot_template.format(question_text=question)
            cot_prompt_token = tokenizer.encode(cot_prompt, return_tensors="pt").to(device)
            model_return = model.generate(cot_prompt_token, max_length=300, temperature=0.0)
            model_return = tokenizer.decode(*model_return)
            prompt = prompt_template_boolean.format(question_text=model_return)
        
            prompt_token = tokenizer.encode(prompt, return_tensors="pt").to(device)
            model_return = model(prompt_token, return_dict=True)
            logits = model_return.logits[0, -1, :]
            if is_probs:
                probs = softmax(logits, dim=0)
                probs = torch.tensor([probs[token_map[letter]].item() for letter in "AB"])
                probs = probs / probs.sum()
                probs = probs.tolist()
                probs.append(answer_index)
                results.append(probs)
            else:
                logits_letter = torch.tensor([logits[token_map[letter]].item() for letter in "AB"])
                results.append([answer_index, logits_letter.argmax().item()])
    if is_probs:
        df = pd.DataFrame(results, columns=["prob_1", "prob_2", "answer"])
        df.to_csv(f"cp_probs_{alpha}_{qhat}.csv", index=False)
    else:
        df = pd.DataFrame(results, columns=["answer", "prediction"])
        df.to_csv(f"cp_{alpha}_{qhat}.csv", index=False)

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
        inference_vanilla(dataset, device, token_map, letter_to_index_map, model, tokenizer, prompt_template_boolean, config["val_range"])
    elif config["inference_mode"] == "cot":
        inference_cot(dataset, device, token_map, letter_to_index_map, model, tokenizer, prompt_template_boolean, cot_template, config["val_range"])
    elif config["inference_mode"] == "cp":
        inference_conformal_prediction(dataset, device, token_map, letter_to_index_map, model, tokenizer, prompt_template_boolean, cot_template, config["val_range"], config["conformal_scores_file"], config["alpha"])