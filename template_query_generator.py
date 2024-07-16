import argparse
import openai
import numpy as np
from IPython.display import display, HTML
from time import time
import re
import spacy
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax
from copy import deepcopy
from termcolor import colored
import math
import json
import os

def highlight_ngrams(sentence, ngrams_to_highlight=[], color='red', highlight_all=True):
    if highlight_all:
        return colored(sentence, color)
    ngrams_to_highlight = sorted(ngrams_to_highlight, key=lambda x: len(x.split()), reverse=True)
    for ngram in ngrams_to_highlight:
        sentence = sentence.replace(ngram, colored(ngram, color))
    return sentence

def estimate_cost(total_samples, fixed_instructions_msg, model, questions, max_output_tokens_per_sample, contexts=None, word_to_token_ratio=1.33, include_contexts=True, question_key='question'):
    fixed_instructions_tokens = word_to_token_ratio * len(fixed_instructions_msg.replace('\n', ' ').strip().split())
    total_context_tokens = word_to_token_ratio * sum(list(map(lambda x: sum(list(map(lambda y: len(y.replace('\n', ' ').strip().split()), x))), contexts))) if include_contexts else 0
    total_input_tokens = word_to_token_ratio * sum(list(map(lambda q: len(q[question_key].replace('\n', ' ').strip().split()), questions))) + total_samples * fixed_instructions_tokens + total_context_tokens
    total_output_tokens = total_samples * max_output_tokens_per_sample
    cost_per_1M_tokens_input = 5 if model == 'gpt-4o' else (.5 if model == 'gpt-3.5-turbo' else (10 if model == 'gpt-4-turbo' else None))
    cost_per_1M_tokens_output = 15 if model == 'gpt-4o' else (1.5 if model == 'gpt-3.5-turbo' else (30 if model == 'gpt-4-turbo' else None))

    total_tokens, estimated_cost = total_input_tokens + total_output_tokens, cost_per_1M_tokens_input * (total_input_tokens / 1000000) + cost_per_1M_tokens_output * (total_output_tokens / 1000000)

    return estimated_cost, total_tokens

def calculate_entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:  # To avoid log(0)
            entropy -= p * math.log2(p)
    return entropy

def get_token_level_entropies(token_probabilities):
    entropies = []
    for probabilities in token_probabilities:
        entropy = calculate_entropy(probabilities)
        entropies.append(entropy)
    return entropies

def get_word_level_entropies(tokens, token_entropies):
    def combine_entropies(entropies):
        return max(entropies)
    words = []
    word_entropies = []
    current_word = ""
    current_entropies = []

    for token, entropy in zip(tokens, token_entropies):
        if token.startswith(' ') and current_word:
            words.append(current_word)
            word_entropies.append(combine_entropies(current_entropies))
            current_word = token.strip()
            current_entropies = [entropy]
        else:
            current_word += token
            current_entropies.append(entropy)

    if current_word:
        words.append(current_word)
        word_entropies.append(combine_entropies(current_entropies))

    return words, word_entropies

def remove_punctuation(input_string):
    pattern = r'[.,;:!?]'
    normalized_string = re.sub(pattern, '', input_string)
    return normalized_string

def generate_ngrams(words, n):
    words = list(map(remove_punctuation, words))
    ngrams_with_indices = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        indices = list(range(i, i+n))
        ngrams_with_indices.append((ngram, indices))
    return ngrams_with_indices

def locate_ngrams(words, n, output_entities):
    sentence_ngrams_with_indices = generate_ngrams(words, n)
    found_ngrams_with_indices = [(ngram, indices) for ngram, indices in sentence_ngrams_with_indices if ngram in list(map(remove_punctuation, output_entities))]
    return found_ngrams_with_indices

def combine_dependent_words_entropies(words, word_entropies, output_entities, max_window_size=5):
    for n in range(1, max_window_size + 1):
        found_ngrams = locate_ngrams(words, n, output_entities)
        for ngram in found_ngrams:
            ents = []
            for i in ngram[1]:
                ents.append(word_entropies[i])
                max_ents = max(ents)
            for i in ngram[1]:
                word_entropies[i] = max_ents
    return word_entropies

def filter_words(words, word_entropies, question_entities, output_entities, thresholds, mask=' '):
    all_filtered_words = {}
    for threshold in thresholds:
        th = np.percentile(word_entropies, threshold) # threshold = 0 -> ignore_entropies = True, threshold = 100 -> template_query = question
        filtered_words = []
        for word, entropy in zip(words, word_entropies):
            norm_word = remove_punctuation(word)
            if (entropy > th) and any(norm_word in ent for ent in output_entities) and (not any(norm_word in ent for ent in question_entities)):
                filtered_words.append(mask)
                continue
            filtered_words.append(word)
        all_filtered_words[threshold] = filtered_words
    return all_filtered_words

def get_entities(sentence):
    entities = []
    nlp = spacy.load("en_core_web_md")
    doc = nlp(sentence)
    for ent in doc.ents:
        entities.append(ent.text)
    return entities

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def read_json(filepath):
    data = []
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

class InvalidFileExtensionError(Exception):
    def __init__(self, extension, message="Invalid file extension"):
        self.extension = extension
        self.message = f"{message}: {extension}"
        super().__init__(self.message)

def meta_get_formatted_input(question, system_instruction):
    system = "System: This is a chat between a user and an artificial intelligence assistant."
    instruction = f"Please give a complete and concise answer to the following question. {system_instruction}"

    conversation = "User: " + instruction + " " + question + "\n\nAssistant:"
    formatted_input = system + "\n\n" + conversation

    return formatted_input

def meta_generator(model, tokenizer, question, max_tokens, temperature, top_p, system_instruction, template_query_generation=False):
    full_prompt = meta_get_formatted_input(question, system_instruction) # make this consistent with the factoid_answer_generator.py file
    tokenized_prompt = tokenizer(tokenizer.bos_token + full_prompt, return_tensors="pt").to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    if temperature == 0 and top_p == .1: # deterministic
        outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=max_tokens, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id, output_scores=template_query_generation, return_dict_in_generate=template_query_generation)
    else: # stochastic
        assert temperature > 0
        outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=max_tokens, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=temperature, top_p=top_p, output_scores=template_query_generation, return_dict_in_generate=template_query_generation)
    if template_query_generation == True:
        response = outputs[0][0][tokenized_prompt.input_ids.shape[-1]:]
    else:
        response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    if template_query_generation == True:
        top_probs_and_tokens = []
        for scores in outputs.scores:
            probs = softmax(scores, dim=-1)
            top_probs, top_indices = torch.topk(probs, 20) # 20 is the max value for Open AI models

            step_result = {
                'top_probs': [
                    {'token': tokenizer.decode([idx], skip_special_tokens=True), 'prob': prob.item()}
                    for idx, prob in zip(top_indices[0], top_probs[0])
                ],
                'token': tokenizer.decode([torch.argmax(scores[0])], skip_special_tokens=True)
            }
            top_probs_and_tokens.append(step_result)
        return tokenizer.decode(response, skip_special_tokens=True), top_probs_and_tokens
    return tokenizer.decode(response, skip_special_tokens=True), []

def read_data(filepath, ext):
    data = []
    if ext == 'json':
        data = read_json(filepath)
    elif ext == 'jsonl':
        data = read_jsonl(filepath)
    else:
        raise InvalidFileExtensionError(ext)
    id_key = None
    if 'id' in data[0]:
        id_key = 'id'
    elif '_id' in data[0]:
        id_key = '_id'
    n = len(data)
    return data, id_key, n

def main(opt):
    filepath = opt.data
    ext = filepath.split('.')[-1]
    dataset = filepath.split('/')[-2]
    assert dataset in ['MuSiQue', 'HotpotQA', 'IIRC', '2WikiMultihopQA', 'NQ', 'TQA']
    
    data, id_key, n = read_data(filepath, ext)
    
    question_key = 'question'
    answer_key = 'answer'
    
    model = opt.model
    assert model in ['gpt-3.5-turbo', 'gpt-4o', 'llama-3']
    if model == 'llama-3':
        model_id = "nvidia/Llama3-ChatQA-1.5-8B"
        meta_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        meta_tokenizer = AutoTokenizer.from_pretrained(model_id)
    v = opt.verbose
    assert v == 0 or v == 1
    max_tokens = opt.max_tokens
    assert max_tokens > 0
    percentile = list(set(opt.percentile))
    assert len(percentile) > 0
    temperature = opt.temperature
    assert 0 <= temperature <= 1
    model_top_p = opt.top_p
    assert .1 <= model_top_p <= 1
    
    system_instruction = "When answering questions, always include relevant information from the question in your response."
    
    ##### <prompt the user with the estimated total cost before the first iteration> #####
    if model in ['gpt-3.5-turbo', 'gpt-4o']:
        assert opt.api_key != None, "Please specify your OpenAI API key."
        openai.api_key = opt.api_key
        estimated_cost, _ = estimate_cost(n, system_instruction, model, data, max_tokens, include_contexts=False)
        res = input(f'Total estimated cost is: ${estimated_cost:.2f}. Continue? [y/n] ')
        assert res.strip().lower() == 'y', "User decided to abort the process."
    ##### </prompt the user with the estimated total cost before the first iteration> #####
    
    samples = {}
    for sample_index in trange(n):
        id = data[sample_index][id_key] if id_key != None else sample_index
        question = data[sample_index][question_key]
        question_entities = get_entities(question)
        answer = data[sample_index][answer_key]
        if v:
            print(f"Q: {highlight_ngrams(question, question_entities, highlight_all=False)}")
            print(f"A: {answer}")
            print(highlight_ngrams(f"{model} is responding...", color='green'))
            tic = time()
        if model in ['gpt-3.5-turbo', 'gpt-4o']:
            response = openai.chat.completions.create(
                model=model,
                messages=[{
                    "role": "system",
                    "content": system_instruction
                },{
                    "role": "user",
                    "content": question
                }],
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=20,
                temperature=temperature,
                top_p=model_top_p,
            )
            generated_answer = response.choices[0].message.content
        elif model in ['llama-3']:
            generated_answer, meta_top_probs = meta_generator(meta_model, meta_tokenizer, question, max_tokens, temperature, model_top_p, system_instruction, template_query_generation=True)
        if v:
            toc = time()
            print(highlight_ngrams(f"response time: {(toc - tic):.2f} s", color='blue'))
        generated_answer_entities = get_entities(generated_answer)
        if v:
            print('generated answer:')
            print(generated_answer)
            print('generated answer entities:')
            print(' '.join(generated_answer_entities))
            print(highlight_ngrams(generated_answer, generated_answer_entities, highlight_all=False))

        token_probabilities, possible_tokens, generated_tokens, generated_tokens_indices, top_p = [], [], [], [], []

        x = 0 # num_answers is set to 1 by default
        if model in ['gpt-3.5-turbo', 'gpt-4o']:
            for j in range(len(response.choices[x].logprobs.content)):
                top_20_logprobs = response.choices[x].logprobs.content[j].top_logprobs
                token_probabilities.append(list(map(lambda x: np.exp(x.logprob), top_20_logprobs)))
                possible_tokens.append(list(map(lambda x: x.token, top_20_logprobs)))
                generated_token = response.choices[x].logprobs.content[j].token
                generated_tokens.append(generated_token)
                try:
                    generated_token_index = list(map(lambda x: x.token, top_20_logprobs)).index(generated_token)
                    generated_tokens_indices.append(generated_token_index)
                    top_p.append(token_probabilities[0][generated_token_index])
                except:
                    generated_tokens_indices.append(None)
                    top_p.append(None)
        elif model in ['llama-3']:
            for j in range(len(meta_top_probs)):
                top_20_probs = meta_top_probs[j]['top_probs']
                token_probabilities.append(list(map(lambda x: x['prob'], top_20_probs)))
                possible_tokens.append(list(map(lambda x: x['token'], top_20_probs)))
                generated_token = meta_top_probs[j]['token']
                generated_tokens.append(generated_token)
                try:
                    generated_token_index = list(map(lambda x: x['token'], top_20_probs)).index(generated_token)
                    generated_tokens_indices.append(generated_token_index)
                    top_p.append(token_probabilities[0][generated_token_index])
                except:
                    generated_tokens_indices.append(None)
                    top_p.append(None)

        entropies = get_token_level_entropies(token_probabilities)
        words, word_entropies = get_word_level_entropies(generated_tokens, entropies)
        word_entropies = combine_dependent_words_entropies(words, deepcopy(word_entropies), generated_answer_entities)

        filtered_words = filter_words(words, word_entropies, question_entities, generated_answer_entities, percentile)
        if v:
            for key, value in filtered_words.items():
                print(f"template query (th={key}): {' '.join(value)}\n")
        for key, value in filtered_words.items():
            template_query = question + ' ' + ' '.join(value).strip()
            new_sample = {'id': id, 'question_org': question, 'question': template_query, 'answer': answer, 'generated_answer': generated_answer, 'token_level_entropies': list(zip(generated_tokens, list(map(lambda x: round(x, 3), entropies)))), 'word_level_entropies': list(zip(words, list(map(lambda x: round(x, 3), word_entropies)))), 'question_entities': question_entities, 'answer_entities': generated_answer_entities, 'token_level_median': round(np.percentile(entropies, 50), 3), 'token_level_mean': round(np.mean(entropies), 3), 'token_level_std': round(np.std(entropies), 3)}
            samples.setdefault(key, []).append(new_sample)
    for percent in percentile:
        if temperature == 0 and model_top_p == .1:
            file_path = f"results/{dataset}/{model}/template_queries_{dataset}_{model}_{percent}.jsonl"
        else:
            file_path = f"results/{dataset}/{model}/template_queries_{dataset}_{model}_{percent}_{temperature}_{model_top_p}.jsonl"
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'w') as file:
            for sample in samples[percent]:
                file.write(json.dumps(sample) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str, default=None, 
                        help="Path to the data")
    parser.add_argument('--model', required=False, type=str, default='gpt-3.5-turbo', 
                        help="Model name: gpt-3.5-turbo, gpt-4o, llama-3 (llama3-ChatQA-1.5-8B)")
    parser.add_argument('--api_key', required=False, type=str, default=None, 
                        help="OpenAI's API key")
    parser.add_argument('--max_tokens', required=False, type=int, default=50, 
                        help="Maximum number of the output tokens")
    parser.add_argument('--temperature', required=False, type=float, default=0, 
                        help="Model config")
    parser.add_argument('--top_p', required=False, type=float, default=.1, 
                        help="Model config")
    parser.add_argument('--percentile', required=False, type=int, nargs='+', default=[0, 50], 
                    help="0: ignore_entropies = True | 100: template_query = question + answer | 50: named_entities_masking_threshold = median(all_entropies)")
    parser.add_argument('--verbose', required=False, type=int, default=0, 
                        help="0: Do not print the logs | 1: Print the logs")

    args = parser.parse_args()
    main(args)
