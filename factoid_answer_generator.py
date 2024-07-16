# Credit for Strict (Sacc) and Lenient (Lacc) Accuracy Prompts for OpenAI LLMs:
# https://github.com/SamyAteia/bioasq

import argparse
import re
from unidecode import unidecode
import json
from tqdm import trange
import openai
from rouge_score import rouge_scorer
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from termcolor import colored
import os
import torch

def correct_json_list(json_string):
    corrected_string = re.sub(r'^.*?\[', '[', json_string, flags=re.DOTALL)
    corrected_string = re.sub(r'\].*$', ']', corrected_string, flags=re.DOTALL)
    try:
        _ = json.loads(corrected_string)
    except json.JSONDecodeError:
        return '[]'
    return corrected_string

def highlight_ngrams(sentence, ngrams_to_highlight=[], color='red', highlight_all=True):
    if highlight_all:
        return colored(sentence, color)
    ngrams_to_highlight = sorted(ngrams_to_highlight, key=lambda x: len(x.split()), reverse=True)
    for ngram in ngrams_to_highlight:
        sentence = sentence.replace(ngram, colored(ngram, color))
    return sentence

def compute_semantic_similarity(reference, generated):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        # Access the first element of the tuple which contains the hidden states
        hidden_states = outputs[0]
        return hidden_states.mean(dim=1).detach().numpy()

    ref_embedding = get_embedding(reference)
    gen_embedding = get_embedding(generated)

    cosine_similarity = np.dot(ref_embedding, gen_embedding.T) / (np.linalg.norm(ref_embedding) * np.linalg.norm(gen_embedding))
    return cosine_similarity.item()

def normalize(str1):
    str1 = str(str1)
    str1_normalized = re.sub(r'[^\w\d]', '', unidecode(re.sub(r'\b(a|an|the)\b', '', str1, flags=re.IGNORECASE)).lower())
    return str1_normalized

def compare(str1, str2):
    return str1 == str2

def compare_with_list(list1, str1):
    return any(list(map(lambda x: x == str1, list1)))

def estimate_cost(total_samples, fixed_instructions_msg, model, questions, max_output_tokens_per_sample, contexts=None, word_to_token_ratio=1.33, include_contexts=True, question_key='question'):
    fixed_instructions_tokens = word_to_token_ratio * len(fixed_instructions_msg.replace('\n', ' ').strip().split())
    total_context_tokens = word_to_token_ratio * sum(list(map(lambda x: sum(list(map(lambda y: len(y.replace('\n', ' ').strip().split()), x))), contexts))) if include_contexts else 0
    total_input_tokens = word_to_token_ratio * sum(list(map(lambda q: len(q[question_key].replace('\n', ' ').strip().split()), questions))) + total_samples * fixed_instructions_tokens + total_context_tokens
    total_output_tokens = total_samples * max_output_tokens_per_sample
    cost_per_1M_tokens_input = 5 if model == 'gpt-4o' else (.5 if model == 'gpt-3.5-turbo' else (10 if model == 'gpt-4-turbo' else None))
    cost_per_1M_tokens_output = 15 if model == 'gpt-4o' else (1.5 if model == 'gpt-3.5-turbo' else (30 if model == 'gpt-4-turbo' else None))

    total_tokens, estimated_cost = total_input_tokens + total_output_tokens, cost_per_1M_tokens_input * (total_input_tokens / 1000000) + cost_per_1M_tokens_output * (total_output_tokens / 1000000)

    return estimated_cost, total_tokens

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

def compute_strict_accuracy(exact_answer, generated_answers):
    return exact_answer == generated_answers[0] if len(generated_answers) > 0 else False

def compute_lenient_accuracy(exact_answer, generated_answers):
    return exact_answer in generated_answers

class InvalidFileExtensionError(Exception):
    def __init__(self, extension, message="Invalid file extension"):
        self.extension = extension
        self.message = f"{message}: {extension}"
        super().__init__(self.message)

def evaluate_saved_file(opt):
    filepath = opt.data
    data = read_jsonl(filepath)
    n = len(data)
    metrics = opt.metrics # 0: Exact Match and ROUGE | 1: Lenient and Strict Accuracy
    file_metrics = int(filepath.split('/')[-1].split('_')[4])
    assert file_metrics == metrics, "metrics argument does not match the saved file's metrics."
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # Only works if metrics = 0
    EM = 0; precision = 0; recall = 0; fmeasure = 0; sem_sim = 0 # Only works if metrics = 0
    Sacc = 0; Lacc = 0 # Only works if metrics = 1
    print(f'\n# of samples: {n}')
    for i in trange(n):
        datum = data[i]
        answer = datum['gold_answer']
        generated_answer = datum['pred_answer']
        mets = evaluate_factoid_answers(metrics, answer, generated_answer, scorer)
        EM += mets[0]; precision += mets[1]; recall += mets[2]; fmeasure += mets[3]; sem_sim += mets[4]; Sacc += mets[5]; Lacc += mets[6]
    if metrics == 0:
        acc = round(EM/n, 3); pre = round(precision/n, 3); rec = round(recall/n, 3); f1 = round(fmeasure/n, 3); ss = round(sem_sim/n, 3)
        print(f'\nEM: {EM}')
        print(f"Accuracy: {acc}")
        print(f"Precision: {pre}")
        print(f"Recall: {rec}")
        print(f"F-1: {f1}")
        print(f"Sem-Sim: {ss}")
    else:
        Sa = round(Sacc/n, 3); La = round(Lacc/n, 3)
        print(f'\nSacc: {Sa}')
        print(f"Lacc: {La}")

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

def add_contexts(full_prompt, contexts, top_k, include_titles):
    for j in range(top_k):
        title = contexts[j]['title']; text = contexts[j]['text']
        tmp = "\nText: "
        full_prompt += f"{j + 1}. {f'Title: {title}{tmp}' if include_titles else ''}{text}"
        full_prompt += '\n'
        if j != top_k - 1:
            full_prompt += '\n'
    return full_prompt

def openai_generator(api_key, model, full_prompt, max_tokens, temperature, model_top_p):
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": full_prompt
        }],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=model_top_p,
    )
    return response.choices[0].message.content

def evaluate_factoid_answers(metrics, answer, generated_answer, scorer):
    assert type(answer) == str or type(answer) == list
    EM, precision, recall, fmeasure, sem_sim, Sacc, Lacc = 0, 0, 0, 0, 0, 0, 0
    if metrics == 0 or metrics == 2: # Exact Match (EM) and ROUGE
        if type(answer) == str: # for MultihopQA datasets
            EM = compare(normalize(answer), normalize(generated_answer))
            scores = scorer.score(answer, generated_answer)['rougeL']
            precision = scores.precision; recall = scores.recall; fmeasure = scores.fmeasure
        else: # for NQ and TQA
            EM = compare_with_list(list(map(normalize, answer)), normalize(generated_answer))
            precisions, recalls, fmeasures = [], [], []
            for ans in answer:
                scores = scorer.score(ans, generated_answer)['rougeL']
                precisions.append(scores.precision); recalls.append(scores.recall); fmeasures.append(scores.fmeasure)
            precision = max(precisions); recall = max(recalls); fmeasure = max(fmeasures)
    elif metrics == 1: # Sacc and Lacc (Strict and Lenient Accuracy)
        factoids = json.loads(correct_json_list(generated_answer))
        if type(answer) == str: # for MultihopQA datasets
            Sacc = compute_strict_accuracy(normalize(answer), list(map(normalize, factoids)))
            Lacc = compute_lenient_accuracy(normalize(answer), list(map(normalize, factoids)))
        else: # for NQ and TQA
            Saccs, Laccs = [], []
            for ans in answer:
                Sacc = compute_strict_accuracy(normalize(ans), list(map(normalize, factoids)))
                Lacc = compute_lenient_accuracy(normalize(ans), list(map(normalize, factoids)))
                Saccs.append(Sacc); Laccs.append(Lacc)
            Sacc = max(Saccs); Lacc = max(Laccs)
    if metrics == 2: # Semantic Similarity
        if type(answer) == str: # for MultihopQA datasets
            sem_sim = compute_semantic_similarity(answer, generated_answer)
        else: # for NQ and TQA
            sem_sims = []
            for ans in answer:
                sem_sim = compute_semantic_similarity(ans, generated_answer)
                sem_sims.append(sem_sim)
            sem_sim = max(sem_sims)
    return EM, precision, recall, fmeasure, sem_sim, Sacc, Lacc

def meta_generator(model, tokenizer, question, max_tokens, temperature, top_p, top_k, contexts=None):
    full_prompt = meta_get_formatted_input(question, top_k, contexts)
    tokenized_prompt = tokenizer(tokenizer.bos_token + full_prompt, return_tensors="pt").to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    if temperature == 0 and top_p == .1: # deterministic
        outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=max_tokens, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id)
    else: # stochastic
        outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=max_tokens, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=temperature, top_p=top_p)
    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def meta_get_formatted_input(question, top_k, contexts, include_titles=True):
    system = "System: This is a chat between a user and an artificial intelligence assistant."
    if contexts == None:
        instruction = "Your task is to answer the following question in 2 to 3 words and in a format of factoid answer. DO NOT GENERATE ANYTHING MORE and generate TO-THE-POINT answers."
    else:
        instruction = "Your task is to answer the following question in 2 to 3 words and in a format of factoid answer with respect to the given contexts. DO NOT GENERATE ANYTHING MORE and generate TO-THE-POINT answers."
    
    all_contexts = ''
    for j in range(top_k):
        title = contexts[j]['title']; text = contexts[j]['text']
        tmp = "\nText: "
        all_contexts += f"{j + 1}. {f'Title: {title}{tmp}' if include_titles else ''}{text}"
        all_contexts += '\n'
        if j != top_k - 1:
            all_contexts += '\n'

    conversation = "User: " + instruction + " " + question + "\n\nAssistant:"
    formatted_input = system + "\n\n" + all_contexts + "\n\n" + conversation

    return formatted_input

def main(opt):
    filepath = opt.data
    ext = filepath.split('.')[-1]
    dataset = filepath.split('/')[-3]
    retriever = filepath.split('/')[-1].split('_')[-1].split('.')[0]
    template_temperature = 'high' if filepath.split('/')[-1].count('_') == 5 else 'low'
    assert retriever in ['nq', 'tqa']
    percentile = int(filepath.split('/')[-2].split('_')[-1])
    assert dataset in ['MuSiQue', 'HotpotQA', 'IIRC', '2WikiMultihopQA', 'NQ', 'TQA']
    
    data, id_key, n = read_data(filepath, ext)

    question_key = 'question' if percentile == 100 else 'question_org'
    answer_key = 'answer'
    
    model = opt.model
    assert model in ['gpt-3.5-turbo', 'gpt-4o', 'llama-3']
    if model == 'llama-3':
        model_id = "nvidia/Llama3-ChatQA-1.5-8B"
        meta_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        meta_tokenizer = AutoTokenizer.from_pretrained(model_id)
    v = opt.verbose
    assert v == 0 or v == 1
    LAG = opt.LAG
    assert LAG in ['gpt-3.5-turbo', 'gpt-4o', 'llama-3']
    max_tokens = opt.max_tokens
    assert max_tokens > 0
    temperature = opt.temperature
    assert 0 <= temperature <= 1
    model_top_p = opt.top_p
    assert .1 <= model_top_p <= 1
    api_key = opt.api_key
    assert api_key != None if model in ['gpt-3.5-turbo', 'gpt-4o'] else True, "Please specify your OpenAI API key."
    top_k = opt.top_k
    assert 0 <= top_k <= 10
    include_titles = opt.include_titles
    assert include_titles == 0 or include_titles == 1
    metrics = opt.metrics # 0: Exact Match (EM) and ROUGE | 1: Strict and Lenient Accuracy | 2: Exact Match (EM), ROUGE, and Semantic Similarity
    assert metrics == 0 or metrics == 1 or metrics == 2
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # Only works if metrics = 0
    EM = 0; precision = 0; recall = 0; fmeasure = 0; # Only works if metrics = 0 | 2
    sem_sim = 0 # Only works if metrics = 2
    Sacc = 0; Lacc = 0 # Only works if metrics = 1
    samples = []
    first_iter = True if opt.forced == 0 else False
    
    for i in trange(n):
        datum = data[i]
        question = datum[question_key]
        answer = datum[answer_key]
        id = datum[id_key] if id_key != None else i
        ##### <defining the prompt> #####
        full_prompt = ''; initial_prompt = ''
        if top_k > 0: # include_contexts = True
            if metrics == 0: # Exact Match (EM), ROUGE, and Semantic Similarity
                initial_prompt = f"Your task is to answer the following question in 2 to 3 words and in a format of factoid answer with respect to the given contexts. DO NOT GENERATE ANYTHING MORE and generate TO-THE-POINT answers.\n\nQuestion: {question}"
            else: # Sacc and Lacc (Strict and Lenient Accuracy)
                initial_prompt = f"Answer the following question with respect to the given contexts by returning only a JSON string array of entity names, numbers, or similar short expressions that are an answer to the question, ordered by decreasing confidence. The array should contain at max 5 elements but can contain less. If you don't know any answer return an empty list. Return only this list, it must not contain phrases and **must be valid JSON**.\n\nQuestion: {question}"
            full_prompt = f"{initial_prompt}\n\nContexts:\n"
            contexts = datum['ctxs']
            full_prompt = add_contexts(full_prompt, contexts, top_k, include_titles)
        else: # include_contexts = False
            if metrics == 0: # Exact Match (EM), ROUGE, and Semantic Similarity
                full_prompt = f"Your task is to answer the following question in 2 to 3 words and in a format of factoid answer. DO NOT GENERATE ANYTHING MORE and generate TO-THE-POINT answers.\n\nQuestion: {question}"
            else: # Sacc and Lacc (Strict and Lenient Accuracy)
                full_prompt = f"Answer the following question by returning only a JSON string array of entity names, numbers, or similar short expressions that are an answer to the question, ordered by decreasing confidence. The array should contain at max 5 elements but can contain less. If you don't know any answer return an empty list. Return only this list, it must not contain phrases and **must be valid JSON**.\n\nQuestion: {question}"
        ##### </defining the prompt> #####
        ##### <prompt the user with the estimated total cost before the first iteration> #####
        if model in ['gpt-3.5-turbo', 'gpt-4o'] and first_iter:
            estimated_cost, _ = estimate_cost(n, initial_prompt if top_k > 0 else full_prompt, model, data, max_tokens, list(map(lambda x: list(map(lambda y: y['title'] + '\n' + y['text'], x['ctxs'][:top_k])), data)))
            res = input(f'Total estimated cost is: ${estimated_cost:.2f}. Continue? [y/n] ')
            assert res.strip().lower() == 'y', "User decided to abort the process."
            first_iter = False
        ##### </prompt the user with the estimated total cost before the first iteration> #####
        if model in ['gpt-3.5-turbo', 'gpt-4o']:
            generated_answer = openai_generator(api_key, model, full_prompt, max_tokens, temperature, model_top_p)
        elif model in ['llama-3']:
            if top_k > 0:
                generated_answer = meta_generator(meta_model, meta_tokenizer, question, max_tokens, temperature, model_top_p, top_k, contexts)
            else:
                generated_answer = meta_generator(meta_model, meta_tokenizer, question, max_tokens, temperature, model_top_p, top_k)
        mets = evaluate_factoid_answers(metrics, answer, generated_answer, scorer)
        EM += mets[0]; precision += mets[1]; recall += mets[2]; fmeasure += mets[3]; sem_sim += mets[4]; Sacc += mets[5]; Lacc += mets[6]

        samples.append({'id': id, 'question': question, 'gold_answer': answer, 'pred_answer': generated_answer})
        ##### <printing logs> #####
        if v:
            print(f'\nP: {highlight_ngrams(full_prompt, color="blue")}')
            color = ''
            if metrics == 0:
                color = 'green' if mets[0] else 'red'
            else:
                color = 'green' if mets[6] else 'red'
            print(f'A: {answer}\nO: {highlight_ngrams(generated_answer, color=color)}')
            if metrics == 0:
                print(f'EM: {mets[0]:.3f}')
            elif metrics == 1:
                print(f'Lacc: {mets[6]:.3f}')
            elif metrics == 2:
                print(f'Sem-Sim: {mets[4]:.3f}')
        ##### </printing logs> #####
    ##### <printing evaluation results> #####
    if metrics == 0 or metrics == 2:
        acc = round(EM/n, 3); pre = round(precision/n, 3); rec = round(recall/n, 3); f1 = round(fmeasure/n, 3)
        print(f'\nEM: {EM}')
        print(f"Accuracy: {acc}")
        print(f"Precision: {pre}")
        print(f"Recall: {rec}")
        print(f"F-1: {f1}")
    elif metrics == 1:
        Sa = round(Sacc/n, 3); La = round(Lacc/n, 3)
        print(f'\nSacc: {Sa}')
        print(f"Lacc: {La}")
    if metrics == 2:
        ss = round(sem_sim/n, 3)
        print(f"Sem-Sim: {ss}")
    ##### </printing evaluation results> #####
    ##### <saving the results> #####
    if template_temperature == 'high':
        file_path = f'results/{dataset}/{model}/{retriever}/{dataset}_1.0_1.0_{percentile}_{model}_{LAG}_{top_k}_{include_titles}_{metrics}_{f"{EM}_{acc}_{pre}_{rec}_{f1}_{ss}" if metrics == 2 else (f"{Sa}_{La}" if metrics == 1 else f"{EM}_{acc}_{pre}_{rec}_{f1}")}.jsonl'
    else:
        file_path = f'results/{dataset}/{model}/{retriever}/{dataset}_{percentile}_{model}_{LAG}_{top_k}_{include_titles}_{metrics}_{f"{EM}_{acc}_{pre}_{rec}_{f1}_{ss}" if metrics == 2 else (f"{Sa}_{La}" if metrics == 1 else f"{EM}_{acc}_{pre}_{rec}_{f1}")}.jsonl'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    ##### </saving the results> #####

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str, default=None, 
                        help="Path to the data")
    parser.add_argument('--model', required=False, type=str, default='gpt-3.5-turbo', 
                        help="Model name: gpt-3.5-turbo, gpt-4o, llama-3 (llama3-ChatQA-1.5-8B)")
    parser.add_argument('--top_k', required=False, type=int, default=0, 
                        help="Top-k contexts used for Retrieval-Augmented Generation (RAG)")
    parser.add_argument('--include_titles', required=False, type=int, default=1, 
                        help="0: Do not include the titles for the contexts | 1: Include the titles for the contexts | Only works when top_k > 0")
    parser.add_argument('--api_key', required=False, type=str, default=None, 
                        help="OpenAI's API key")
    parser.add_argument('--max_tokens', required=False, type=int, default=50, 
                        help="Maximum number of the output tokens")
    parser.add_argument('--temperature', required=False, type=float, default=0, 
                        help="Model config")
    parser.add_argument('--top_p', required=False, type=float, default=.1, 
                        help="Model config")
    parser.add_argument('--metrics', required=False, type=int, default=0, 
                        help="0: Exact Match (EM) and ROUGE | 1: Strict and Lenient Accuracy | 2: Exact Match (EM), ROUGE, and Semantic Similarity")
    parser.add_argument('--eval_only', required=False, type=int, default=0, 
                        help="0: Generate factoid answers, then evaluate | 1: Only evaluate a pre-saved file")
    parser.add_argument('--verbose', required=False, type=int, default=0, 
                        help="0: Do not print the logs | 1: Print the logs")
    parser.add_argument('--LAG', required=False, type=str, default='gpt-3.5-turbo', 
                        help="Long-form Answer Generator (LAG): gpt-3.5-turbo, gpt-4o, llama-3")
    parser.add_argument('--forced', required=False, type=int, default=0, 
                        help="Skip the total cost estimation process for OpenAI models: 1")

    args = parser.parse_args()
    
    if args.eval_only:
        evaluate_saved_file(args)
    else:
        main(args)
