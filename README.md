# GReG

<table style="border: none;">
  <tr>
    <td valign="top" width="80%">
      <h2>A Generate-Retrieve-Generate Framework for Enhanced Generative Question Answering</h2>
      <p>We present the <b>G</b>enerate-<b>Re</b>trieve-<b>G</b>enerate (GReG) pipeline, which leverages the intrinsic knowledge of large language models (LLMs) for improved passage retrieval without requiring architectural changes or additional training. The GReG pipeline operates in three stages: (1) generating an initial long-form response with an LLM, (2) refining this response to form a retrieval query by removing hallucinations, and (3) using the original question and the retrieved passages to prompt an LLM to produce a concise factoid answer. We utilize advanced models such as GPT-3.5-Turbo-0125 and Llama3-ChatQA-1.5-8B for this process. Evaluations on four multi-hop QA datasets and two open-domain QA datasets demonstrate that GReG outperforms traditional QA approaches and in-context retrieval-augmented generation methods. These results highlight the effectiveness of incorporating LLM insights to enhance retrieval and improve answer accuracy in open-domain question answering.</p>
    </td>
    <td valign="top" width="20%">
      <img src="src/greg.png" width="100%">
    </td>
  </tr>
</table>

Initial Setup:
```
conda create --name greg python=3.8
conda activate greg
conda install pip
git clone https://github.com/facebookresearch/FiD.git
git clone https://github.com/the-greg-pipeline/main.git
pip install pydantic==2.7.4
pip install -r FiD/requirements.txt
pip install -r GReG/requirements.txt
cd FiD
bash get-data.sh
bash get-model.sh -m nq_retriever
python  generate_passage_embeddings.py \
        --model_path pretrained_models/nq_retriever \
        --passages open_domain_data/psgs_w100.tsv \
        --output_path wikipedia_embeddings_nq \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 500 \
cd GReG
python -m spacy download en_core_web_md
bash download_datasets.sh # MuSiQue, HotpotQA, IIRC, and 2WikiMultihopQA
```

1. Replace ```FiD/src/data.py``` with ```GReG/src/data.py```

2. Replace ```FiD/src/evaluation.py``` with ```GReG/src/evaluation.py```


```
<dataset_name> = [MuSiQue, HotpotQA, IIRC, 2WikiMultihopQA, NQ, TQA]
<model_name> = [gpt-3.5-turbo, gpt-4o, llama-3]
<retriever_name> = [nq]
<temperature> = [0, 1]
<top_p> = [0.1, 1]
```

Passage retrieval using a pre-trained DPR on NQ (percentile = 100: query = question + answer, 50: template query considering entropies, 0: template query ignoring entropies):
```
python ../FiD/passage_retrieval.py \
    --model_path ../FiD/pretrained_models/<retriever_name>_retriever \
    --passages ../FiD/open_domain_data/psgs_w100.tsv \
    --data results/<dataset_name>/<model_name>/template_queries_<dataset_name>_<model_name>_<0,50>.jsonl \
    --passages_embeddings ../FiD/wikipedia_embeddings_<retriever_name>_00 \
    --output_path results/<dataset_name>/percentile_<0,50,100>/retrieved_passages_<model_name>_<retriever_name>.json \
    --n-docs 10 \
```

Template query generation in 2 ways (percentile = 50: considering the entropies' median as a threshold to mask the named entities | 0: masking all the named entities present in the answer yet absent in the question):
```
python template_query_generator.py \
    --data data/<dataset_name>/<model_name>/template_queries_<dataset_name>_<model_name>_<0,50>_<temperature>_<top_p>.jsonl \
    --model <model_name> \
    --api_key <your_openai_api_key> \
```

Even if you are generating the factoid answers for the dataset itself, make sure to follow the --data path format to ensure all the parameters (like percentile) are set correctly.
Factoid answer generation using 4 (top_k > 0: with context | top_k = 0: without context) different prompts (metrics are either 0:(EM, ROUGE-F1), 1:(Sacc, Lacc), or 2:(EM, ROUGE-F1, Semantic Similarity)):
```
python factoid_answer_generator.py \
      --data results/<dataset_name>/percentile_<0,50,100>/retrieved_passages_<model_name>.json \
      --model <model_name> \
      --top_k <0,1,5,10> \
      --api_key <your_openai_api_key> \
      --metrics <0,1,2>
```

## Prompts
### Long-form Answer Generation Prompts:
#### GPT-3.5:
```
SYSTEM: When answering questions, always include relevant information from the question in your response.
USER: {question}
```
#### ChatQA:
```
System: This is a chat between a user and an artificial intelligence assistant.

User: Please give a complete and concise answer to the following question. When answering questions, always include relevant information from the question in your response. {question}

Assistant:
```

### Short-form (factoid) Answer Generation Prompts:
#### GPT-3.5:
```
Your task is to answer the following question in 2 to 3 words and in a format of factoid answer with respect to the given contexts.
DO NOT GENERATE ANYTHING MORE and generate TO-THE-POINT answers.

Question: {question}

Contexts:
1. Title: {title[0]}
Text: {text[0]}

2. Title: {title[1]}
Text: {text[1]}

...
```
#### ChatQA:
```
System: This is a chat between a user and an artificial intelligence assistant.

1. Title: {title[0]}
Text: {text[0]}

2. Title: {title[1]}
Text: {text[1]}

...

User: Your task is to answer the following question in 2 to 3 words and in a format of factoid answer with respect to the given contexts. DO NOT GENERATE ANYTHING MORE and generate TO-THE-POINT answers. {question}

Assistant:
```
