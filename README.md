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

#### Initial Setup:
The following is the initial setup to run the GReG pipeline on 4 multi-hop QA datasets and 2 open-domain QA datasets. First, we install the prerequisite Python packages. Next, we download a DPR model fine-tuned on the Natural Questions (NQ) dataset. After that, we generate the Wikipedia passage embeddings file using a script from the FiD repository. We then download SpaCy's medium-size language core. Finally, we preprocess some of the downloaded datasets. Note that we have modified two of the FiD scripts (data.py and evaluation.py) to fit the GReG pipeline.
```
conda create --name greg python=3.8
conda activate greg
conda install pip
git clone https://github.com/facebookresearch/FiD.git
git clone https://github.com/the-greg-pipeline/main.git
mv main GReG
pip install pydantic==2.7.4
pip install -r FiD/requirements.txt
pip install -r GReG/requirements.txt
cd FiD
bash get-data.sh # NQ and TQA
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
bash src/download_datasets.sh # MuSiQue, HotpotQA, IIRC, and 2WikiMultihopQA
python src/iirc_preprocess.py --input data/IIRC/dev.json --output data/IIRC/dev_preprocessed.json
python src/nq_tqa_preprocess.py --input ../FiD/open_domain_data/NQ/test.json --output ../FiD/open_domain_data/NQ/test_preprocessed.json
python src/nq_tqa_preprocess.py --input ../FiD/open_domain_data/TQA/test.json --output ../FiD/open_domain_data/TQA/test_preprocessed.json
mv src/data.py ../FiD/src/
mv src/evaluation.py ../FiD/src/
```

From now on, all placeholders used in the commands can be replaced with a value from their corresponding option list below:
```
<dataset_name> = [MuSiQue, HotpotQA, IIRC, 2WikiMultihopQA, NQ, TQA]
<model_name> = [gpt-3.5-turbo, gpt-4o, llama-3]
<percentile> = [0, 50, 100]
<temperature> = [0, 1]
<top_p> = [0.1, 1]
<metrics> = [0, 1, 2]
<top_k> = [0, 1, 5, 10]
```

#### Template Query Generation using a Long-form Answer Generator (LAG)
Template query generation in two ways (percentile = 50: considering the entropies' median as a threshold to mask the named entities | 0: masking all the named entities present in the answer but absent in the question):
```
python template_query_generator.py \
    --data data/<dataset_name>/<model_name>/template_queries_<dataset_name>_<model_name>_<percentile>_<temperature>_<top_p>.jsonl \
    --model <model_name> \
    --api_key <your_openai_api_key> \
```

#### Passage Retrieval
Passage retrieval using a DPR model fine-tuned on the Natural Questions (NQ) dataset (percentile = 100: search query = question + answer, 50: search query = template query considering entropies, 0: search query = template query ignoring entropies):
```
python ../FiD/passage_retrieval.py \
    --model_path ../FiD/pretrained_models/nq_retriever \
    --passages ../FiD/open_domain_data/psgs_w100.tsv \
    --data results/<dataset_name>/<model_name>/template_queries_<dataset_name>_<model_name>_<percentile>.jsonl \
    --passages_embeddings ../FiD/wikipedia_embeddings_nq_00 \
    --output_path results/<dataset_name>/percentile_<percentile>/retrieved_passages_<model_name>_nq.json \
    --n-docs 10 \
```

#### Factoid Answer Generation using a Short-form Answer Generator (SAG)
Even if you are generating factoid answers for the dataset questions without additional context, make sure to follow the --data path format to ensure all parameters (such as percentile) are set correctly. Factoid answer generation using four different prompts (top_k > 0: with context | top_k = 0: without context) with metrics being either 0: (EM, ROUGE-L F1), 1: (Sacc, Lacc), or 2: (EM, ROUGE-L F1, Semantic Similarity):
```
python factoid_answer_generator.py \
      --data results/<dataset_name>/percentile_<percentile>/retrieved_passages_<model_name>.json \
      --model <model_name> \
      --top_k <top_k> \
      --api_key <your_openai_api_key> \
      --metrics <metrics>
```

## Prompts
Following are the prompts used for the GReG pipeline. We have two sets of prompts: Long-form Answer Generation prompts and Short-form Answer Generation prompts. The former is designed to encourage the LLM to output as much knowledge as possible to help enhance the template query for better passage retrieval. The latter is designed to make the model respond in a few words (factoid), avoiding additional or unnecessary responses. Please note that these prompts are used to obtain the Exact Match (EM) and ROUGE-L F1 results reported in the results table. We also used another set of prompts ([credit](https://github.com/SamyAteia/bioasq)) to evaluate the models with Strict and Lenient Accuracies (Sacc and Lacc); however, we did not find these metrics informative, so we avoided including them in the results. Nevertheless, you can find these prompts in the code.

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
