# -*- coding: utf-8 -*-
"""tech_challenge_fase_3_v2.ipynb

Code for use on Colab with Google Drive

Original file is located at
    https://colab.research.google.com/drive/1Ntg4ALJ3VWLdyfilRHw-onS0zmo8bkmM
"""

from google.colab import drive
drive.mount('/content/drive')

import json

def process_titles_descriptions_file(file_path, processed_data):
    #Lê um arquivo JSON do nosso dataset, processa cada notícia para formatar conforme o solicitado e adiciona à lista processed_data
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = [json.loads(line) for line in open(file_path, 'r')]

        count = 0
        limit = 100000 #pegando as primeiras 10 mil linhas primeiro para testar

        for item in json_data:

            if count >= limit:
                break

            title = item["title"]
            description = item["content"]
            formatted_text = f"CREATE A DESCRIPTION FOR THIS TITLE.\n[|Title|] {title}[|eTitle|]\n\n[|description|]{description}[|eDescription|]"

            if description and description.strip():
              processed_data.append({"input": formatted_text})
              count += 1

              if count >= limit:
                break

# Lista para armazenar todos os dados processados
processed_data = []

# Adicionar dados processados do arquivo JSON à lista
process_titles_descriptions_file(r'drive/MyDrive/COLAB/trn.json', processed_data)

# Salvar todos os dados processados em um arquivo JSON
output_filename = r'drive/MyDrive/COLAB/titles_dataset_chat_data.json'
with open(output_filename, 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, ensure_ascii=False, indent=4)

print(f"Todos os dados reformatados foram salvos em '{output_filename}'.")
print(f"Exemplo de processed_data: {processed_data[:5]}")  # Mostra os primeiros 5 exemplos para checar

!python -m pip install --upgrade pip
!pip install triton
!pip install xformers
!pip install transformers
!pip install torch

!conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
!conda activate unsloth_env

!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import json
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

DATA_PATH = "drive/MyDrive/COLAB/titles_dataset_chat_data.json"
OUTPUT_PATH_DATASET = "drive/MyDrive/COLAB/formatted_titles_dataset_chat_data.json"

# Reduzir o tamanho máximo de sequência
max_seq_length = 1024  # Tamanho reduzido para economizar memória
dtype = torch.float16  # Usar float16 para otimizar o uso de memória
load_in_4bit = True

# Tentar um modelo menor primeiro para evitar problemas de memória
model_name = "unsloth/mistral-7b-bnb-4bit"  # Modelo menor com quantização 4-bit

# Carregar o modelo com offloading automático entre CPU e GPU
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,  # Modelo escolhido
    max_seq_length=max_seq_length,  # Sequência ajustada
    dtype=dtype,  # Usando float16
    load_in_4bit=load_in_4bit,  # Quantização 4-bit
    device_map="auto"  # Mapeamento automático entre CPU e GPU
)

print("Modelo carregado com sucesso!")

def format_dataset_into_model_input(data):
    def separate_text(full_text):
        title_start = full_text.find("[|Title|]") + len("[|Title|]")
        title_end = full_text.find("[|eTitle|]")
        description_start = full_text.find("[|description|]") + len("[|description|]")
        description_end = full_text.find("[|eDescription|]")

        instruction = full_text.split('\n')[0]
        input_text = full_text[title_start:title_end].strip()
        response = full_text[description_start:description_end].strip()

        return instruction, input_text, response

    # Inicializando as listas para armazenar os dados
    instructions = []
    inputs = []
    outputs = []

    # Processando o dataset
    for prompt in data['train']['input']:
        instruction, input_text, response = separate_text(prompt)
        instructions.append(instruction)
        inputs.append(input_text)
        outputs.append(response)

    # Criando o dicionário final
    formatted_data = {
        "instruction": instructions,
        "input": inputs,
        "output": outputs
    }

    # Salvando o resultado em um arquivo JSON
    with open(OUTPUT_PATH_DATASET, 'w') as output_file:
        json.dump(formatted_data, output_file, indent=4)

    print(f"Dataset salvo em {OUTPUT_PATH_DATASET}")

data_from_json = load_dataset("json", data_files="drive/MyDrive/COLAB/titles_dataset_chat_data.json")
format_dataset_into_model_input(data_from_json)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",

    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

from datasets import load_dataset

OUTPUT_PATH_DATASET = "drive/MyDrive/COLAB/formatted_titles_dataset_chat_data.json"

dataset = load_dataset("json", data_files=OUTPUT_PATH_DATASET, split = "train")

print(dataset[:5])

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):

        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

dataset = dataset.map(formatting_prompts_func, batched = True,)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 20,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "CREATE A DESCRIPTION FOR THIS TITLE",
        "The Lord Of The Rings.", # input
        "",
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "CREATE A DESCRIPTION FOR THIS TITLE",
        "The Little Prince", # input
        "",
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

model.save_pretrained("drive/MyDrive/COLAB/fine-tuning-fiap/lora_model") # Local saving
tokenizer.save_pretrained("drive/MyDrive/COLAB/fine-tuning-fiap/lora_model")

import json
titles_test = []

json_data = [json.loads(line) for line in open('drive/MyDrive/COLAB/tst.json', 'r')]
titles_test = [item['title'] for item in json_data]

FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "CREATE A DESCRIPTION FOR THIS TITLE",
        titles_test[15], # input
        "",
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

import torch
print(torch.cuda.is_available())  # Deve retornar True se a GPU estiver ativa

!nvidia-smi

if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "drive/MyDrive/COLAB/fine-tuning-fiap/lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


inputs = tokenizer(
[
alpaca_prompt.format(
        "CREATE A DESCRIPTION FOR THIS TITLE",
        titles_test[1],
        "",
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
