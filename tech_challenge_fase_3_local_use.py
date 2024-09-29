# -*- coding: utf-8 -*-
import json
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Função para processar títulos e descrições
def process_titles_descriptions_file(file_path, processed_data):
    # Lê um arquivo JSON do dataset local, processa cada item para formatar conforme solicitado
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = [json.loads(line) for line in file]

        count = 0
        limit = 100000  # Processa as primeiras 100 mil linhas (ajuste conforme necessário)

        for item in json_data:
            if count >= limit:
                break

            title = item["title"]
            description = item["content"]
            formatted_text = f"CREATE A DESCRIPTION FOR THIS TITLE.\n[|Title|] {title}[|eTitle|]\n\n[|description|]{description}[|eDescription|]"

            if description and description.strip():
                processed_data.append({"input": formatted_text})
                count += 1

# Lista para armazenar dados processados
processed_data = []

# Substitua o caminho pelo caminho local do seu dataset
process_titles_descriptions_file('caminho/para/seu/dataset.json', processed_data)

# Salva os dados processados em um arquivo JSON local
output_filename = 'caminho/para/saida/titles_dataset_chat_data.json'
with open(output_filename, 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, ensure_ascii=False, indent=4)

print(f"Dados reformatados salvos em '{output_filename}'.")

# Parâmetros para o modelo
DATA_PATH = output_filename
OUTPUT_PATH_DATASET = 'caminho/para/saida/formatted_titles_dataset_chat_data.json'

# Reduzir o tamanho máximo de sequência para economizar memória
max_seq_length = 1024
dtype = torch.float16  # Usar float16 para otimizar o uso de memória
load_in_4bit = True

# Carregar um modelo menor com quantização em 4-bit
model_name = "unsloth/mistral-7b-bnb-4bit"

# Carregar o modelo com offloading automático entre CPU e GPU
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)

print("Modelo carregado com sucesso!")

# Função para formatar o dataset
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

    instructions, inputs, outputs = [], [], []

    for prompt in data['train']['input']:
        instruction, input_text, response = separate_text(prompt)
        instructions.append(instruction)
        inputs.append(input_text)
        outputs.append(response)

    formatted_data = {"instruction": instructions, "input": inputs, "output": outputs}

    with open(OUTPUT_PATH_DATASET, 'w') as output_file:
        json.dump(formatted_data, output_file, indent=4)

    print(f"Dataset salvo em {OUTPUT_PATH_DATASET}")

# Carregar o dataset local
data_from_json = load_dataset("json", data_files=DATA_PATH)
format_dataset_into_model_input(data_from_json)

# Treinamento do modelo
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=20,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    output_dir="outputs",
    logging_steps=1
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data_from_json['train'],
    args=training_args
)

trainer.train()

# Função para gerar descrições baseadas em títulos
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
CREATE A DESCRIPTION FOR THIS TITLE

### Input:
{}

### Response:
{}
"""

# Função para gerar uma descrição
def generate_description(title):
    inputs = tokenizer(
        [
            alpaca_prompt.format(title, "")
        ], return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Exemplo de uso para gerar descrições
print(generate_description("The Lord Of The Rings"))
print(generate_description("The Little Prince"))

# Salvar o modelo ajustado
model.save_pretrained("caminho/para/salvar/modelo/final")
tokenizer.save_pretrained("caminho/para/salvar/modelo/final")
