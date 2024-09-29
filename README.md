# Fine-Tuning de Modelos Pré-Treinados para Geração de Descrições

Este repositório contém o código para o fine-tuning de um modelo pré-treinado **Mistral-7B**, que está sendo usado para gerar descrições de produtos com base em seus títulos. Nesse projeto usamos a quantização de 4-bit para otimizar o uso de memória no Google Colab.

## Visão Geral

O objetivo deste projeto é ajustar um modelo pré-treinado para melhorar a capacidade de gerar descrições de produtos de forma precisa e relevante a partir de títulos. Utilizamos o **AmazonTitles-1.3MM**, um dataset contendo títulos e descrições de produtos da Amazon, e fazemos o fine-tuning usando a biblioteca **unsloth**.

## Funcionalidades

- **Processamento de Dataset**: O código formata os títulos e descrições no formato necessário para o treinamento.
- **Carregamento de Modelo Quantizado**: Usa o modelo Mistral-7B quantizado em 4-bit.
- **Fine-Tuning**: Ajusta o modelo para melhorar a geração de descrições.
- **Geração de Respostas**: Permite gerar descrições para novos títulos após o treinamento.

## Requisitos

- Python 3.11+
- Google Colab com GPU (para treinamento)
- Bibliotecas:
  - `transformers`
  - `torch`
  - `trl`
  - `unsloth`
  - `bitsandbytes`
  - `accelerate`

### Instalação das Dependências

Você pode configurar o ambiente no Google Colab usando os comandos abaixo:

```bash
!python -m pip install --upgrade pip
!pip install triton xformers transformers torch
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
```

Certifique-se de que a GPU está ativada:

```python
import torch
print(torch.cuda.is_available())  # Verifica se a GPU está ativa
!nvidia-smi  # Mostra o status da GPU
```

## Arquivos
- tech_challenge_fase_3.py: Arquivo para uso no Google Colab com o Google Drive.
- tech_challenge_fase_3.ipynb: Modelo executado do Colab.
- tech_challenge_fase_3_local_use.py: Arquivo para uso local. Neste, você deve alterar as rotas para os locais desejados de onde deseja buscar e armazenar as informações.

## Estrutura do Código

### Preparação do Dataset

```python
import json

def process_titles_descriptions_file(file_path, processed_data):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = [json.loads(line) for line in open(file_path, 'r')]
        limit = 100000  # Processa até 100 mil linhas

        for item in json_data[:limit]:
            title = item["title"]
            description = item["content"]
            formatted_text = f"CREATE A DESCRIPTION FOR THIS TITLE.\n[|Title|] {title}[|eTitle|]\n\n[|description|]{description}[|eDescription|]"
            if description.strip():
                processed_data.append({"input": formatted_text})

processed_data = []
process_titles_descriptions_file('drive/MyDrive/COLAB/trn.json', processed_data)

output_filename = 'drive/MyDrive/COLAB/titles_dataset_chat_data.json'
with open(output_filename, 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, ensure_ascii=False, indent=4)
```

### Carregamento do Modelo Pré-Treinado

```python
from unsloth import FastLanguageModel
import torch

model_name = "unsloth/mistral-7b-bnb-4bit"
max_seq_length = 1024
dtype = torch.float16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)

print("Modelo carregado com sucesso!")
```

### Fine-Tuning

```python
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

data_from_json = load_dataset("json", data_files="drive/MyDrive/COLAB/titles_dataset_chat_data.json")

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=20,
    fp16=True,
    output_dir="outputs"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data_from_json,
    args=training_args
)

trainer.train()
```

### Geração de Descrições

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
CREATE A DESCRIPTION FOR THIS TITLE

### Input:
{}

### Response:
{}
"""

inputs = tokenizer(
    [
        alpaca_prompt.format(
            "The Little Prince",  # Título para teste
            ""
        )
    ], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.batch_decode(outputs))
```

### Salvando o Modelo

```python
model.save_pretrained("drive/MyDrive/COLAB/fine-tuning-fiap/lora_model")
tokenizer.save_pretrained("drive/MyDrive/COLAB/fine-tuning-fiap/lora_model")
```

## Fonte de Dados

- **Dataset**: O dataset usado é o **AmazonTitles-1.3MM**, que contém títulos e descrições de produtos.
- **Formato**: O arquivo `trn.json` contém os títulos na chave `title` e as descrições na chave `content`.
