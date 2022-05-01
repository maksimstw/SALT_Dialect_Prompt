from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import torch
import csv
import pandas as pd

import os
device = 'cuda' if torch.cuda.is_available else 'cpu'

import argparse
parser = argparse.ArgumentParser(description='commands')
parser.add_argument('--file_path', type=str, help='file path')
parser.add_argument('--output_path', type=str, help='output file path')
parser.add_argument('--cuda_id', type=str, help='output file path')
args = parser.parse_args()


def generate():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large").to(device=device)

    file_path = args.file_path
    output_path = args.output_path

    df = pd.read_csv(file_path)
    f = open(output_path, 'a', encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(['aave_gen', 'sae_gen'])

    for i in tqdm(range(len(df['history']))):
        chat_history_sae_ids = tokenizer.encode(df.iloc[i]['history'], return_tensors='pt').to(device=device)
        chat_history_aave_ids = tokenizer.encode(df.iloc[i]['history_aave'], return_tensors='pt').to(device=device)
        output_aave_ids = model.generate(chat_history_aave_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, top_k=50, top_p=0.95, do_sample=True)
        output_sae_ids = model.generate(chat_history_sae_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, top_k=50, top_p=0.95, do_sample=True)
        writer.writerow([
            tokenizer.decode(output_aave_ids[:, chat_history_aave_ids.shape[-1]:][0], skip_special_tokens=True), 
            tokenizer.decode(output_sae_ids[:, chat_history_sae_ids.shape[-1]:][0], skip_special_tokens=True)
        ])

    f.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    generate()
