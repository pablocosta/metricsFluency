"""USING COLA DATA-SET"""

import torch

from tqdm import tqdm
   
import os
import re
import csv
import sys
import ftfy
import nltk
from nltk.tokenize import sent_tokenize
import argparse
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", default='Helsinki-NLP/opus-mt-en-ROMANCE', type=str, required=False,
                    help="Model to be used on the translation")
parser.add_argument("--target_language", default="pt_br", type=str, required=False,
                    help="Target language code. The available codes can be found here \
                    https://developers.google.com/admin-sdk/directory/v1/languages")
parser.add_argument("--input_file", default="./data/outDomainDev.tsv", type=str, required=False,
                    help=".tsv file with MSMarco documents to be translated.")
parser.add_argument("--output_dir", default="./data/", type=str, required=False,
                    help="Path to save the translated file.")
parser.add_argument("--batch_size", default=256, type=int, required=False,
                    help="Batch size for translation")
parser.add_argument("--num_workers", default=8, type=int, required=False,
                    help="Background workers")
parser.add_argument("--num_beams", default=5, type=int, required=False,
                    help="Beams used in translation decoding")                        
parser.add_argument("--max_seq_length", default=30, type=int, required=False,
                    help="The maximum total input sequence length after tokenization. Sequences longer \
                          than this will be truncated, sequences shorter will be padded.")
  



class nmtData(Dataset):
    '''
    Pytorch's dataset abstraction for EuroParl.
    '''

    def __init__(self, filePath, targetLanguage):

        self.documents = self.loadCOLA(filePath, targetLanguage)
    def __len__(self):
        return len(self.documents)

    def cleanText(self, text: str):

      return text.translate(str.maketrans('', '', "\"'"))

    def loadCOLA(self, filePath:str, targetLanguage: str):
        '''

        Returns a list with tuples of [(doc_id, doc)].
        It uses ftfy to decode special carachters.
        Also, the special translation token ''>>target_language<<' is
        added to sentences.
        Args:
            - file_path (str): The path to the UStance collection file.
            - mode (str): Source or Target
        '''

        documents = []
        df = open(filePath, "r")
        for index, row in tqdm(enumerate(df), desc="Reading .csv file"):
            
            #for
            lines = row.strip().split("\t")

            if index == 0:
              continue

            doc_lines = sent_tokenize(self.cleanText(ftfy.ftfy(lines[3].lower())), "english")
            for doc in doc_lines:
                if len(doc) > 1:
                    documents.append((index, ">>"+targetLanguage+"<< " + doc))
        
        return documents

    def __getitem__(self,idx):
        doc_id, doc = self.documents[idx]
        return doc_id, doc




args = parser.parse_known_args()[0]


device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = MarianTokenizer.from_pretrained(args.model_name_or_path)
model     = MarianMTModel.from_pretrained(args.model_name_or_path).to(device).eval()

output_file = args.input_file + '_translated'
train_ds = nmtData(args.input_file, args.target_language)

output = open(output_file, 'a', encoding='utf-8', errors='ignore')
output.close()


translation_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

output = open(output_file, 'w+', encoding='utf-8', errors='ignore')

for batch in tqdm(translation_loader, desc="Translating..."):

    doc_ids   = batch[0]
    documents = batch[1]
    
    tokenized_text = tokenizer.prepare_seq2seq_batch(
        documents, 
        max_length=args.max_seq_length,
        return_tensors="pt")

    with torch.no_grad():
        translated = model.generate(
            input_ids=tokenized_text['input_ids'].to(device), 
            max_length=args.max_seq_length, 
            num_beams=args.num_beams,
            do_sample=True)

        translated_documents = tokenizer.batch_decode(
            translated, 
            skip_special_tokens=True)
    for doc_id, translated_doc in zip(doc_ids, translated_documents):
        output.write(str(doc_id.tolist()) + '\t' + translated_doc + '\n')
    
print("Done!")
output.close()


