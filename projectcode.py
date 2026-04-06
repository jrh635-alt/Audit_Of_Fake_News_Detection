#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:38:02 2026

@author: jack
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")

model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

import torch
def predict_fake(title,text):
    input_str = "<title>" + title + "<content>" +  text + "<end>"
    input_ids = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with torch.no_grad():
        output = model(input_ids["input_ids"].to(device), attention_mask=input_ids["attention_mask"].to(device))
    return dict(zip(["Fake","Real"], [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])] ))
    
print(predict_fake('This is a headline','This is real news, trust me'))


# %%%
fake_news = pd.read_csv(
    "Fake.csv",
    engine="python",      # more tolerant parser
    quotechar='"',
    escapechar='\\',      # helps with weird quotes
    on_bad_lines="skip"   # skip broken rows
)
real_news = pd.read_csv(
    "True.csv",
    engine="python",      # more tolerant parser
    quotechar='"',
    escapechar='\\',      # helps with weird quotes
    on_bad_lines="skip"   # skip broken rows
)

fake_news_results = {'fake': [],
                     'real': []}

fake_news_test = fake_news.head(10)


for _, article in fake_news_test.iterrows():
    result = predict_fake(article['title'], article['text'])
    fake_news_results['fake'].append(result['Fake'])
    fake_news_results['real'].append(result['Real'])
    


print (sum(fake_news_results['fake'])/len(fake_news_results['fake']))


#%%%

columns = [
    "id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_on_fire_count",
    "context"
]

newstotest = pd.read_csv(
    "test.tsv",
    sep="\t",
    header=None,     # important: tells pandas there are no headers
    names=columns    # your custom headers
)
    
fake_news_test = newstotest.head(10)

for _, article in fake_news_test.iterrows():
    result = predict_fake(article['id'], article['statement'])
    fake_news_results['fake'].append(result['Fake'])
    fake_news_results['real'].append(result['Real'])
    print(f"Statement: {article['statement']}, Prediction: {result['Real']}, True Value: {article['label']}")
    


print (sum(fake_news_results['fake'])/len(fake_news_results['fake']))
    
#%%%

reuters_count = real_news['text'].str.contains('Reuters').sum()

print(f"Number of articles containing 'Reuters': {reuters_count}")