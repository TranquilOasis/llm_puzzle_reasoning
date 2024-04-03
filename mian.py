# Import necessary packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import transformers
import spacy
import pandas as pd
import numpy as np
import random
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load data
def load_data(data_file):
    data = pd.read_csv(data_file)
    return data

# Data preprocessing
def preprocess_data(data, tokenizer, max_seq_length):
    # Tokenize data
    input_ids = []
    attention_masks = []

    for text in data['text']:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_seq_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'].squeeze())
        attention_masks.append(encoded_dict['attention_mask'].squeeze())

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

# Model definition
class ClassificationModel(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(ClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

# Initialize model, tokenizer, and data loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClassificationModel(768, 2).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_seq_length = 128
train_data = load_data('train_data.csv')
input_ids, attention_masks = preprocess_data(train_data, tokenizer, max_seq_length)
train_dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, train_data['label'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Train model
for epoch in range(5):
    train(model, train_loader, optimizer, criterion, device)