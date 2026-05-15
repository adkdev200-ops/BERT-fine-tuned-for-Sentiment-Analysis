# Generated from: Sentiment_Analysis.ipynb
# Converted at: 2026-05-15T16:59:42.759Z


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import kagglehub
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



def preprocess(text):
  out = bert_tokenizer(text, padding = 'max_length', return_tensors ='pt')
  return out['input_ids'], out['token_type_ids'], out['attention_mask']


class SentimentModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()

        self.bert = bert_model
        self.classifier = nn.Sequential(
           nn.Linear(768, 3))

    def forward(self, input_ids, token_type_ids, attention_mask):
      out  = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)['pooler_output']
      out = self.classifier(out)
      return out




if __name__ == "__main__":

  path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")

  print("Path to dataset files:", path)

  import os
  os.listdir(path)

  train_path = path + '/train.csv'

  df = pd.read_csv(train_path, encoding = 'latin1', on_bad_lines = 'skip').dropna()

  df = df[['text', 'sentiment']]

  encoder = LabelEncoder()

  df['sentiment'] = encoder.fit_transform(df['sentiment'])



  device = torch.device('cuda')

  tensors = bert_tokenizer(df['text'].tolist(), padding = 'longest', max_length = 128, truncation = True, return_tensors = 'pt')

  class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
      self.input_ids =  encodings['input_ids']
      self.token_type_ids = encodings['token_type_ids']
      self.attention_mask = encodings['attention_mask']
      self.labels = torch.tensor(labels)

    def __len__(self):
      return len(self.labels)

    def __getitem__(self, idx):
      return self.input_ids[idx],self.token_type_ids[idx], self.attention_mask[idx], self.labels[idx]

  test_ds = SentimentDataset(tensors, df['sentiment'].values)

  test_dataloader = DataLoader(test_ds, batch_size = 400, shuffle = True)


  sentiment_model = SentimentModel(bert_model)

  sentiment_model = sentiment_model.to(device)

  for p in sentiment_model.parameters():
    p.requires_grad = False

  sentiment_model.linear1  = sentiment_model.classifier.requires_grad_(True)
  epochs = 10

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(sentiment_model.classifier.parameters(), lr=1e-4)

  for epoch in range(epochs):
      sum_loss = 0

      for input_ids, token_type_ids, attention_mask, label in tqdm(test_dataloader):
          input_ids, token_type_ids, attention_mask, label = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), label.to(device)
          preds = sentiment_model(input_ids, token_type_ids, attention_mask)

          loss = loss_fn(preds, label)

          optimizer.zero_grad()

          loss.backward()

          optimizer.step()

          sum_loss += loss.item()

      print(f"Total Loss: {sum_loss /(len(test_dataloader))}")


  torch.save(sentiment_model.state_dict(), 'model_parameters.pt')

