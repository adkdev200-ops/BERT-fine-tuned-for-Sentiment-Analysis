from Sentiment_Analysis import preprocess, SentimentModel, bert_model
model = SentimentModel(bert_model)
import torch
import numpy as np

model.load_state_dict(torch.load('model_parameters.pt'))

sentiments = ['Negative', 'Neutral', 'Positive']

model.eval()

print("Type 'exit' to quit")

while True:
    text = input("Enter your text  :")
    if text.lower() == 'exit':
        break

    else:
        input_ids, token_type_ids, attention_mask = preprocess(text)
        out = model(input_ids, token_type_ids, attention_mask)
        _, pred = torch.max(out, dim = -1)
        pred.squeeze_(0)
        pred = np.array(pred)

        print("Sentiment: ",sentiments[pred])