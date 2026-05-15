# Sentiment Analysis with BERT

A fine-tuned BERT model that classifies text into **Negative**, **Neutral**, or **Positive** sentiment. Trained on a Kaggle dataset, this project freezes most of BERT's weights and only trains the final classification layer — which keeps things fast and avoids overfitting.

---

## How it works

The model is built on top of `bert-base-uncased` from HuggingFace. Instead of training BERT from scratch (which would take forever), we freeze all of BERT's parameters and just attach a small linear layer on top that learns to map BERT's 768-dimensional output into one of 3 sentiment classes.

It's a classic transfer learning setup — BERT does the heavy lifting of understanding language, and our tiny classifier head learns what "negative", "neutral", and "positive" look like in that space.

---

## Project Structure

```
sentiment_analysis/
├── Sentiment_Analysis.ipynb   # The original notebook (exploration + training)
├── Sentiment_Analysis.py      # Clean script version of the notebook
├── inference.py               # Run the model interactively on your own text
└── model_parameters.pt        # Saved model weights (generated after training)
```

---

## Setup

You'll need Python and the following packages:

```bash
pip install torch transformers pandas scikit-learn kagglehub tqdm
```

Also make sure you have a GPU — training on CPU will be painfully slow.

---

## Training

The training script downloads the dataset automatically via `kagglehub`, preprocesses the text with BERT's tokenizer, and trains for 10 epochs using Adam and CrossEntropy loss.

```bash
python Sentiment_Analysis.py
```

After training, the model weights are saved to `model_parameters.pt`.

---

## Running Inference

Once you have `model_parameters.pt`, you can chat with the model interactively:

```bash
python inference.py
```

It'll prompt you to type some text and will return one of:
- `Negative`
- `Neutral`
- `Positive`

Type `exit` to quit.

**Example:**

```
Enter your text  : I absolutely loved the movie!
Sentiment:  Positive

Enter your text  : It was okay, nothing special.
Sentiment:  Neutral

Enter your text  : This was a complete waste of time.
Sentiment:  Negative
```

---

## Dataset

[Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset) from Kaggle — downloaded automatically during training via `kagglehub`.

---

## Notes

- Only the classifier head is trained; BERT's weights are frozen.
- Max token length is capped at 128 to keep memory usage reasonable.
- Batch size is set to 400 — lower it if you run into GPU memory issues.
- Learning rate: `1e-4` with Adam optimizer.
