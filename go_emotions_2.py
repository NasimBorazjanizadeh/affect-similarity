import transformers
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          PreTrainedModel, DistilBertModel, DistilBertForSequenceClassification,
                          TrainingArguments, Trainer, AutoConfig)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset

#importing the dataset and splitting tha datset to make the train and test splits
emotions = load_dataset("go_emotions", split="train")
emotions = emotions.train_test_split(test_size=0.15)
print(emotions)

label_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
              'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
              'disapproval', 'disgust', 'embarrassment', 'excitement',
              'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 
              'optimism', 'pride', 'realization', 'relief', 'remorse', 
              'sadness', 'surprise', 'neutral']

#changing the format of labels from [i, j] where i and j are the labels associated 
#with this sample to [0..010..010..0] where the ith and jth index are set to one
ds = emotions.map(lambda x : {"labels": [1 if i in x["labels"] else 0 for i in range(len(label_cols))]})
print(ds)
print(ds["train"][0])


#model checkpoint to import the pre-trained model and tokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, problem_type="multi_label_classification")


#tokenizing
def tokenize_and_encode(examples):
  return tokenizer(examples["text"], truncation=True)

cols = ds["train"].column_names
cols.remove("labels")
ds_enc = ds.map(tokenize_and_encode, batched=True, remove_columns=cols)
print(ds_enc)

#changing the label types to floats for the loss
ds_enc.set_format("torch")
ds_enc = (ds_enc
          .map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
          .rename_column("float_labels", "labels"))

print(ds_enc["train"][0])

#importing pre-trained model
num_labels=28
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels, problem_type="multi_label_classification").to('cuda')

def compute_metrics(eval_pred):
  """changing the evaluation metric to return the accuracy thershold 

  Args:
      eval_pred (tuple(numpy.ndarray)): evaluation predictions calculated at
      the end of each each evaluation phase on the whole arrays of predictions/labels.

  Returns:
      dict :  dictionary of the name of the metric (accuracy_threshold) and float values
  """
  predictions, labels = eval_pred
  predictions = torch.from_numpy(predictions)
  labels = torch.from_numpy(labels)
  predictions = predictions.sigmoid()
  return {'accuracy_thresh': ((predictions>0.5)==labels.bool()).float().mean().item()}


batch_size = 32
#logging configuration to see the training loss
logging_steps = len(ds_enc["train"]) // batch_size

args = TrainingArguments(
    output_dir="go_emotions_8",
    evaluation_strategy = "epoch",
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_steps=logging_steps,
    push_to_hub=True
)

trainer = Trainer(
    model,
    args,
    train_dataset=ds_enc["train"], 
    eval_dataset=ds_enc["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer)

trainer.train()
print(trainer.evaluate())
