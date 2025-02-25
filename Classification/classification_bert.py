# use bert from huggingface to train a bet model to classify the data's category and sub-category
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the data
data = pd.read_csv('../data/merged_data.csv')

# Encode the labels
label_encoder = LabelEncoder()
# data['Category'] = label_encoder.fit_transform(data['Category'])
# data['Sub-Category'] = label_encoder.fit_transform(data['Sub-Category'])

# Combine Category and Sub-Category into a single label
data['Label'] = data['Category'].astype(str) + '-' + data['Sub-Category'].astype(str)
data['Label'] = label_encoder.fit_transform(data['Label'])

# Split the data into training, validation, and test sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(data['Product Name'], data['Label'], test_size=0.3)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5)

# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

class ClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        # labels is a pandas series, convert it to a list
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # print(idx)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ClassificationDataset(train_encodings, train_labels)
val_dataset = ClassificationDataset(val_encodings, val_labels)
test_dataset = ClassificationDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Move data to device
for batch in train_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
for batch in val_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}

# mapping function of label num to label text
def map_num_to_label(num):
    combine = label_encoder.inverse_transform([num])[0]
    category = combine.split('-')[0]
    sub_category = combine.split('-')[1]
    # decoder again
    # category = label_encoder.inverse_transform([int(category)])
    # sub_category = label_encoder.inverse_transform([int(sub_category)])
    return category, sub_category
# test the mapping function
print(map_num_to_label(6))
# build the dictionary of label num to label text
label_num_to_text = {}
for i in range(len(label_encoder.classes_)):
    label_num_to_text[i] = map_num_to_label(i)
print(label_num_to_text)

# Load the model, save the pretrained model at ../models/bert-base-uncased
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_), cache_dir='../models/bert-base-uncased')
model.to(device)  # Move model to GPU if available

# Define training arguments, save checkpoints at each epoch to the specified directory
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model on the test set
trainer.evaluate(test_dataset)

from sklearn.metrics import accuracy_score

# Get predictions on the test set
predictions = trainer.predict(test_dataset)

# Get the predicted labels
preds = predictions.predictions.argmax(-1)

# Calculate accuracy
accuracy = accuracy_score(test_labels, preds)
print(f'Test Accuracy: {accuracy:.4f}')

# predict the word "Comfortable Executive Mouse"
inputs = tokenizer("Comfortable Executive Mouse", return_tensors="pt").to(device)
outputs = model(**inputs)
preds = outputs.logits.argmax(-1)
print(label_num_to_text[preds.item()])