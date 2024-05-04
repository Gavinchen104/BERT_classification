import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW  # Import AdamW from torch.optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess_data(data_file):
    df = pd.read_csv(data_file, sep=';')
    df_new = df[['title', 'label']]
    labels_to_sample = ['R', 'java', 'javascript', 'php', 'python']
    sampled_dfs = []
    for label in labels_to_sample:
        label_df = df_new[df_new['label'] == label]
        sampled_dfs.append(label_df.sample(frac=1))  # Consider adjusting this fraction if needed
    df_balanced = pd.concat(sampled_dfs)
    label_encoder = LabelEncoder()
    df_balanced['category'] = label_encoder.fit_transform(df_balanced['label'])
    return df_balanced['title'].tolist(), df_balanced['category'].tolist(),label_encoder

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_rate=0.1, hidden_units=256):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, output_attentions=True)
        self.dropout = nn.Dropout(dropout_rate)
        # First transform the BERT output to your desired hidden unit size
        self.transform = nn.Linear(self.bert.config.hidden_size, hidden_units)
        self.fc = nn.Linear(hidden_units, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        attention = outputs.attentions
        x = self.dropout(pooled_output)
        x = self.transform(x)  # Transform from 768 to 256
        logits = self.fc(x)
        return logits, attention

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    for batch_idx, batch in enumerate(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        _, predicted_labels = torch.max(outputs, dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_examples += labels.size(0)
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch {batch_idx + 1}/{len(data_loader)}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_examples
    print(f'End of Epoch, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

def evaluate(model, data_loader, device, label_encoder):
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

            #
    # Display confusion matrix and classification report
    print("Classification Report:")
    print(classification_report(actual_labels, predictions, target_names=label_encoder.classes_))

    cm = confusion_matrix(actual_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.title('Confusion Matrix for BERT Model')
    plt.show()

if __name__ == "__main__":
    data_file = 'full_dataset_v3.csv'  # Assume this is the path to your dataset
    texts, labels, label_encoder = load_and_preprocess_data(data_file)
    
    bert_model_name = 'bert-base-uncased'
    num_classes = 5
    max_length = 128
    batch_size = 16
    num_epochs = 4
    learning_rate = 0.0001
    dropout_rate = 0.3  # Example dropout rate
    hidden_units = 256  # Example number of hidden units in the linear layer

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BERTClassifier(bert_model_name, num_classes, dropout_rate, hidden_units).to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(num_epochs):
        print(f'\n--- Epoch {epoch + 1}/{num_epochs} ---')
        train(model, train_dataloader, optimizer, scheduler, device)
        evaluate(model, val_dataloader, device, label_encoder)
    
    torch.save(model.state_dict(), "bert_classifier2.pth")
