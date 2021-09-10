from transformers import DistilBertForQuestionAnswering
from prepare_dataset import prepare_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
from tqdm import tqdm
import pytorch_lightning as pl


def main():
    dataset = prepare_dataset()

    train_dataset = dataset["train"]
    train_dataset = val_dataset = dataset["val"]

    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in tqdm(range(1), desc="Epoch"):
        for batch in tqdm(train_loader, "Training"):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optim.step()

    model.eval()

if __name__ == "__main__":
    main()
