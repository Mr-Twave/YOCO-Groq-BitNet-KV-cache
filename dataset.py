from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
def init(self, texts, tokenizer, max_len=512):
self.texts = texts
self.tokenizer = tokenizer
self.max_len = max_len

def __len__(self):
    return len(self.texts)

def __getitem__(self, idx):
    text = self.texts[idx]
    encoding = self.tokenizer.encode_plus(
        text,
        max_length=self.max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten()
    }



def create_data_loader(texts, tokenizer, max_len, batch_size):
ds = TextDataset(texts, tokenizer, max_len)
return DataLoader(ds, batch_size=batch_size, num_workers=4)
