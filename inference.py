def infer(model, tokenizer, text, max_len=512):
model = model.eval()
encoding = tokenizer.encode_plus(
text,
max_length=max_len,
padding='max_length',
truncation=True,
return_tensors='pt'
)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
return outputs
