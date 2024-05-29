def evaluate_model(model, test_data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in tqdm(test_data_loader):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs[:, -1, :]
            loss = loss_fn(logits, input_ids[:, -1])
            correct_predictions += torch.sum(logits.argmax(dim=1) == input_ids[:, -1])
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

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
