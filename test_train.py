import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in tqdm(data_loader):
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
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in tqdm(data_loader):
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

def train_model(model, train_data_loader, val_data_loader, loss_fn, optimizer, device, scheduler, n_examples_train, n_examples_val, epochs):
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, n_examples_train)
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, n_examples_val)
        
        print(f'Train loss {train_loss} accuracy {train_acc}')
        print(f'Val loss {val_loss} accuracy {val_acc}')
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
    
    return history
