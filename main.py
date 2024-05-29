import argparse
import torch
from transformers import BertTokenizer
from model import YOCOModel
from dataset import create_data_loader
from train import train_model, eval_model
from inference import infer

def main(args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  
  # Load and preprocess data
  train_data_loader = create_data_loader(args.train_data, tokenizer, args.max_len, args.batch_size)
  val_data_loader = create_data_loader(args.val_data, tokenizer, args.max_len, args.batch_size)
  test_data_loader = create_data_loader(args.test_data, tokenizer, args.max_len, args.batch_size)
  
  # Initialize model, loss function, optimizer, and scheduler
  model = YOCOModel(args.hidden_size, args.num_layers, args.window_size).to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
  
  # Train the model
  train_model(model, train_data_loader, val_data_loader, loss_fn, optimizer, device, scheduler, len(args.train_data), len(args.val_data), args.epochs)
  
  # Evaluate the model
  test_acc, test_loss = eval_model(model, test_data_loader, loss_fn, device, len(args.test_data))
  print(f'Test loss {test_loss} accuracy {test_acc}')
  
  # Perform inference
  text = "Sample text for inference"
  output = infer(model, tokenizer, text)
  print(f'Inference output: {output}')



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
  parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
  parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
  parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of the model')
  parser.add_argument('--num_layers', type=int, default=12, help='Number of layers in the model')
  parser.add_argument('--window_size', type=int, default=512, help='Window size for sliding window attention')
  parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
  parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
  parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
  parser.add_argument('--scheduler_step_size', type=int, default=10, help='Step size for learning rate scheduler')
  parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma value for learning rate scheduler')
  parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
  args = parser.parse_args()

main(args)
