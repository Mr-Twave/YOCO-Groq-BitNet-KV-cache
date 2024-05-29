# YOCO-Groq-BitNet-KV-cache
At the suggestion of a reddit user


# Install the required packages
`pip install torch torchvision transformers`

# Additional packages for YOCO
`pip install apex`
`pip install flash-attention`

No command line interpretation yet. This is just a demonstration repository.

# The math

Gated Retention:

The gated retention mechanism is inspired by the concept of a gate in recurrent neural networks (RNNs) and long short-term memory (LSTM) networks.
In the GatedRetention module, a linear transformation is applied to the input tensor x using a fully connected layer (nn.Linear). This linear transformation learns a set of weights and biases to map the input to a new space.
The output of the linear transformation is then passed through a sigmoid activation function (torch.sigmoid). The sigmoid function squashes the values between 0 and 1, acting as a gate.
The gated output is obtained by element-wise multiplication of the sigmoid output and the original input x. This allows the model to selectively retain or discard information based on the learned gate values.

Sliding Window Attention:

The sliding window attention mechanism is a variant of the self-attention mechanism used in Transformers.
In the SlidingWindowAttention module, the input tensor x is divided into overlapping windows of a fixed size (window_size).
For each window, the multi-head attention operation (nn.MultiheadAttention) is applied. Multi-head attention allows the model to attend to different parts of the input sequence simultaneously.
The attention operation computes a weighted sum of the input values within each window, where the weights are determined by the similarity between the query, key, and value vectors.
The attended output for each window is then concatenated to form the final output tensor.

BitNet:

BitNet is a quantization technique that reduces the memory footprint and computational cost of the model.
In the BitNet module, a linear transformation is applied to the input tensor x using a fully connected layer (nn.Linear).
The sign function is then applied element-wise to the output of the linear transformation. The sign function maps positive values to 1 and non-positive values to -1, effectively quantizing the tensor to 1-bit precision.
The quantized tensor is then passed through another linear transformation to map it back to the original dimension.

Self-Decoder and Cross-Decoder:

The self-decoder and cross-decoder are based on the Transformer decoder architecture.
In the SelfDecoder and CrossDecoder modules, multiple Transformer decoder layers (nn.TransformerDecoderLayer) are stacked on top of each other.
Each decoder layer consists of multi-head self-attention, cross-attention (in the case of CrossDecoder), and feedforward neural network sublayers.
The self-attention sublayer allows the model to attend to different positions within the input sequence, capturing dependencies and relationships.
The cross-attention sublayer (in CrossDecoder) attends to the output of the self-decoder, enabling the model to fuse information from different sources.

Loss Function:

The code uses the cross-entropy loss (nn.CrossEntropyLoss) as the objective function for training the model.
Cross-entropy loss measures the dissimilarity between the predicted probability distribution and the true probability distribution.
It is commonly used in classification tasks, where the goal is to minimize the difference between the predicted class probabilities and the true class labels.

Optimization:

The code uses the Adam optimizer (optim.Adam) for updating the model's parameters during training.
Adam is an adaptive optimization algorithm that computes individual learning rates for different parameters based on their historical gradients and second moments.
It combines the benefits of AdaGrad and RMSProp algorithms, adapting the learning rates dynamically to achieve faster convergence.

Learning Rate Scheduling:

The code employs a learning rate scheduler (optim.lr_scheduler.StepLR) to adjust the learning rate during training.
The learning rate is multiplied by a factor (gamma) every step_size epochs.
Decreasing the learning rate over time allows the model to fine-tune its parameters and converge to a better solution.
