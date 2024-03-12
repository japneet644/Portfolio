# Optimizing PyTorch Training and Inference with Intel® Extension for PyTorch*

Deep learning practitioners are constantly seeking ways to optimize their training and inference workflows to achieve faster performance and better efficiency. With the introduction of the Intel® Extension for PyTorch*, developers can now leverage optimizations specifically designed for Intel® hardware, including CPUs and Intel® Iris® Xe Graphics.

In this blog post, we'll explore how to harness the power of the Intel® Extension for PyTorch* to optimize both training and inference tasks. We'll cover key changes required to integrate the extension into your PyTorch code, provide examples for training on the CIFAR-10 dataset using ResNet50, and demonstrate inference with popular models like ResNet50 and BERT.

## Getting Started with Intel® Extension for PyTorch*

To begin optimizing your PyTorch workflows with the Intel® Extension, follow these steps:

1. Install the Intel® Extension for PyTorch* using the provided installation instructions.
2. Update your PyTorch code to incorporate the necessary changes for optimization.
3. Apply the `ipex.optimize` function to your model and optimizer objects to enable optimizations.
4. Utilize Auto Mixed Precision (AMP) with BFloat16 data type for improved performance.
5. Convert input tensors, loss criterion, and model to the XPU device for efficient processing.

## Training Example: CIFAR-10 Classification with ResNet50

Let's walk through a basic example of training a ResNet50 model on the CIFAR-10 dataset using the Intel® Extension for PyTorch*:

```python
# Import necessary libraries and modules
import torch
import torchvision
import intel_extension_for_pytorch as ipex

# Define hyperparameters
LR = 0.001
DOWNLOAD = True
DATA = 'datasets/cifar10/'

# Define data transformations and create data loader
transform = torchvision.transforms.Compose([
  torchvision.transforms.Resize((224, 224)),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(
  root=DATA,
  train=True,
  transform=transform,
  download=DOWNLOAD,
)
train_loader = torch.utils.data.DataLoader(
  dataset=train_dataset,
  batch_size=128
)

# Initialize model, criterion, and optimizer
model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
model.train()

# Apply optimizations for Float32
model = model.to("xpu")
criterion = criterion.to("xpu")
model, optimizer = ipex.optimize(model, optimizer=optimizer)

# Training loop
for batch_idx, (data, target) in enumerate(train_loader):
  data = data.to("xpu")
  target = target.to("xpu")
  optimizer.zero_grad()
  output = model(data)
  loss = criterion(output, target)
  loss.backward()
  optimizer.step()
  print(batch_idx)

# Save model checkpoint
torch.save({
   'model_state_dict': model.state_dict(),
   'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pth')

```



## Inference Example: ResNet50 and BERT Models
For inference tasks, we can apply similar optimizations to achieve efficient processing with Intel® hardware. Here's an example of performing inference with ResNet50 and BERT models:

```python
# Import necessary libraries and modules
import torch
from transformers import BertModel
import intel_extension_for_pytorch as ipex

# Load pre-trained models
model_resnet50 = torchvision.models.resnet50()
model_bert = BertModel.from_pretrained("bert-base-uncased")
model_resnet50.eval()
model_bert.eval()

# Apply optimizations for Float32
model_resnet50 = model_resnet50.to("xpu")
model_bert = model_bert.to("xpu")
model_resnet50 = ipex.optimize(model_resnet50)
model_bert = ipex.optimize(model_bert)

# Perform inference with ResNet50
data_resnet50 = torch.rand(1, 3, 224, 224).to("xpu")
with torch.no_grad():
  model_resnet50(data_resnet50)

# Perform inference with BERT
vocab_size = model_bert.config.vocab_size
seq_length = 512
data_bert = torch.randint(vocab_size, size=[1, seq_length]).to("xpu")
with torch.no_grad():
  model_bert(data_bert)
``` 
