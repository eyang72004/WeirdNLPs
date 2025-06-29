import torch
import torch.nn as nn

# A simple classifier for demonstration
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=6, num_classes=2):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# Instantiate and eval mode
model = SimpleClassifier()
model.eval()

# Trace the model using example input
example_input = torch.randint(0, 100, (1, 6)).float()  # Simulate a tensor of shape [1, 6]
traced_model = torch.jit.trace(model, example_input)

# Save the traced model
traced_model.save("example_model.pt")
print("Saved model to example_model.pt")
