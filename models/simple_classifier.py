from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, ReLU

from models.base_model import BaseModel


class SimpleClassifier(BaseModel):
    def __init__(self, num_inputs: int = 784, num_hidden: int = 256, num_outputs: int = 10):
        super().__init__()

        # Define the model layers
        self.linear1 = Linear(in_features=num_inputs, out_features=num_hidden)
        self.relu = ReLU()
        self.linear2 = Linear(in_features=num_hidden, out_features=num_outputs)

        # Define the loss function
        self.loss = CrossEntropyLoss()

    def forward(self, x: [Tensor]) -> [Tensor]:
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
