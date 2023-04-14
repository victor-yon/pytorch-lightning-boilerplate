from torch import Tensor, log_softmax
from torch.nn import CrossEntropyLoss, Linear, ReLU

from models.base_model import BaseModel


class SimpleClassifier(BaseModel):
    def __init__(self, num_inputs: int = 784, num_hidden: int = 256, nb_class: int = 10,
                 learning_rate: float = 0.001):
        """
        Simple Neural Network multi-classes classifier.

        Args:
            num_inputs: The input size.
            num_hidden: The number of neurons in the hidden layer.
            nb_class: The number of classes.
            learning_rate: The learning rate to use for the optimizer.
        """
        super().__init__(learning_rate)

        # Define the model layers
        self.linear1 = Linear(in_features=num_inputs, out_features=num_hidden)
        self.relu = ReLU()
        self.linear2 = Linear(in_features=num_hidden, out_features=nb_class)

        # Define the loss function
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x: [Tensor]) -> [Tensor]:
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return log_softmax(x, dim=1)
