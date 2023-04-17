from typing import Optional

from lightning.pytorch import LightningModule
from torch import Tensor, argmax
from torch.optim import Adam
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MetricCollection, Precision, Recall

from plots.test_results import plot_test_results


class BaseModel(LightningModule):
    """
    Neural Network classifier model.
    """

    def __init__(self, learning_rate: float = 0.001):
        """
        Create a class that define the model logic.

        Args:
            learning_rate: The learning rate to use for the optimizer.
        """
        super().__init__()
        self.learning_rate = learning_rate

        # TODO add/edit metrics
        # See https://torchmetrics.readthedocs.io/en/latest/ for available metrics
        self.metrics = MetricCollection({
            'acc': Accuracy('multiclass', num_classes=10),
            'precision': Precision('multiclass', num_classes=10),
            'recall': Recall('multiclass', num_classes=10),
            'f1': F1Score('multiclass', num_classes=10),
        })
        # Confusion matrix is separated because we want to compute it for test only
        self.metrics_cm = ConfusionMatrix('multiclass', num_classes=10, normalize='true')

        # The loss function should be defined in the child classes
        self.loss_fn = None

    def forward(self, x: [Tensor]) -> [Tensor]:
        raise NotImplementedError('This method should be implemented in the child class')

    def evaluate(self, batch, stage: Optional[str] = None):
        x, y = batch
        logits = self(x)

        if stage:
            self.metrics(logits, y, stage=stage)
            if stage == 'test':
                self.metrics_cm(logits, y)

            # Log metrics and update progress bar
            loss = self.loss_fn(logits, y)
            self.log(f'{stage}_loss', loss, prog_bar=True)
            for label, metric in self.metrics.items():
                self.log(f'{stage}_{label}', metric, prog_bar=True)

        return argmax(logits, dim=1)

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        logits = self(x)  # Forward
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage='val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')

    def on_test_end(self) -> None:
        plot_test_results(self.metrics, self.metrics_cm)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
