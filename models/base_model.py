from typing import Optional

from pytorch_lightning import LightningModule
from torch import Tensor, argmax
from torch.optim import Adam
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from plots.test_results import plot_test_results


class BaseModel(LightningModule):
    """
    Abstract class that define the model logic.
    """

    def __init__(self):
        super().__init__()

        # TODO add/edit metrics
        # See https://torchmetrics.readthedocs.io/en/latest/ for available metrics
        self.metrics = MetricCollection({
            'acc': Accuracy('multiclass', num_classes=10),
            'precision': Precision('multiclass', num_classes=10),
            'recall': Recall('multiclass', num_classes=10),
            'f1': F1Score('multiclass', num_classes=10),
        })

        self.loss_fn = None

    def forward(self, x: [Tensor]) -> [Tensor]:
        raise NotImplementedError('This method should be implemented in the child class')

    def evaluate(self, batch, stage: Optional[str] = None):
        x, y = batch
        logits = self(x)

        if stage:
            self.metrics(logits, y, stage=stage)

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

    def test_epoch_end(self, _) -> None:
        plot_test_results(self.metrics)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
