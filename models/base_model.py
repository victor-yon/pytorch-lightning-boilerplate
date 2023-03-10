from typing import Optional

from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import MetricCollection

from plots.test_results import plot_test_results


class BaseModel(LightningModule):
    """
    Abstract class that define the model logic.
    """

    def __init__(self):
        super().__init__()

        self.metrics = MetricCollection({})  # TODO add metrics

        raise NotImplementedError  # TODO implement model

    def forward(self, x: [Tensor]) -> [Tensor]:
        raise NotImplementedError('This method should be implemented in the child class')

    def evaluate(self, batch, stage: Optional[str] = None):
        x, y = batch
        logits = self(x)
        loss = None  # TODO compute loss

        if stage:
            self.metrics(logits, y, stage=stage)

            # Log metrics and update progress bar
            self.log(f'{stage}_loss', loss, prog_bar=True)
            for label, metric in self.metrics.items():
                self.log(f'{stage}_{label}', metric, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage='val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')

    def test_epoch_end(self, _) -> None:
        plot_test_results(self.metrics)
