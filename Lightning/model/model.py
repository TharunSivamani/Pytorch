import torch
import torchvision
import numpy
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from config import LEARNING_RATE
from torch import nn, optim
import torchmetrics
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torch_lr_finder import LRFinder
from utils.metrics import MyAccuracy
import matplotlib.pyplot as plt
from utils.utils import plot_images,plot_graphs,plot_train_images


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv_layer1 = ConvBlock(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.conv_layer2 = ConvBlock(
            out_channels, out_channels, kernel_size, stride, padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        res = residual + out
        return res


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        return x


class MakeLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        residual_block=None,
    ):
        super().__init__()
        self.residual_block = residual_block
        self.downsample_layer = DownsampleLayer(
            in_channels, out_channels, kernel_size, padding, stride
        )
        if residual_block:
            self.residual_layer = ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_layer(x)
        if self.residual_block:
            x = self.residual_layer(x)
        return x


class NN(pl.LightningModule):
    def __init__(self, lr=LEARNING_RATE, num_classes=10, max_epochs=24):
        super().__init__()
        self.network = nn.Sequential(
            ConvBlock(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            MakeLayer(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                residual_block=ResidualBlock,
            ),
            MakeLayer(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                residual_block=None,
            ),
            MakeLayer(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                residual_block=ResidualBlock,
            ),
            nn.MaxPool2d(kernel_size=4, stride=1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = lr
        self.train_accuracy = MyAccuracy()
        self.val_accuracy = MyAccuracy()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.max_epochs = max_epochs
        self.epoch_counter = 1

        self.classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        self.training_acc = list()
        self.training_loss = list()
        self.testing_acc = list()
        self.testing_loss = list()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.epoch_counter == 1:
            if not hasattr(self, "images_plotted"):
                self.images_plotted = True

                scores = self.forward(x)
                preds = torch.argmax(scores, dim=1)

                self.train_images = []
                self.train_predictions = []
                self.train_labels = []

                for i in range(20):
                    x, target = batch

                    output = self.forward(x)

                    _, preds = torch.max(output, 1)

                    for i in range(len(preds)):
                        if preds[i] != target[i]:
                            self.train_images.append(x[i])
                            self.train_predictions.append(preds[i])
                            self.train_labels.append(target[i])

                plot_train_images(
                    self.train_images,
                    self.train_labels,
                    self.classes,
                )
                print("\n")
            return self._common_step(batch, self.train_loss, self.train_accuracy)

        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 3, 32, 32))
            self.logger.experiment.add_image("cifar10_images", grid, self.global_step)

        return self._common_step(batch, self.train_loss, self.train_accuracy)

    def on_train_epoch_end(self):
        self.training_acc.append(self.train_accuracy.compute())
        self.training_loss.append(self.train_loss.compute())
        print(
            f"Epoch: {self.epoch_counter}, Train: Loss: {self.train_loss.compute():0.4f}, Accuracy: "
            f"{self.train_accuracy.compute():0.2f}"
        )
        print("\n")
        self.train_loss.reset()
        self.train_accuracy.reset()
        self.epoch_counter += 1

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, self.val_loss, self.val_accuracy)

        self.log("val_step_loss", self.val_loss, prog_bar=True, logger=True)
        self.log("val_step_acc", self.val_accuracy, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self):
        self.testing_acc.append(self.val_accuracy.compute())
        self.testing_loss.append(self.val_loss.compute())
        print(
            f"Epoch: {self.epoch_counter}, Valid: Loss: {self.val_loss.compute():0.4f}, Accuracy: "
            f"{self.val_accuracy.compute():0.2f}"
        )
        self.val_loss.reset()
        self.val_accuracy.reset()

        if(self.epoch_counter == self.max_epochs):
            if not hasattr(self,"graphs_plotted"):
                self.graphs_plotted = True
                
                train_acc_cpu = [acc.item() for acc in self.training_acc]
                train_loss_cpu = [acc.item() for acc in self.training_loss]
                test_acc_cpu = [acc.item() for acc in self.testing_acc]
                test_loss_cpu = [acc.item() for acc in self.testing_loss]

                plot_graphs(train_loss_cpu, train_acc_cpu, test_loss_cpu, test_acc_cpu)

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, self.val_loss, self.val_accuracy)
        self.log("test_loss", loss)

        return loss

    def _common_step(self, batch, loss_metric, acc_metric):
        x, y = batch
        batch_len = y.numel()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        loss_metric.update(loss, batch_len)
        acc_metric.update(logits, y)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        # x = x.reshape(x.size(0),-1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)

        self.images = []
        self.predictions = []
        self.labels = []

        for i in range(20):
            x, target = batch

            output = self.forward(x)

            _, preds = torch.max(output, 1)

            for i in range(len(preds)):
                if preds[i] != target[i]:
                    self.images.append(x[i])
                    self.predictions.append(preds[i])
                    self.labels.append(target[i])

        return self.images, self.predictions, self.labels

    def train_dataloader(self):

        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader

    def find_lr(self, optimizer):
        lr_finder = LRFinder(self, optimizer, self.criterion)
        lr_finder.range_test(
            self.train_dataloader(), end_lr=0.1, num_iter=100, step_mode="exp"
        )
        _, best_lr = lr_finder.plot()
        lr_finder.reset()
        return best_lr

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-7, weight_decay=1e-2)
        best_lr = self.find_lr(optimizer)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=best_lr,
            epochs=self.max_epochs,
            pct_start=5 / self.max_epochs,
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
