import os

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.data import ToTensor
from torch import nn, optim
from torch.utils.data import DataLoader

from ViTModel import ViT

model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

train_data = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
valid_data = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())
loaders = {
    "train": DataLoader(train_data, batch_size=32),
    "valid": DataLoader(valid_data, batch_size=32),
}

runner = dl.SupervisedRunner(
    input_key="features",
    output_key="logits",
    target_key="targets",
    loss_key="loss"
)

# model training
# noinspection PyTypeChecker
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=5,
    callbacks=[
        dl.AccuracyCallback(
            input_key="logits", target_key="targets", topk_args=(1, 3, 5)),
        dl.PrecisionRecallF1SupportCallback(
            input_key="logits", target_key="targets", num_classes=10
        ),
    ],
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
    load_best_on_end=True,
)

# model evaluation
# noinspection PyTypeChecker
metrics = runner.evaluate_loader(
    loader=loaders["valid"],
    callbacks=[dl.AccuracyCallback(input_key="logits",
                                   target_key="targets",
                                   topk_args=(1, 3, 5))],
)
assert "accuracy" in metrics.keys()
