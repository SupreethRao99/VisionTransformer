import os
import yaml
from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.data import ToTensor
from torch import nn, optim
from torch.utils.data import DataLoader

from ViTModel import ViT

with open('config.yaml', 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

model = ViT(image_size=config['image_size'],
            patch_size=config['patch_size'],
            num_classes=config['num_classes'],
            channels=config['channels'],
            dim=config['dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

train_data = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
valid_data = MNIST(os.getcwd(), train=False, download=True,
                   transform=ToTensor())
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
            input_key="logits", target_key="targets",
            num_classes=config['num_classes']
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
