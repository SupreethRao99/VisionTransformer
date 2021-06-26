import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow_datasets as tfds
from model import VisionTransformer
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

num_classes = 10
inputshape = (32, 32, 3)
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 1
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8
mlp_head_units = [2048, 1024]

ds = tfds.load("cifar10", as_supervised=True)

ds_train = (ds["train"].cache().shuffle(5 * batch_size).batch(batch_size)
            .prefetch(AUTOTUNE))
ds_test = (ds["test"].cache().batch(batch_size).prefetch(AUTOTUNE))

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = VisionTransformer(
        inputshape,
        patch_size,
        num_patches,
        projection_dim,
        transformer_layers,
        num_heads,
        transformer_units,
        mlp_head_units,
        num_classes
    )

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=[
                      keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                      keras.metrics.SparseTopKCategoricalAccuracy
                      (5, name="top-5-accuracy"), ],
                  )

    history = model.fit(
        ds_train,
        validation_data=ds_train,
        batch_size=batch_size,
        epochs=num_epochs,
    )
