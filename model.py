import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images, **kwargs):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim * 2)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim * 2
        )

    def call(self, patch, **kwargs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
        return x


class VisionTransformer(tf.keras.Model):
    def __init__(self, inputshape, patch_size, num_patches, projection_dim,
                 transformer_layers, num_heads, transformer_units,
                 mlp_head_units, num_classes):
        super(VisionTransformer, self).__init__()
        self.inputshape = inputshape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.mlp_head_units = mlp_head_units
        self.num_classes = num_classes

    def call(self, input, training, **kwargs):
        inputs = layers.Input(shape=self.inputshape)
        patches = Patches(self.patch_size)(inputs)
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(
            patches)

        for _ in range(self.transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim,
                dropout=0.1)(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])

        representation = layers.LayerNormalization(epsilon=1e-6)(
            encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)

        features = mlp(representation, hidden_units=self.mlp_head_units,
                       dropout_rate=0.5)
        logits = layers.Dense(self.num_classes)(features)
        model = keras.Model(inputs=inputs, outputs=logits)
        return model
