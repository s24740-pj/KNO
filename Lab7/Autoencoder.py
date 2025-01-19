import matplotlib.pyplot as plt
import numpy as np
from keras import layers, losses, Model, preprocessing, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():

    img_size = 64

    dataset = preprocessing.image_dataset_from_directory(
        './Images/Animals',
        image_size=(img_size, img_size),
        color_mode='rgb',
        batch_size=4,
        label_mode=None,
    )

    x_train = np.concatenate([x.numpy() for x in dataset], axis=0) / 255.0

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    datagen.fit(x_train)

    class Autoencoder(Model):
        def __init__(self, latent_dim, shape):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim
            self.shape = shape
            self.encoder = Sequential([
                layers.Conv2D(32, 5, data_format="channels_last", activation='relu', padding='same', strides=2),
                layers.BatchNormalization(),
                layers.Conv2D(64, 3, data_format="channels_last", activation='relu', padding='same', strides=2),
                layers.BatchNormalization(),
                layers.Conv2D(128, 3, data_format="channels_last", activation='relu', padding='same', strides=2),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                # layers.Dropout(0.2),
                layers.Dense(latent_dim, activation='tanh'),
            ])
            # Decoder with Batch Normalization
            self.decoder = Sequential([
                layers.Dense(128, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(img_size * img_size * 3, activation='relu'),
                layers.Reshape([img_size, img_size, 3]),
                layers.Conv2DTranspose(64, 3, data_format="channels_last", activation='relu', padding="same"),
                layers.BatchNormalization(),
                layers.Conv2DTranspose(32, 3, data_format="channels_last", activation='relu', padding="same"),
                layers.BatchNormalization(),
                layers.Conv2DTranspose(16, 3, data_format="channels_last", activation='relu', padding="same"),
                layers.Conv2DTranspose(3, 3, data_format="channels_last", activation='sigmoid', padding="same"),
                layers.Reshape(shape)
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    x_train = np.reshape(x_train, (x_train.shape[0], img_size, img_size, 3))

    shape = x_train.shape[1:]
    latent_dim = 2
    autoencoder = Autoencoder(latent_dim, shape)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(datagen.flow(x_train, x_train, batch_size=128),
                    epochs=2000,
                    shuffle=True)

    # encoded_imgs = np.random.normal(size=(100, latent_dim))
    # decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    #
    # decoded_imgs = np.clip(decoded_imgs, 0, 1)
    #
    # fig, axs = plt.subplots(10, 10)
    #
    # for y in range(10):
    #     for x in range(10):
    #         axs[y, x].imshow(decoded_imgs[y * 10 + x].reshape(img_size, img_size, 3))
    #         axs[y, x].axis('off')
    #
    # plt.show()

    # Pick 10 random images to compare
    indices = np.random.choice(len(x_train), 10, replace=False)
    selected_images = x_train[indices]

    # Reconstruct selected images
    reconstructed_images = autoencoder(selected_images).numpy()
    reconstructed_images = np.clip(reconstructed_images, 0, 1)

    # Create a 10-row, 2-column plot
    fig, axs = plt.subplots(2, 10, figsize=(20, 5))

    for i in range(10):
        # Show original
        axs[0, i].imshow(selected_images[i])
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')

        # Show reconstructed
        axs[1, i].imshow(reconstructed_images[i])
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
