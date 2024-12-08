import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler


class WineClassificationModel(tf.keras.Model):
    def __init__(self, units_1, units_2, num_classes=3):
        super(WineClassificationModel, self).__init__()
        self.layer_1 = tf.keras.layers.Dense(units_1, activation='relu', name='Layer_1')
        self.layer_2 = tf.keras.layers.Dense(units_2, activation='relu', name='Layer_2')
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name='Output_Layer')

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        return self.output_layer(x)

def model_builder(hp):
    units_1 = hp.Int('units_1', min_value=16, max_value=256, step=16)
    units_2 = hp.Int('units_2', min_value=8, max_value=128, step=8)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = WineClassificationModel(units_1=units_1, units_2=units_2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def main():
    """
        Description:
            Główna funkcja.
    """
    """
        Description:
            Zczytywanie danych z pliku.
    """
    wine_data = pd.read_csv("./wine/wine.data")
    wine_data.columns = [
        'Class',
        'Alcohol',
        'Malic acid',
        'Ash',
        'Alcalinity of ash',
        'Magnesium',
        'Total phenols',
        'Flavanoids',
        'Nonflavanoid phenols',
        'Proanthocyanins',
        'Color intensity',
        'Hue',
        'OD280/OD315 of diluted wines',
        'Proline'
    ]

    """
        Description:
            Tasowanie danych.
    """
    wine_data = wine_data.sample(frac=1, random_state=42).reset_index(drop=True)

    """
        Description:
            Podział na zbiór cech i etykiet.
    """
    wine_data['Class'] = wine_data['Class'].apply(lambda x: x - 1)
    X = wine_data.drop('Class', axis=1).values
    Y = wine_data['Class'].values

    # Skalowanie danych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    """
        Description:
            Tworzymy tuner do modelu.
    """
    tuner = kt.Hyperband(model_builder,
                     objective='accuracy',
                     max_epochs=1000,
                     factor=3,
                     directory='my_dir',
                     project_name='wine_classification')

    """
        Description:
            Funkcja do wczesnego zatrzymania. Pozwala szybko uzyskać model o wysokiej wydajności.
    """
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)

    """
        Description:
            Załączamy szukanie.
    """
    tuner.search(X, Y, epochs=1000, callbacks=[stop_early])

    """
        Description:
            Pobranie najlepszego modelu.
    """
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Najlepsze hiperparametry: units_1={best_hps.get('units_1')}, units_2={best_hps.get('units_2')}, learning_rate={best_hps.get('learning_rate')}")

    """
        Description:
            Trenowanie najlepszego modelu.
    """
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X, Y, epochs=1000)

    """
            Description:
                Evaluate modelu.
        """
    loss, accuracy = model.evaluate(X, Y)
    print(f"Dokładność na zbiorze: {accuracy:.4f}")

    """
        Description:
            Przetrenowanie modelu na najlepszej wartości epok.
    """
    val_acc_per_epoch = history.history['accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    hypermodel.fit(X, Y, epochs=best_epoch)

    eval_result = hypermodel.evaluate(X, Y)
    print("[test loss, test accuracy]:", eval_result)

if __name__ == '__main__':
    main()

