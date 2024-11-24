import argparse
import datetime
import pandas as pd
import tensorflow as tf
import sklearn.model_selection as ms
import numpy as np

def main():
    # 2. Zczytywanie danych i przypisanie kolumn
    wine_data = pd.read_csv("./wine/wine.data")
    wine_data.columns = [
        'Class',
        'Alcohol',
        'Malic acid',
        'Ash',
        'Alcalinity of ash  ',
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
    # print(wine_data)

    # 3. Potasowanie danych
    wine_data = wine_data.sample(frac=1, random_state=42).reset_index(drop=True)
    # print(wine_data)

    # 4. Postać kodowania z gorącą jedynką
    wine_data_hotone = pd.get_dummies(wine_data,columns=['Class'], dtype=int)
    # print(wine_data_hotone.head(10))

    # 6. Przygotowanie modeli sieci neuronowej wraz z metodą uczenia

    # Podział na zbiór cech i etykiet
    wine_data['Class'] = wine_data['Class'].apply(lambda x: x - 1)
    X = wine_data.drop('Class', axis=1).values
    Y = wine_data['Class'].values

    X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.25, random_state=30)

    # Model1
    model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), name='Layer_1'),
        tf.keras.layers.Dense(32, activation='relu', name='Layer_2'),
        tf.keras.layers.Dense(3, activation='softmax', name='Output_Layer')
    ])

    model1.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model1.summary()

    log_dir1 = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_model1"
    tensorboard_callback1 = tf.keras.callbacks.TensorBoard(log_dir=log_dir1, histogram_freq=1)

    history1 = model1.fit(
        X_train,
        Y_train,
        epochs=50,
        validation_data=(X_test, Y_test),
        callbacks=[tensorboard_callback1])

    # Ocena model1
    EV1 = model1.evaluate(X_test, Y_test)

    # Model2
    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), name='Layer_1'),
        tf.keras.layers.Dense(64, activation='relu', name='Layer_2'),
        tf.keras.layers.Dropout(0.5, name='Dropout_Layer'),
        tf.keras.layers.Dense(32, activation='relu', name='Layer_3'),
        tf.keras.layers.Dense(16, activation='softmax', name='Layer_4'),
        tf.keras.layers.Dense(3, activation='softmax', name='Output_Layer')
    ])

    model2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model2.summary()

    log_dir2 = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_model2"
    tensorboard_callback2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir2, histogram_freq=1)

    history2 = model2.fit(
        X_train,
        Y_train,
        epochs=50,
        validation_data=(X_test, Y_test),
        callbacks=[tensorboard_callback2])

    # Ocena model2
    EV2 = model2.evaluate(X_test, Y_test)

    # Porównanie
    # [0.5741689205169678, 0.800000011920929]
    # Strata, Dokładność
    print("Model1 - EV1:", EV1)
    print("Model2 - EV2:", EV2)

    def predict_wine(features, model):
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        return predicted_class

    parser = argparse.ArgumentParser(description='Przewidywanie kategorii wina.')
    parser.add_argument('wine_params', type=str,
                        help='Parametry wina w formacie: 11.81 2.12 2.74 21.5 134 1.6 .99 .14 1.56 2.5 .95 2.26 625')

    args = parser.parse_args()
    wine_params = list(map(float, args.wine_params.split()))
    predicted_class = predict_wine(wine_params, model1)
    print(f"Przewidywana kategoria wina: {predicted_class + 1}")

if __name__ == '__main__':
    main()

