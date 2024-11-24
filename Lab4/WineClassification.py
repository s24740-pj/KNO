import datetime
import pandas as pd
import tensorflow as tf
import sklearn.model_selection as ms

def main():
    # Zczytywanie danych i przypisanie kolumn
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

    # Potasowanie danych
    wine_data = wine_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Podział na zbiór cech i etykiet
    wine_data['Class'] = wine_data['Class'].apply(lambda x: x - 1)
    X = wine_data.drop('Class', axis=1).values
    Y = wine_data['Class'].values

    X_train, X_temp, Y_train, Y_temp = ms.train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = ms.train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    def create_simple_model(units_1, units_2, learning_rate):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units_1, activation='relu', input_shape=(13,), name='Layer_1'),
            tf.keras.layers.Dense(units_2, activation='relu', name='Layer_2'),
            tf.keras.layers.Dense(3, activation='softmax', name='Output_Layer')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_complex_model(units_1, units_2, units_3, learning_rate):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units_1, activation='relu', input_shape=(13,), name='Layer_1'),
            tf.keras.layers.Dense(units_2, activation='relu', name='Layer_2'),
            tf.keras.layers.Dense(units_3, activation='relu', name='Layer_3'),
            tf.keras.layers.Dense(3, activation='softmax', name='Output_Layer')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # Mój model z poprzednich zajęć
    model1 = create_simple_model(64, 32, 0.01)

    # TensorBoard
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_model_baseline"
    baseline_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model1.fit(X_train, Y_train, verbose=1, epochs=50, batch_size=32, callbacks=[baseline_tensorboard_callback])

    validation_loss, validation_accuracy = model1.evaluate(X_val, Y_val)
    test_loss, test_accuracy = model1.evaluate(X_test, Y_test)

    baseline = {
        "model": "model1",
        "units_1": 64,
        "units_2": 32,
        "learning_rate": 0.01,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

    print("Model1 - Wyniki bazowe (baseline):")
    print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    best_model = {
        "model": None,
        "units_1": None,
        "units_2": None,
        "units_3": None,
        "learning_rate": None,
        "val_accuracy": 0,
        "test_accuracy": 0
    }

    # Definicja hiperparametrów do przetestowania
    experiments = [
        {"model": "simple", "units_1": 64, "units_2": 32, "learning_rate": 0.01},
        {"model": "simple", "units_1": 128, "units_2": 64, "learning_rate": 0.005},
        {"model": "complex", "units_1": 128, "units_2": 64, "units_3": 32, "learning_rate": 0.001},
        {"model": "complex", "units_1": 256, "units_2": 128, "units_3": 64, "learning_rate": 0.0005}
    ]

    # Iteracja po konfiguracjach
    for experiment in experiments:
        if experiment["model"] == "simple":
            model = create_simple_model(experiment["units_1"], experiment["units_2"],
                                        experiment["learning_rate"])
        elif experiment["model"] == "complex":
            model = create_complex_model(experiment["units_1"], experiment["units_2"],
                                         experiment["units_3"], experiment["learning_rate"])

        # TensorBoard
        log_dir2 = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_model_experiments"
        experiments_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir2, histogram_freq=1)

        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, verbose=1, batch_size=32, callbacks=[experiments_tensorboard_callback])

        test_loss, test_accuracy = model.evaluate(X_test, Y_test)
        print(f"Test loss: {test_loss:.4f}; Test Accuracy: {test_accuracy:.4f}; Model: {experiment};")

        validation_loss, validation_accuracy = model.evaluate(X_val, Y_val)
        print(f"Val loss: {validation_loss:.4f}; Val Accuracy: {validation_accuracy:.4f}; Model: {experiment};")

        if validation_accuracy > best_model["val_accuracy"]:
            best_model.update({
                "model": experiment["model"],
                "units_1": experiment["units_1"],
                "units_2": experiment["units_2"],
                "units_3": experiment.get("units_3", None),
                "learning_rate": experiment["learning_rate"],
                "val_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy
            })

    print("\nModel z poprzednich zajęć:")
    print(f"Model: {baseline['model']}")
    print(f"Units_1: {baseline['units_1']}")
    print(f"Units_2: {baseline['units_2']}")
    print(f"Learning Rate: {baseline['learning_rate']}")
    print(f"Validation Accuracy: {baseline['validation_accuracy']:.4f}")
    print(f"Test Accuracy: {baseline['test_accuracy']:.4f}")

    print("\nNajlepszy model z testowanych:")
    print(f"Model: {best_model['model']}")
    print(f"Units_1: {best_model['units_1']}")
    print(f"Units_2: {best_model['units_2']}")
    print(f"Units_3: {best_model['units_3']}")
    print(f"Learning Rate: {best_model['learning_rate']}")
    print(f"Validation Accuracy: {best_model['val_accuracy']:.4f}")
    print(f"Test Accuracy: {best_model['test_accuracy']:.4f}")

if __name__ == '__main__':
    main()

