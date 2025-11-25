
import tensorflow as tf
import time

def build_model(genome):
    num_layers = genome[0]
    units = genome[1:1+num_layers]
    activation_choice = genome[-1]
    activations = ["relu", "tanh", "sigmoid"]

    model = tf.keras.Sequential([tf.keras.layers.Flatten()])
    for u in units:
        model.add(tf.keras.layers.Dense(u, activation=activations[activation_choice]))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def evaluate(genome, train_X, train_y, test_X, test_y):
    model = build_model(genome)

    model.fit(train_X, train_y, epochs=1, batch_size=32, verbose=0)

    loss, acc = model.evaluate(test_X, test_y, verbose=0)

    params = model.count_params()

    start = time.time()
    model.predict(test_X[:1])
    latency = time.time() - start

    return -acc, params, latency
