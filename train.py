import logging

import gcsfs
import tensorflow as tf
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

fs = gcsfs.GCSFileSystem(project="pipelines-hackathon")

HISTORY_LENGTH = 30

FEATURE_DESCRIPTION = {
    "past_high": tf.io.FixedLenFeature(shape=(HISTORY_LENGTH,), dtype=tf.float32),
    "past_low": tf.io.FixedLenFeature(shape=(HISTORY_LENGTH,), dtype=tf.float32),
    "past_open": tf.io.FixedLenFeature(shape=(HISTORY_LENGTH,), dtype=tf.float32),
    "past_close": tf.io.FixedLenFeature(shape=(HISTORY_LENGTH,), dtype=tf.float32),
    "past_volume": tf.io.FixedLenFeature(shape=(HISTORY_LENGTH,), dtype=tf.float32),
    "past_adj_close": tf.io.FixedLenFeature(shape=(HISTORY_LENGTH,), dtype=tf.float32),
    "open": tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32),
    "close": tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32),
}


def load_dataset(input_file: str, batch_size: int = 32, shuffle: bool = False):
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=input_file,
        batch_size=batch_size,
        features=FEATURE_DESCRIPTION,
        reader=tf.data.TFRecordDataset,
        shuffle=shuffle,
        num_epochs=1,
        prefetch_buffer_size=2,
        label_key="close",
    )


def build_model():
    past_high = tf.keras.Input((HISTORY_LENGTH,), name="past_high", dtype=tf.float32)
    past_low = tf.keras.Input((HISTORY_LENGTH,), name="past_low", dtype=tf.float32)
    past_open = tf.keras.Input((HISTORY_LENGTH,), name="past_open", dtype=tf.float32)
    past_close = tf.keras.Input((HISTORY_LENGTH,), name="past_close", dtype=tf.float32)
    past_volume = tf.keras.Input(
        (HISTORY_LENGTH,), name="past_volume", dtype=tf.float32
    )
    past_adj_close = tf.keras.Input(
        (HISTORY_LENGTH,), name="past_adj_close", dtype=tf.float32
    )
    # and a single point for current round data
    current_open = tf.keras.Input((1,), name="open", dtype=tf.float32)
    stacked_past_features = tf.stack(
        [past_high, past_low, past_open, past_close, past_volume, past_adj_close],
        axis=-1,
        name="stacked_past_features",
    )
    past = tf.keras.layers.LSTM(32, return_sequences=True, name="past_1")(
        stacked_past_features
    )
    past = tf.keras.layers.LSTM(16, name="past_2")(past)

    current = tf.keras.layers.Dense(16, activation="relu", name="current")(current_open)

    joined = tf.keras.layers.Concatenate(name="concat")([past, current])
    output = tf.keras.layers.Dense(1, name="net_output")(joined)

    model = tf.keras.Model(
        inputs={
            "past_high": past_high,
            "past_low": past_low,
            "past_open": past_open,
            "past_close": past_close,
            "past_volume": past_volume,
            "past_adj_close": past_adj_close,
            "open": current_open,
        },
        outputs=output,
    )
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        rankdir="LR",
    )
    return model


def main() -> None:
    batch_size = 32
    epochs = 100
    patience = 10
    train_input_file = (
        "gs://pipelines-hackathon/preprocessed_data/yahoo_stock.train.tfrecord"
    )
    val_input_file = (
        "gs://pipelines-hackathon/preprocessed_data/yahoo_stock.val.tfrecord"
    )
    test_input_file = (
        "gs://pipelines-hackathon/preprocessed_data/yahoo_stock.test.tfrecord"
    )
    model_path = "gs://pipelines-hackathon/models/simple"
    logger.info("Building model")
    model = build_model()
    logger.info("Training model")
    train_ds = load_dataset(train_input_file, batch_size=batch_size, shuffle=True)
    val_ds = load_dataset(val_input_file)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ],
    )
    logger.info("Saving model")
    model.save(model_path)

    logger.info("Plotting learning curves")
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    with fs.open("gs://pipelines-hackathon/models/simple_train.png", "wb") as f:
        plt.savefig(f, format="png")

    logger.info("Restoring model")
    model = tf.keras.models.load_model(model_path)
    logger.info("Evaluating model")
    test_ds = load_dataset(test_input_file)
    test_loss = model.evaluate(test_ds)
    logger.info("Test loss %.4f", test_loss)
    logger.info("Storing test loss")
    with fs.open(
        "gs://pipelines-hackathon/models/simple_train.txt", "w", encoding="utf-8"
    ) as f:
        f.write(f"Test loss: {test_loss}")
    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    main()
