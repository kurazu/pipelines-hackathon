import logging
import os

import click
import tensorflow as tf

logger = logging.getLogger(__name__)


def get_feature_description(history_length: int):
    return {
        "past_high": tf.io.FixedLenFeature(shape=(history_length,), dtype=tf.float32),
        "past_low": tf.io.FixedLenFeature(shape=(history_length,), dtype=tf.float32),
        "past_open": tf.io.FixedLenFeature(shape=(history_length,), dtype=tf.float32),
        "past_close": tf.io.FixedLenFeature(shape=(history_length,), dtype=tf.float32),
        "past_volume": tf.io.FixedLenFeature(shape=(history_length,), dtype=tf.float32),
        "past_adj_close": tf.io.FixedLenFeature(
            shape=(history_length,), dtype=tf.float32
        ),
        "open": tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32),
        "close": tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32),
    }


def load_dataset(
    input_file: str, history_length: int, batch_size: int = 32, shuffle: bool = False
):
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=input_file,
        batch_size=batch_size,
        features=get_feature_description(history_length),
        reader=tf.data.TFRecordDataset,
        shuffle=shuffle,
        num_epochs=1,
        prefetch_buffer_size=2,
        label_key="close",
    )


def build_model(history_length: int):
    past_high = tf.keras.Input((history_length,), name="past_high", dtype=tf.float32)
    past_low = tf.keras.Input((history_length,), name="past_low", dtype=tf.float32)
    past_open = tf.keras.Input((history_length,), name="past_open", dtype=tf.float32)
    past_close = tf.keras.Input((history_length,), name="past_close", dtype=tf.float32)
    past_volume = tf.keras.Input(
        (history_length,), name="past_volume", dtype=tf.float32
    )
    past_adj_close = tf.keras.Input(
        (history_length,), name="past_adj_close", dtype=tf.float32
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

    return model


@click.command()
@click.option(
    "--train-ds",
    "train_input_file",
    type=str,
    required=True,
    default="gs://pipelines-hackathon/preprocessed_data/yahoo_stock.train.tfrecord",
)
@click.option(
    "--val-ds",
    "val_input_file",
    type=str,
    required=True,
    default="gs://pipelines-hackathon/preprocessed_data/yahoo_stock.val.tfrecord",
)
@click.option(
    "--test-ds",
    "test_input_file",
    type=str,
    required=True,
    default="gs://pipelines-hackathon/preprocessed_data/yahoo_stock.test.tfrecord",
)
@click.option("--batch-size", type=int, required=True, default=32)
@click.option("--epochs", type=int, required=True, default=100)
@click.option("--patience", type=int, required=True, default=10)
@click.option("--history-length", type=int, required=True, default=30)
def main(
    train_input_file: str,
    val_input_file: str,
    test_input_file: str,
    batch_size: int,
    epochs: int,
    patience: int,
    history_length: int,
) -> None:
    model_artifact_dir = os.environ["AIP_MODEL_DIR"]
    checkpoint_dir = os.environ["AIP_CHECKPOINT_DIR"]
    tensorboard_log_dir = os.environ["AIP_TENSORBOARD_LOG_DIR"]

    logger.info("Building model")
    model = build_model(history_length=history_length)
    logger.info("Training model")
    train_ds = load_dataset(
        train_input_file,
        history_length=history_length,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = load_dataset(val_input_file, history_length=history_length)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_log_dir, histogram_freq=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ],
    )
    logger.info("Saving model")
    model.save(model_artifact_dir)

    logger.info("Evaluating model")
    test_ds = load_dataset(test_input_file, history_length=history_length)
    test_loss = model.evaluate(test_ds)
    logger.info("Test loss %.4f", test_loss)
    tf.summary.scalar("test_loss", data=test_loss, step=epochs)
    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s][%(levelname)-7s][%(name)s] %(message)s",
    )
    # Make matplotlib shut up about fonts
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    main()
