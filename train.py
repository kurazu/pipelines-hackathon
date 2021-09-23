import logging

import tensorflow as tf

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
    )
    past = tf.keras.layers.LSTM(32, return_sequences=True)(stacked_past_features)
    past = tf.keras.layers.LSTM(16)(past)

    current = tf.keras.layers.Dense(16, activation="relu")(current_open)

    joined = tf.keras.layers.Concatenate()([past, current])
    output = tf.keras.layers.Dense(1)(joined)

    model = tf.keras.Model(
        inputs=[
            past_high,
            past_low,
            past_open,
            past_close,
            past_volume,
            past_adj_close,
            current_open,
        ],
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
    patience = 5
    train_input_file = (
        "gs://pipelines-hackathon/preprocessed_data/yahoo_stock.train.tfrecord"
    )
    val_input_file = (
        "gs://pipelines-hackathon/preprocessed_data/yahoo_stock.val.tfrecord"
    )
    test_input_file = (
        "gs://pipelines-hackathon/preprocessed_data/yahoo_stock.test.tfrecord"
    )
    train_ds = load_dataset(train_input_file, batch_size=batch_size, shuffle=True)
    val_ds = load_dataset(val_input_file)
    model = build_model()
    model.fit(
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

    test_ds = load_dataset(test_input_file)
    test_loss = model.evaluate(test_ds)
    print("test loss", test_loss)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
