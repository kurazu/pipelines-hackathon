import tensorflow as tf


def read_ds(filename) -> None:
    raw_dataset = tf.data.TFRecordDataset(filename)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)


if __name__ == "__main__":
    print("TRAIN")
    read_ds(
        "gs://pipelines-hackathon/preprocessed_data/yahoo_stock.beam.train.tfrecord-00000-of-00001"
    )
    print("TEST")
    read_ds(
        "gs://pipelines-hackathon/preprocessed_data/yahoo_stock.beam.test.tfrecord-00000-of-00001"
    )
