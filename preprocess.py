import logging
import tempfile
from typing import Dict

import apache_beam as beam
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tfx_bsl.coders.example_coder import RecordBatchToExamples
from tfx_bsl.public import tfxio

RAW_DATA_FEATURE_SPEC = {
    "Date": tf.io.FixedLenFeature((), tf.string),
    "High": tf.io.FixedLenFeature((), tf.float32),
    "Low": tf.io.FixedLenFeature((), tf.float32),
    "Open": tf.io.FixedLenFeature((), tf.float32),
    "Close": tf.io.FixedLenFeature((), tf.float32),
    "Volume": tf.io.FixedLenFeature((), tf.float32),
    "Adj Close": tf.io.FixedLenFeature((), tf.float32),
}

SCHEMA = tft.tf_metadata.dataset_metadata.DatasetMetadata(
    tft.tf_metadata.schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC)
).schema

ORDERED_CSV_COLUMNS = ["Date", "High", "Low", "Open", "Close", "Volume", "Adj Close"]


def preprocessing_fn(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Preprocess input columns into transformed columns."""
    # parsed_date = tfa.text.parse_time(inputs["Date"], "%Y-%m-%d", "SECOND")
    # scaled_date = tft.scale_to_0_1(parsed_date)
    # # 0 - 0.8 => train
    # # 0.8 - 0.9 => validation
    # # 0.9 - 1.0 => test
    # bucket_boundaries = tf.constant([[0.8, 0.9]])
    # bucketized_date = tft.apply_buckets(scaled_date, bucket_boundaries)
    # subset = tf.where(
    #     bucketized_date == 0,
    #     tf.constant("train", dtype=tf.string),
    #     tf.where(
    #         bucketized_date == 1,
    #         tf.constant("val", dtype=tf.string),
    #         tf.constant("test", dtype=tf.string),
    #     ),
    # )
    return {
        "high": tft.scale_to_z_score(inputs["High"]),
        "low": tft.scale_to_z_score(inputs["Low"]),
        "open": tft.scale_to_z_score(inputs["Open"]),
        "close": tft.scale_to_z_score(inputs["Close"]),
        "volume": tft.scale_to_z_score(inputs["Volume"]),
        "adj_close": tft.scale_to_z_score(inputs["Adj Close"]),
        # "subset": subset,
        "raw_close": inputs["Close"],
    }


def transform_data(
    *,
    train_source_file: str,
    train_target_file: str,
    test_files: Dict[str, str],
    transform_target_dir: str
) -> None:
    """Transform the data and write out as a TFRecord of Example protos.

    Read in the data using the CSV reader, and transform it using a
    preprocessing pipeline that scales numeric data and converts categorical data
    from strings to int64 values indices, by creating a vocabulary for each
    category.

    Args:
      train_data_file: File containing training data
      test_data_file: File containing test data
      working_dir: Directory to write transformed data and metadata to
    """

    # The "with" block will create a pipeline, and run that pipeline at the exit
    # of the block.
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
            # Create a TFXIO to read the census data with the schema. To do this we
            # need to list all columns in order since the schema doesn't specify the
            # order of columns in the csv.
            train_csv_tfxio = tfxio.CsvTFXIO(
                train_source_file,
                column_names=ORDERED_CSV_COLUMNS,
                schema=SCHEMA,
                skip_header_lines=1,
                telemetry_descriptors=["train"],
            )
            raw_data = pipeline | "ToRecordBatches" >> train_csv_tfxio.BeamSource()

            # Combine data and schema into a dataset tuple.  Note that we already used
            # the schema to read the CSV data, but we also need it to interpret
            # raw_data.
            raw_dataset = (raw_data, train_csv_tfxio.TensorAdapterConfig())

            # The TFXIO output format is chosen for improved performance.
            (
                transformed_dataset,
                transform_fn,
            ) = raw_dataset | tft_beam.AnalyzeAndTransformDataset(
                preprocessing_fn, output_record_batches=True
            )

            # Transformed metadata is not necessary for encoding.
            transformed_data, _ = transformed_dataset

            # Extract transformed RecordBatches, encode and write them to the given
            # directory.
            _ = (
                transformed_data
                | "EncodeTrainData"
                >> beam.FlatMapTuple(lambda batch, _: RecordBatchToExamples(batch))
                | "WriteTrainData" >> beam.io.WriteToTFRecord(train_target_file)
            )

            assert test_files, "At least on test file needed"
            for test_source_file, test_target_file in test_files.items():
                # Now apply transform function to test data.  In this case we remove the
                test_csv_tfxio = tfxio.CsvTFXIO(
                    test_source_file,
                    column_names=ORDERED_CSV_COLUMNS,
                    schema=SCHEMA,
                    skip_header_lines=1,
                    telemetry_descriptors=["val", "train"],
                )
                raw_test_data = (
                    pipeline | "ToTestRecordBatches" >> test_csv_tfxio.BeamSource()
                )

                raw_test_dataset = (raw_test_data, test_csv_tfxio.TensorAdapterConfig())

                # The TFXIO output format is chosen for improved performance.
                transformed_test_dataset = (
                    raw_test_dataset,
                    transform_fn,
                ) | tft_beam.TransformDataset(output_record_batches=True)

                # Transformed metadata is not necessary for encoding.
                transformed_test_data, _ = transformed_test_dataset

                # Extract transformed RecordBatches, encode and write them to the given
                # directory.
                _ = (
                    transformed_test_data
                    | "EncodeTestData"
                    >> beam.FlatMapTuple(lambda batch, _: RecordBatchToExamples(batch))
                    | "WriteTestData" >> beam.io.WriteToTFRecord(test_target_file)
                )

            # Will write a SavedModel and metadata to working_dir, which can then
            # be read by the tft.TFTransformOutput class.
            _ = transform_fn | "WriteTransformFn" >> tft_beam.WriteTransformFn(
                transform_target_dir
            )


def main() -> None:
    transform_data(
        train_source_file="gs://pipelines-hackathon/input_data/yahoo_stock.csv",
        train_target_file="gs://pipelines-hackathon/preprocessed_data/yahoo_stock.beam.train.tfrecord",
        test_files={
            "gs://pipelines-hackathon/input_data/yahoo_stock.csv": (
                "gs://pipelines-hackathon/preprocessed_data/yahoo_stock.beam.val.tfrecord"
            ),
            "gs://pipelines-hackathon/input_data/yahoo_stock.csv": (
                "gs://pipelines-hackathon/preprocessed_data/yahoo_stock.beam.test.tfrecord"
            ),
        },
        transform_target_dir="gs://pipelines-hackathon/models/preprocessing",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
