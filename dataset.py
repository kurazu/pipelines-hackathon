import datetime
import json
import logging

import apache_beam as beam
import tensorflow as tf
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.io.tfrecordio import WriteToTFRecord
from apache_beam.options.pipeline_options import PipelineOptions


class SplitCSVLine(beam.DoFn):
    def process(self, element):
        date, high, low, open_, close, volume, adj_close = element.split(",")
        return [
            {
                "date": datetime.datetime.strptime(date, "%Y-%m-%d").date(),
                "high": float(high),
                "low": float(low),
                "open": float(open_),
                "close": float(close),
                "volume": float(volume),
                "adj_close": float(adj_close),
            }
        ]


def _float_feature(value: float) -> tf.train.Feature:
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


class ToExample(beam.DoFn):
    def process(self, element):
        features = {
            "open": _float_feature(element["open"]),
            "close": _float_feature(element["close"]),
            "date_range": _float_feature(element["date_range"]),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return [example_proto]
        # message: bytes = example_proto.SerializeToString()
        # return []


def main() -> None:
    min_date = datetime.date(2015, 11, 23)
    max_date = datetime.date(2020, 11, 20)
    input_file = "gs://pipelines-hackathon/input_data/yahoo_stock.csv"
    output_path_prefix_template = (
        "gs://pipelines-hackathon/processed_data/yahoo_stock.{subset}."
    )
    output_path_suffix = ".tfrecord"
    output_shards = 4
    job_name = f"dataset_preprocessing_{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}"

    beam_options = PipelineOptions(
        runner="DirectRunner",
        project="pipelines-hackathon",
        job_name=job_name,
        temp_location="gs://pipelines-hackathon/temp_data",
    )

    with beam.Pipeline(options=beam_options) as p:
        csv_lines = p | "Read" >> ReadFromText(input_file, skip_header_lines=1)
        dicts = csv_lines | "Parse" >> beam.ParDo(SplitCSVLine())
        dicts_with_range = dicts | "Add date range" >> beam.FlatMap(
            lambda element: [
                {
                    **element,
                    "date_range": (element["date"] - min_date) / (max_date - min_date),
                }
            ]
        )

        json_dicts = dicts_with_range | "ToJSON" >> beam.FlatMap(
            lambda elem: [json.dumps(elem, default=datetime.date.isoformat)]
        )
        json_dicts | "WriteJSON" >> WriteToText("data/x.json")
        protobufs = dicts_with_range | "ToExample" >> beam.ParDo(ToExample())
        protobufs | "Write" >> WriteToTFRecord(
            file_path_prefix=output_path_prefix_template.format(subset="all"),
            file_name_suffix=output_path_suffix,
            num_shards=output_shards,
            coder=beam.coders.ProtoCoder(None),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
