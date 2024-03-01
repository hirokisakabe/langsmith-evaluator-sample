import csv
import os

from dotenv import load_dotenv
from langsmith import Client


def main():
    client = Client()
    dataset_name = os.getenv("DATASET_NAME")

    if client.has_dataset(dataset_name=dataset_name):
        client.delete_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(
        dataset_name=dataset_name,
    )

    dataset_csv_file = "dataset.csv"

    with open(dataset_csv_file) as f:
        reader = csv.reader(f)
        input_output_list = [row for row in reader]

    client.create_examples(
        inputs=[{"input": input_output[0]} for input_output in input_output_list],
        outputs=[{"output": input_output[1]} for input_output in input_output_list],
        dataset_id=dataset.id,
    )


if __name__ == "__main__":
    load_dotenv()

    main()
