import csv
import json

input_csv = "dataset.csv"
output_jsonl = "train.jsonl"

with open(input_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for row in reader:
            instruction = (
                "Classify the validity of an academic paper based on its metadata."
            )
            context = (
                f"Author(s): {row['Author']}\n"
                f"Title: {row['Title']}\n"
                f"Year: {row['Year']}\n"
                f"Venue: {row['Venue']}\n"
                f"DOI: {row['DOI']}"
            )
            output = row["Label"]

            json_obj = {
                "instruction": instruction,
                "input": context,
                "output": output
            }
            out.write(json.dumps(json_obj) + "\n")
