import csv
import json
import os
import pickle


def load_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_to_pickle(data, path):
    with open(ensure_path(path), "wb") as f:
        pickle.dump(data, f)


def load_json(input_path):
    with open(input_path, "rb") as f:
        return json.load(f)


def save_json(data, output_path):
    with open(output_path, "w") as f:
        return json.dump(data, f)


def save_list_as_text(data, output_path):
    output_path = ensure_path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for word in data:
            f.write("{}\n".format(word))


def load_text_as_list(input_path):
    with open(input_path, 'r', encoding="utf-8") as f:
        return f.read().splitlines()


def ensure_path(path):
    directory = os.path.dirname(path)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return path


def write_csv(content, header, path, delimiter=","):
    path = ensure_path(path)
    with open(path, 'w', encoding="utf-8", newline='') as f:
        csv_writer = csv.writer(f, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header is not None:
            csv_writer.writerow(header)

        for row in content:
            csv_writer.writerow(row)


def read_csv(path, load_header=False, delimiter=","):
    content = []
    with open(path, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=delimiter, quotechar='"')
        if load_header:
            [content.append(row) for row in csv_reader]
        else:
            [content.append(row) for i, row in enumerate(csv_reader) if i > 0]
    return content