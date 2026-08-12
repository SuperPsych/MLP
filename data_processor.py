import pandas as pd
from PIL import Image
import io

def to_data_txt(parquet_path, out_path):
    df = pd.read_parquet(parquet_path)
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            img = Image.open(io.BytesIO(row["image"]["bytes"]))
            pixels = list(img.getdata())
            one_hot = [0] * 10
            one_hot[row["label"]] = 1
            f.write(" ".join(map(str, pixels)))
            f.write(" " + " ".join(map(str, one_hot)) + "\n")

to_data_txt("./data/mnist/train.parquet", "./mnist_train.txt")
to_data_txt("./data/mnist/test.parquet", "./mnist_test.txt")