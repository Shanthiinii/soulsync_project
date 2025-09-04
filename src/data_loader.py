import pandas as pd
import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_combined_dataset():
    """
    Loads the combined songs+movies+books dataset.
    Returns a Pandas DataFrame.
    """
    path = os.path.join(BASE_DIR, "data", "combined_df.csv")
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_combined_dataset()
    print("Dataset Shape:", df.shape)
    print("Sample Data:\n", df.head())
