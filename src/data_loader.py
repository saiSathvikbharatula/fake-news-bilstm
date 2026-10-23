"""Load, clean and split the WELFake dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.preprocess import clean_text, strip_dateline


def load_dataset(path=config.DATA_PATH, sample: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download WELFake_Dataset.csv from Kaggle (see data/README.md) "
            "and place it in the data/ folder."
        )
    df = pd.read_csv(path)
    df = df[["title", "text", "label"]]
    df["title"] = df["title"].fillna("")
    # Strip the dateline from the body *before* joining, so a long title
    # cannot push it past the regex's 80-character window.
    df["text"] = df["text"].fillna("").apply(strip_dateline)
    df["content"] = (df["title"] + " " + df["text"]).str.strip()

    before = len(df)
    df = df[df["content"].str.len() > 0]
    df = df.drop_duplicates(subset="content")
    print(f"Loaded {before:,} rows -> {len(df):,} after removing empty/duplicate articles")

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=config.SEED)
        print(f"Using a random sample of {len(df):,} rows")

    if config.FLIP_LABELS:
        df["label"] = 1 - df["label"]  # 1 = FAKE, 0 = REAL

    print("Cleaning text ...")
    df["clean"] = df["content"].apply(clean_text)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame):
    """Stratified train / val / test split."""
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.SEED
    )
    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, stratify=train_df["label"], random_state=config.SEED
    )
    print(f"Split -> train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")
    return train_df, val_df, test_df
