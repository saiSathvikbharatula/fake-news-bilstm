# Dataset: WELFake

1. Go to Kaggle and search **"WELFake fake news classification"** (dataset by Saurabh Shahane).
2. Download `WELFake_Dataset.csv` (~245 MB, 72,134 articles).
3. Place it here: `data/WELFake_Dataset.csv`

Columns: `title`, `text`, `label` (0 = fake, 1 = real in the original file).
The training script flips this so the model outputs **1 = FAKE**.

> WELFake merges four sources (Kaggle, McIntire, Reuters, BuzzFeed Political), which makes it
> more varied and less prone to shortcuts than single-source datasets like ISOT.
