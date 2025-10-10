from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data/plantdex_clip_pairs.csv")
train_df, val_df = train_test_split(df, test_size=0.1, random_state=1337)
train_df.to_csv("data/train_pairs.csv", index=False)
val_df.to_csv("data/val_pairs.csv", index=False)
