import pandas as pd

df = pd.read_csv("medium_articles.csv")
print(df.shape)
sub = df[:5000]
sub.to_csv("subset2.csv", index=False)