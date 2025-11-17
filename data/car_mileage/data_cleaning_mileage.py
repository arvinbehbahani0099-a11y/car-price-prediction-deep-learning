import pandas as pd

df = pd.read_csv("data/cleaned_color_data.csv")
df["mileage"] = df["mileage"].apply(lambda x: x/999999)
df.to_csv("data/csv_outputs/cleaned_color_mileage_data.csv", index=False)