import pandas as pd

df = pd.read_csv("data/csv_outputs/cleaned_color_mileage_model_data.csv")
df = df[df["price"] != 39700000000]
df["price"] = df["price"].apply(lambda x : x/35000000000)

df.to_csv("data/csv_outputs/cleaned_color_mileage_model_price_data.csv", float_format="%.10f", index=False)

print(df["price"].max())