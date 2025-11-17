import pandas as pd

df = pd.read_csv("data/csv_outputs/cleaned_color_mileage_data.csv")
df["model"] = df["model"].apply(lambda x: x-621 if x > 1950 else x)
df = pd.get_dummies(df, columns=["model"])
df.columns = [col.replace("model_", "") for col in df.columns]
df.to_csv("data/csv_outputs/cleaned_color_mileage_model_data.csv", index=False)

