import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor

df = pd.read_csv("data/csv_outputs/cleaned_mileage_model_price_name_color_data.csv.csv")


cat_idxs = [ df.columns.get_loc("color_id"), df.columns.get_loc("name_cluster") ] #
cat_dims = [ df["color_id"].nunique(), df["name_cluster"].nunique() ] # 

num_cols = ["mileage", "model"]  

X = df[["color_id", "name_cluster"] + num_cols].values.astype(np.float32)
y = df["price"].values.astype(np.float32)
y = y.reshape(-1, 1)  # 

tabnet_params = {
    "n_d": 8,
    "n_a": 8,
    "n_steps": 5, #
    "gamma": 1.5,
    "cat_idxs": cat_idxs, #
    "cat_dims": cat_dims, #
    "cat_emb_dim": [4, 8],
    "optimizer_fn": __import__("torch").optim.Adam,
    "optimizer_params": {"lr": 2e-2}
}

model = TabNetRegressor(**tabnet_params)  #

model.fit(
    X, y,
    max_epochs=100,
    batch_size=256,
    virtual_batch_size=128, #
    patience=20, #
    drop_last=False
)

preds = model.predict(X)
print(preds[:10])
