import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


df = pd.read_csv(
    "data/csv_outputs/cleaned_mileage_model_price_name_color_data.csv")

names = df['name'].astype(str).tolist()

vectorizer = TfidfVectorizer(min_df=5)
X = vectorizer.fit_transform(names)

K_range = range(50, 3000, 10)

results = []

for k in K_range:
    print(f"Testing k={k}")

    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)

    sse = km.inertia_
    sil = silhouette_score(X, labels)

    results.append({
        "k": k,
        "SSE": sse,
        "Silhouette": sil
    })


results_df = pd.DataFrame(results)
print(results_df)

# Best k according to silhouette
best_k_sil = results_df.loc[results_df["Silhouette"].idxmax(), "k"]
print("\nBest k by Silhouette =", best_k_sil)

# Plot both in one figure
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("k")
ax1.set_ylabel("SSE", color="blue")
ax1.plot(results_df["k"], results_df["SSE"],
         marker="o", color="blue", label="SSE (Elbow)")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()  # دومین محور Y
ax2.set_ylabel("Silhouette Score", color="red")
ax2.plot(results_df["k"], results_df["Silhouette"],
         marker="o", color="red", label="Silhouette")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("Elbow + Silhouette Combined")
plt.show()
