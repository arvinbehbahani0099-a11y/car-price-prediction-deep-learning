import numpy as np
import pandas as pd
from data.car_name_k_means.data_cleaning_name import kmeans
from sklearn.metrics import silhouette_samples

df = pd.read_csv("data/csv_outputs/cleaned_mileage_model_price_name_color_data.csv")

# فرض: labels حاصل از KMeans و emb_reduced یا X موجوده
labels = kmeans.labels_   # یا labels_final
df['name_cluster'] = labels

# اندازه هر خوشه
cluster_sizes = df['cluster'].value_counts().sort_values(ascending=False)
print(cluster_sizes.head(30))

# silhouette به ازای هر نمونه
sil_vals = silhouette_samples(emb_reduced, labels, metric='euclidean')
df['silhouette'] = sil_vals

# میانگین silhouette هر خوشه
sil_by_cluster = df.groupby('cluster')['silhouette'].mean().sort_values(ascending=False)
print(sil_by_cluster.head(30))

# درصد خوشه‌های خیلی ریز
small_clusters = cluster_sizes[cluster_sizes < 10].count()
print("Small clusters (<10):", small_clusters, " / ", len(cluster_sizes))
