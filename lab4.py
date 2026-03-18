import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# Generate synthetic dataset
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Visualize generated data
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Generated Data")
plt.show()


# -----------------------------
# 2. APPLY K-MEANS
# -----------------------------

kmeans = KMeans(n_clusters=3, random_state=0)

# Predict clusters
y_kmeans = kmeans.fit_predict(X)

# Plot clustered data
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=30)
print(X.shape)
# Plot centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='X', label='Centroids')

plt.title("K-means Clustering")
plt.legend()
plt.show()


# -----------------------------
# 3. ADJUST & EXPERIMENT
# -----------------------------

# Example with different number of clusters
kmeans_test = KMeans(n_clusters=5, random_state=0)
y_test = kmeans_test.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_test, cmap='viridis', s=30)

# Plot centroids
centers_test = kmeans_test.cluster_centers_
plt.scatter(centers_test[:, 0], centers_test[:, 1], c='red', s=100, marker='X', label='Centroids')

plt.title("K-means with 5Clusters")
plt.legend()
plt.show()



# 4. ELBOW METHOD


inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# elbow method plotting
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()




#EXERCISE 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import kagglehub
import os

# -----------------------------
# 1. VERİ SETİNİ YÜKLEME
# -----------------------------
print("Veri seti indiriliyor...")
path = kagglehub.dataset_download("ziya07/car-performance-dataset")

# İndirilen klasördeki csv dosyasını bulup okutalım
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
data_path = os.path.join(path, csv_files[0])
data = pd.read_csv(data_path)

# -----------------------------
# 2. VERİ ÖN İŞLEME (Sadece 2 Özellik)
# -----------------------------
# Kümelerin grafikte net ayrışması için sadece görselleştireceğimiz 2 özelliği seçiyoruz
features = data[['Fuel_Efficiency', 'Price']]

# Eksik değerleri temizleme
data = data.dropna(subset=['Fuel_Efficiency', 'Price']).reset_index(drop=True)
features = data[['Fuel_Efficiency', 'Price']]

# Özellikleri standartlaştırma (Standard Scaler)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# -----------------------------
# 3. K-MEANS UYGULAMASI (K=4)
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(features_scaled)

# Kümeleri orijinal veri setine ekliyoruz
data['Cluster'] = labels

# Görselleştirme (2 Boyutta: Fiyat vs. Yakıt Verimliliği)
plt.figure(figsize=(10, 6))

# features_scaled[:, 1] -> Price (X ekseni)
# features_scaled[:, 0] -> Fuel_Efficiency (Y ekseni)
scatter = plt.scatter(features_scaled[:, 1], features_scaled[:, 0], c=labels, cmap='viridis', s=50, alpha=0.7)

# Merkezleri (Centroids) çizdirme
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='red', s=200, marker='X', label='Centroids ($\mu_k$)')

plt.title("Vehicle Clusters (Price vs Fuel Efficiency - 2D Clustering)")
plt.xlabel("Price (Standarised)")
plt.ylabel("Fuel Efficiency (Standarised)")
plt.legend()
plt.colorbar(scatter, label="Cluster Number")
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------------
# 4. SONUÇLARIN ANALİZİ
# -----------------------------
print("\n--- Cluster Characteristics ---")
# Güvenlik puanını çıkardığımız için sadece 2 metriğin ortalamasını alıyoruz
cluster_summary = data.groupby('Cluster')[['Fuel_Efficiency', 'Price']].mean()
print(cluster_summary)

# -----------------------------
# 5. ELBOW METODU (Optimum K Değerini Bulma)
# -----------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(features_scaled)
    inertia.append(km.inertia_) # J distorsiyon değerini kaydediyoruz

# Elbow characteristic plot
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--', color='b')
plt.xlabel("Cluster number (k)")
plt.ylabel("Inertia (J Distortion)")
plt.title("Elbow Method with Optimum K Value")
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.show()


##exercise 2 with 3 features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import kagglehub
import os

# -----------------------------
# 1. VERİ SETİNİ YÜKLEME
# -----------------------------
print("Veri seti indiriliyor...")
path = kagglehub.dataset_download("ziya07/car-performance-dataset")

# İndirilen klasördeki csv dosyasını bul
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
data_path = os.path.join(path, csv_files[0])
data = pd.read_csv(data_path)

# -----------------------------
# 2. VERİ ÖN İŞLEME (3 Özellik)
# -----------------------------
# 3 Özelliği birden seçiyoruz
features = data[['Fuel_Efficiency', 'Price', 'Safety_Rating']]

# Eksik verileri temizleme
data = data.dropna(subset=['Fuel_Efficiency', 'Price', 'Safety_Rating']).reset_index(drop=True)
features = data[['Fuel_Efficiency', 'Price', 'Safety_Rating']]

# Özellikleri standartlaştırma (Standard Scaler) - 3 boyut için de şart!
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# -----------------------------
# 3. K-MEANS UYGULAMASI (K=4)
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(features_scaled)

# Kümeleri orijinal veri setine ekleyelim
data['Cluster'] = labels


# 4. MATLAB TARZI 3D GÖRSELLEŞTİRME

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d') 


x_vals = features_scaled[:, 0] # X: Fuel Efficiency
y_vals = features_scaled[:, 1] # Y: Price
z_vals = features_scaled[:, 2] # Z: Safety Rating

# 3D Scatter (Dağılım) Grafiği
scatter = ax.scatter(x_vals, y_vals, z_vals, c=labels, cmap='viridis', s=50, alpha=0.8)

# Merkezleri (Centroids) 3D uzayda çizdirme
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
           c='red', s=300, marker='X', label='Centroids ($\mu_k$)')

# Eksen İsimleri ve Başlık
ax.set_title("3D Vehicle Clusters (Fuel vs Price vs Safety)")
ax.set_xlabel("Fuel Efficiency (Scaled)")
ax.set_ylabel("Price (Scaled)")
ax.set_zlabel("Safety Rating (Scaled)")

# Renk skalası (Colorbar) ve Lejant
fig.colorbar(scatter, ax=ax, label="Cluster Number", pad=0.1)
ax.legend()

# Grafiği göster
plt.show()


print("\n--- 3 Özellikli Küme Karakteristikleri ---")
cluster_summary = data.groupby('Cluster')[['Fuel_Efficiency', 'Price', 'Safety_Rating']].mean()
print(cluster_summary)