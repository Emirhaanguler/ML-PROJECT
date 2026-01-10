import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

# 1. Veriyi Yükleme
df = pd.read_csv('TUANDROMD.csv')

# 2. Genel Bakış 
print(f"Gözlem Sayısı: {df.shape[0]}") # En az 1000 olmalı
print(f"Özellik Sayısı: {df.shape[1]}") # En az 20 olmalı
print("-" * 30)
print(df.info()) # Veri tipleri ve eksik değer kontrolü

print("İlk 5 Satır")
print(df.head())

# 2. İstatistiksel Özet
# Bu veri setinde özellikler 0 ve 1 olduğu için 'mean' (ortalama) bize 
# o iznin/API'nin tüm uygulamalar içinde ne kadar yaygın olduğunu söyler.
# Örneğin mean 0.10 ise, uygulamaların %10'u o izni istiyor demektir.
print("İstatistiksel Özet ")
print(df.describe().T.head(20)) # İlk 20 özelliği dikey (transpose) görmek için

# 3. Benzersiz Değer Kontrolü
# Özelliklerin gerçekten sadece 0 ve 1 mi yoksa başka değerler mi içerdiğini kontrol etme
print("Sütun Bazında Benzersiz Değer Sayıları (İlk 10)")
print(df.nunique().head(10))

# Label dağılımı 
print("\n Label Dağılımı")
counts = df['Label'].value_counts()
percentages = df['Label'].value_counts(normalize=True) * 100

for label, count in counts.items():
    print(f"{label}: {count} örnek ({percentages[label]:.2f}%)")


# Eksik değer kontrolü
missing_values = df.isnull().sum().sum()
print(f"Toplam Eksik Değer: {missing_values}") 
print("-" * 30)

# Eksik değer içeren satırları silme
df.dropna(inplace=True)

# Label Encoding
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Sınıf Eşleşmeleri: {mapping}")
X = df.drop('Label', axis=1)
y = df['Label']

# TRAIN-TEST SPLIT (DATA LEAKAGE ÖNLEME)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nEğitim Seti: {X_train.shape[0]} örnek")
print(f"Test Seti: {X_test.shape[0]} örnek")
print("-" * 30)

# A. Sınıf Dağılımı
plt.figure(figsize=(7, 5))
sns.countplot(x=y_train, hue=y_train, palette='viridis')
plt.title('Eğitim Seti Sınıf Dağılımı')
plt.xticks(ticks=[0, 1], labels=le.classes_)
plt.show()

# 3. FEATURE SELECTION (ÖZELLİK SEÇİMİ)
# A. Sabit Özellikler (Variance Threshold)
selector = VarianceThreshold(threshold=0.01)
X_train_v = selector.fit_transform(X_train)
kept_cols = X_train.columns[selector.get_support()]
print(f"Düşük varyans nedeniyle silinen özellik sayısı: {X_train.shape[1] - len(kept_cols)}")
X_train_filtered = X_train[kept_cols].copy()

# Yüksek Korelasyon Filtresi 
def remove_highly_correlated(data, threshold=0.9):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

dropped_corr_cols = remove_highly_correlated(X_train_filtered, threshold=0.9)
print(f"Çok yüksek korelasyon (%90+) nedeniyle silinen özellik sayısı: {len(dropped_corr_cols)}")
X_train_filtered.drop(columns=dropped_corr_cols, inplace=True)

# Mutual Information (Bilgi Kazancı) Hesaplama
importances = mutual_info_classif(X_train_filtered, y_train, random_state=42)
feature_info = pd.Series(importances, index=X_train_filtered.columns).sort_values(ascending=False)

# Label!ı en çok etkileyen 60 ve 20 feature belirleme
selected_features = feature_info.head(60).index.tolist()
top_20_features = feature_info.head(20).index.tolist()
top_2_features = feature_info.head(2).index.tolist()

# GÖRSELLEŞTİRME ADIMLARI
# En Yüksek MI'ye Sahip 20 Özelliğin Barplot'u
# İsimleri temizleme işlemi (karmaşık isimlerden kurtulma)
clean_names = [name.split('.')[-1].split('->')[-1] for name in feature_info.head(20).index]

plt.figure(figsize=(12, 8))

sns.barplot(
    x=feature_info.head(20).values, 
    y=clean_names, 
    hue=clean_names,
    palette='crest',
    legend=False
)

plt.title('Hedef Değişken ile En Yüksek Bilgi Kazancına (MI) Sahip 20 Özellik', fontsize=14, pad=20)
plt.xlabel('Mutual Information Skoru', fontsize=12)
plt.ylabel('Özellik Adı', fontsize=12, labelpad=15)

# Kenar Boşluklarını Manuel Ayarlama
plt.tight_layout()
plt.subplots_adjust(left=0.25) 

plt.show()

# En Etkili İlk 2 Özelliğin Sınıf Bazlı Dağılımı (COUNTPLOT)
print(f"En etkili 2 özellik görselleştiriliyor: {top_2_features}")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, col in enumerate(top_2_features):
    sns.countplot(x=X_train[col], hue=y_train, ax=axes[i], palette='viridis')
    axes[i].set_title(f'{col} İzninin Dağılımı', fontsize=12)
    axes[i].set_xlabel('İzin Durumu (0: Yok, 1: Var)')
    axes[i].legend(title='Sınıf', labels=list(mapping.keys()))

plt.tight_layout()
plt.show()

# En Kritik 20 Özellik Arasındaki Korelasyon Isı Haritası (HEATMAP)
print("Isı haritası oluşturuluyor...")

# Korelasyon matrisindeki karmaşık isimleri sadeleştiriyoruz
short_names = [name.split('.')[-1].split('->')[-1] for name in top_20_features]

# Korelasyon matrisini hesapla ve isimleri güncelle
correlation_matrix = X_train[top_20_features].corr()
correlation_matrix.index = short_names
correlation_matrix.columns = short_names

# Grafik Ayarları
plt.figure(figsize=(16, 12), dpi=100) 
sns.set_theme(style="white") # Tema ayarı

# Üst üçgeni maskeleme
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Heatmap Çizimi
heatmap = sns.heatmap(
    correlation_matrix, 
    mask=mask, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": .7}, 
    annot_kws={"size": 8} 
)

# 4. Eksen Etiketlerini Düzenleme
plt.xticks(
    rotation=45, 
    ha='right', 
    fontsize=10 
) 

plt.yticks(
    rotation=0, 
    fontsize=10
)

plt.title('En Kritik 20 Özellik - Korelasyon Isı Haritası', fontsize=16, pad=25)

# Kenar Boşluklarını Manuel Ayarlama 
plt.subplots_adjust(left=0.15, bottom=0.25) 

plt.show()

# FİNAL VERİ SETLERİNİN OLUŞTURULMASI
X_train_final = X_train[selected_features].copy()
X_test_final = X_test[selected_features].copy()

print(f"Final Özellik Sayısı: {len(selected_features)}")
print(f"Eğitim Seti Boyutu: {X_train_final.shape}")
print(f"En Kritik 5 Özellik:\n{feature_info.head(5)}")