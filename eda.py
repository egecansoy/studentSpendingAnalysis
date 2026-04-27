import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#region EDA

sns.histplot(df["monthly_income"], kde=True)
plt.title("Income Distribution")
plt.show()
## "Gelirimiz genelde eşit dağılıyor.

# ortalama, median
print("Mean:", df["monthly_income"].mean())
print("Median:", df["monthly_income"].median())

df.groupby("gender")["monthly_income"].mean().sort_values(ascending=False)

##Cinsiyetler arasında çok fark olmasa da genel olarak NB bireyler daha yüksek gelir elde ediyor.

sns.boxplot(x="gender", y="monthly_income", data=df)
plt.show()

#region EFFECTIVE INCOME SEGMENTATION

# 1. Effective income oluştur
df["effective_income"] = df["monthly_income"] + df["financial_aid"]

# 2. Total spending
df["total_spending"] = (
    df["housing"] +
    df["food"] +
    df["transportation"] +
    df["books_supplies"] +
    df["entertainment"] +
    df["personal_care"] +
    df["technology"] +
    df["health_wellness"] +
    df["miscellaneous"]
)

# 3. Effective income segment
df["income_segment"] = pd.qcut(
    df["effective_income"],
    q=3,
    labels=["Low", "Medium", "High"]
)

# 4. Spending segment
df["spending_segment"] = pd.qcut(
    df["total_spending"],
    q=3,
    labels=["Low", "Medium", "High"]
)

# 5. Final segment
df["final_segment"] = (
    df["income_segment"].astype(str) + "_" +
    df["spending_segment"].astype(str)
)

# 6. Segment summary
segment_summary = df.groupby("final_segment").agg({
    "effective_income": "mean",
    "monthly_income": "mean",
    "financial_aid": "mean",
    "total_spending": "mean"
}).sort_values(by="total_spending", ascending=False)

print(segment_summary)

# 7. Segment counts
print("\nSegment counts:")
print(df["final_segment"].value_counts())
## Düşük gelirli ancak yüksek harcama yapan öğrenciler ciddi finansal baskı altında.
## Gelir artışı harcamayı artırmıyor → öğrenciler “luxury consumption” yapmıyor.
## Yüksek gelirli ama düşük harcayan öğrenciler “saving segment”tir.
## Orta gelir – orta harcama segmenti en stabil gruptur ve “baseline student behavior” temsil eder.
## Öğrencilerin harcama davranışı gelirden bağımsızdır; düşük gelirli öğrenciler bile yüksek harcama yaparak finansal baskı altında kalırken, yüksek gelirli öğrenciler ise harcamalarını artırmayarak daha stabil bir tüketim davranışı sergilemektedir.
## Segmentlerin birbirine yakın büyüklükte olması, öğrencilerin gelir ve harcama açısından dengeli bir şekilde dağıldığını ve belirli bir grubun baskın olmadığını göstermektedir. Bu durum, öğrencilerin finansal davranışlarının heterojen olduğunu ve farklı ekonomik profillerin birlikte var olduğunu ortaya koymaktadır.


df.groupby("final_segment")["entertainment"].mean()
# Düşük gelir elde edip yüksek harcama yapanlar, eğlenceye diğer gruplara göre fazla para harcamışlar.
# Yüksek gelir elde edip düşük harcama yapanlar eğlenceye daha az bütçe ayırmışlar.
#endregion

spending_cols = [
    "housing","food","transportation","books_supplies",
    "entertainment","personal_care","technology",
    "health_wellness","miscellaneous"
]

spending_breakdown = df[spending_cols].mean().sort_values(ascending=False)
print(spending_breakdown)

##Öğrencilerin harcamalarının çok büyük bir kısmı, keyfi değil zorunlu harcamalar.

segment_category = df.groupby("final_segment")[spending_cols].mean()
print(segment_category)

df["spending_ratio"] = df["total_spending"] / df["effective_income"]

print(df.groupby("final_segment")["spending_ratio"].mean().sort_values(ascending=False))
#Düşük gelir elde edip yüksek harcama yapanlar, gelirlerinin iki katı kadar harcama yapıyorlar.
# Yüksek gelir elde edip, düşük harcama yapanlar yaklaşık %75ini harcıyor.

df.groupby("gender")["total_spending"].mean()
# Bu örneklemde, cinsiyetler arası çok büyük bir harcama farkı olmasa da genel olarak NB ve erkek bireyler kdaınlardan daha çok harcama yapmışl.ar

df.groupby("year_in_school")["effective_income"].mean()

##Öğrencilerin akademik ilerlemeleri finansal bir doğrusal artış izlemiyor; birinci sınıf öğrencileri en yüksek harcanabilir gelire sahipken, ikinci sınıf öğrencileri finansal daralma ile en hassas segmenti oluşturuyor.

df.groupby("preferred_payment_method")["total_spending"].mean()

##Ödeme yöntemi öğrencilerin toplam harcamasını anlamlı şekilde değiştirmemektedir; harcama seviyesi neredeyse sabit kalmaktadır.

sns.scatterplot(x="tuition", y="total_spending", data=df)
plt.show()
##Tuition (okul ücreti) ile total spending arasında anlamlı bir ilişki yoktur; öğrenciler farklı tuition seviyelerinde olsalar bile benzer harcama düzeylerini korumaktadır.

for col in spending_cols:
    df[col + "_ratio"] = df[col] / df["total_spending"]

category_share = df[[col+"_ratio" for col in spending_cols]].mean().sort_values(ascending=False)
print(category_share)

##Öğrenci harcamalarının yaklaşık %53’ü (housing + food) temel yaşam giderlerine gitmektedir; bu da veri setinin “zorunlu tüketim (survival consumption)” odaklı olduğunu göstermektedir. Öğrenciler keyfi harcamalarını minimum seviyede tutmaktadır. Eğitim ve teknoloji harcamaları önemli bir paya sahiptir, bu da öğrencilerin harcamalarını üretken alanlara yönlendirdiğini gösterir.

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = df[["effective_income", "total_spending"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

df.groupby("cluster")[["effective_income","total_spending"]].mean()

#endregion