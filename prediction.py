target = "total_spending"
print("Target column:", target)
print("Target exists:", target in df.columns)

print("\nTarget summary:")
print(df[target].describe())

#region ML Step 2 - Feature Selection

drop_cols = [
    "total_spending",
    "housing", "food", "transportation", "books_supplies",
    "entertainment", "personal_care", "technology",
    "health_wellness", "miscellaneous",
    "income_segment", "spending_segment", "final_segment",
    "spending_ratio",
    "cluster"
]

# sadece var olanları sil
drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=drop_cols)
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFeatures:")
print(X.columns)

#endregion

#region ML Step 3 - Preprocessing

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# categorical ve numerical ayır
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical:", cat_cols)
print("Numerical:", num_cols)

# preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

#endregion

#region ML Step 4 - Train Test + Baseline Model

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# 2. Pipeline (preprocessing + model)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 3. Model eğit
model.fit(X_train, y_train)

# 4. Tahmin
y_pred = model.predict(X_test)

# 5. Performans
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE (Linear Regression)")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

#endregion

#Öğrenci harcaması tahmin edilebilir değil çünkü davranış değişkenlere bağlı değil.

#region ML Step 5 - Random Forest

from sklearn.ensemble import RandomForestRegressor

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRANDOM FOREST PERFORMANCE")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R2:", r2_rf)
#endregion

#region ML Step 7 - Segment Included Model (Correct)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Drop (ama income_segment'i SAKLA)
drop_cols = [
    "total_spending",
    "housing", "food", "transportation", "books_supplies",
    "entertainment", "personal_care", "technology",
    "health_wellness", "miscellaneous",
    "spending_segment",   # ❌
    "final_segment",      # ❌
    "spending_ratio",
    "cluster"
]

drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=drop_cols)
y = df["total_spending"]

# 2. Cat / Num ayrımı (income_segment burada cat olacak)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical:", cat_cols)
print("Numerical:", num_cols)

# 3. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# 5. Model
rf_model_segment = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

# 6. Train
rf_model_segment.fit(X_train, y_train)

# 7. Predict
y_pred = rf_model_segment.predict(X_test)

# 8. Metrics
print("\nSEGMENT INCLUDED MODEL RESULTS")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

#endregion

#region Feature Importance (Random Forest)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# pipeline'dan model ve preprocessor al
rf = rf_model.named_steps["regressor"]
preprocessor = rf_model.named_steps["preprocessor"]

# feature isimlerini al
feature_names = preprocessor.get_feature_names_out()

# importance değerleri
importances = rf.feature_importances_

# dataframe oluştur
feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# TOP 20 göster
print("\nTOP 20 FEATURES:")
print(feat_imp.head(20))

# Görselleştirme
plt.figure(figsize=(10,6))
sns.barplot(
    x="importance",
    y="feature",
    data=feat_imp.head(15)
)
plt.title("Top 15 Feature Importance (Random Forest)")
plt.show()

#endregion

#region SHAP Step 1 - Clean Features

drop_cols = [
    "total_spending",
    "housing", "food", "transportation", "books_supplies",
    "entertainment", "personal_care", "technology",
    "health_wellness", "miscellaneous",
    "spending_ratio",
    "housing_ratio","food_ratio","technology_ratio",
    "books_supplies_ratio","transportation_ratio",
    "health_wellness_ratio","miscellaneous_ratio",
    "entertainment_ratio","personal_care_ratio",
    "spending_segment","final_segment",
    "cluster"
]

drop_cols = [col for col in drop_cols if col in df.columns]

X_clean = df.drop(columns=drop_cols)
y_clean = df["total_spending"]

#endregion

#region SHAP Step 2 - Rebuild Model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean,
    test_size=0.2,
    random_state=42
)

cat_cols = X_clean.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

rf_model_clean = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42))
])

rf_model_clean.fit(X_train, y_train)

#endregion

#region SHAP Step 3

import shap

# parçaları al
model = rf_model_clean.named_steps["model"]
preprocessor = rf_model_clean.named_steps["preprocessor"]

# transform et
X_train_transformed = preprocessor.transform(X_train)

# feature isimleri
feature_names = preprocessor.get_feature_names_out()

# explainer
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_train_transformed)

# grafik
shap.summary_plot(
    shap_values,
    X_train_transformed,
    feature_names=feature_names
)

#endregion

#Öğrenci harcama davranışı tek bir değişkenle açıklanamaz; finansal kapasite belirli bir rol oynasa da, harcama kararları büyük ölçüde bireysel tercihler, sosyal etkiler ve davranışsal faktörler tarafından şekillenmektedir.