import os
import numpy as np
import pandas as pd
import streamlit as st
import kagglehub
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==============================================================================
# 1. הקוד המקורי שלך (אימון המודל וטעינת הנתונים)
# ==============================================================================

KAGGLE_DATASET = "nehalbirla/vehicle-dataset-from-cardekho"
FEATURE1_COLUMN = "km_driven"
FEATURE2_COLUMN = "year"
TARGET_COLUMN = "selling_price"

# הורדת הדאטה-סט מ-Kaggle וטעינת ה-CSV ל-DataFrame
path = kagglehub.dataset_download(KAGGLE_DATASET)
csv_path = os.path.join(path, "CAR DETAILS FROM CAR DEKHO.csv")
data = pd.read_csv(csv_path)

# ניקוי בסיסי של ערכים חסרים
data = data.dropna(subset=[FEATURE1_COLUMN, FEATURE2_COLUMN, TARGET_COLUMN])

# חילוץ הנתונים והמרה ל-NumPy
X1 = data[FEATURE1_COLUMN].to_numpy()
X2 = data[FEATURE2_COLUMN].to_numpy()
y = data[TARGET_COLUMN].to_numpy()

# חישוב שגיאת הבסיס (Baseline Loss)
baseline_prediction = np.mean(y)
baseline_loss = np.mean(np.abs(y - baseline_prediction))

# אימון מודל רגרסיה ליניארית
X1_reshaped = X1.reshape(-1, 1)
X2_reshaped = X2.reshape(-1, 1)
model = LinearRegression()
X = np.column_stack((X1_reshaped, X2_reshaped))
model.fit(X, y)

w1 = model.coef_[0]
w2 = model.coef_[1]
b = model.intercept_

y_hat = model.predict(X)
model_loss = np.mean(np.abs(y - y_hat))

# ==============================================================================
# 2. ממשק אינטראקטיבי מתקדם ב-Streamlit
# ==============================================================================

st.set_page_config(
    page_title="סימולטור ויזואלי לתחזית מחירי רכב",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ סימולטור אינטראקטיבי לתחזית מחירי רכב")

# חלוקת המסך לכרטיסיות אינטראקטיביות
tab1, tab2, tab3 = st.tabs(["🎮 סימולטור חי מבוסס סליידרים", "📊 מיקום הרכב ביחס לדאטה", "📈 ניתוח מדדים"])

# ------------------------------------------------------------------------------
# בלשונית 1: סימולטור בזמן אמת עם השפעה ישירה
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("🎛️ כוונן את המאפיינים וראה את המחיר משתנה בלייב")
    
    col_input, col_result = st.columns([1, 1])

    with col_input:
        
        # סליידר שנת ייצור (מוגדר מהגבוה לנמוך)
        user_year = st.slider(
            "שנת ייצור:",
            min_value=int(data[FEATURE2_COLUMN].max()),
            max_value=int(data[FEATURE2_COLUMN].min()),
            value=2016,
            step=-1
        )

        # סליידר קילומטראז' (מוגדר מהגבוה לנמוך)
        user_km = st.slider(
            "קילומטראז' (ק\"מ):",
            min_value=300000,
            max_value=0,
            value=60000,
            step=-5000
        )

        # חיזוי בזמן אמת
        predicted_price = model.predict(np.array([[user_km, user_year]]))[0]

    with col_result:
        st.markdown("### 🏷️ התחזית המיידית:")
        st.metric(label="מחיר משוער", value=f"₹{predicted_price:,.2f}")

        # ניתוח רגישות (אינטראקטיביות נוספת)
        km_impact_10k = w1 * 10000
        st.info(f"""
        **💡 תובנות מהירות מהמודל:**
        * **שנה חדישה יותר** מוסיפה למחיר: **₹{w2:,.2f}**
        * **כל 10,000 ק"מ נוספים** מורידים מהמחיר: **₹{abs(km_impact_10k):,.2f}**
        """)

# ------------------------------------------------------------------------------
# בלשונית 2: גרף אינטראקטיבי שמסמן את הרכב הנבחר על הדאטה
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📍 איפה הרכב הנבחר ממוקם ביחס לשאר השוק?")
    
    plot_feature = st.radio(
        "בחר ציר X להצגה:",
        options=[FEATURE1_COLUMN, FEATURE2_COLUMN],
        format_func=lambda x: "קילומטראז'" if x == FEATURE1_COLUMN else "שנת ייצור",
        horizontal=True
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # נקודות של כל הדאטה
    ax.scatter(data[plot_feature], data[TARGET_COLUMN], alpha=0.25, color="gray", label="רכבים בשוק")
    
    # הנקודה שהמשתמש בחר כרגע
    user_x = user_km if plot_feature == FEATURE1_COLUMN else user_year
    ax.scatter([user_x], [predicted_price], color="red", s=150, zorder=5, label="הרכב הנבחר שלך")

    ax.set_xlabel("קילומטראז'" if plot_feature == FEATURE1_COLUMN else "שנת ייצור")
    ax.set_ylabel("מחיר מכירה")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    st.pyplot(fig)

# ------------------------------------------------------------------------------
# בלשונית 3: מדדי המודל
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("📊 ביצועי המודל והשוואה ל-Baseline")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline MAE", f"₹{baseline_loss:,.2f}")
    c2.metric("Model MAE", f"₹{model_loss:,.2f}")
    c3.metric("שיפור מה-Baseline", f"{((baseline_loss - model_loss) / baseline_loss) * 100:.1f}%")
    
    st.divider()
    st.dataframe(data.head(10))