"""
EV Charging Patterns - Analysis & Cost Prediction
A simple dynamic website made with Python + Streamlit.

Run with:
    pip install -r requirements.txt
    streamlit run app.py

Keep ev_charging_patterns.csv in the same folder (or upload one from the sidebar).
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="EV Charging Patterns", page_icon="⚡", layout="wide")

TARGET = "Charging Cost (USD)"
NUMERIC_COLS = [
    "Battery Capacity (kWh)", "Energy Consumed (kWh)", "Charging Duration (hours)",
    "Charging Rate (kW)", "State of Charge (Start %)", "State of Charge (End %)",
    "Distance Driven (since last charge) (km)", "Temperature (°C)", "Vehicle Age (years)",
]
CAT_COLS = ["Vehicle Model", "Charging Station Location", "Time of Day",
            "Day of Week", "Charger Type", "User Type"]
TIME_ORDER = ["Morning", "Afternoon", "Evening", "Night"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------
# 1. Load and clean the data (same steps as the notebook)
# ---------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Energy Consumed (kWh)", "Charging Rate (kW)",
                "Distance Driven (since last charge) (km)"]:
        if col in df:
            df[col] = df[col].fillna(df[col].median())
    df = df.drop_duplicates()
    df = df[(df["State of Charge (Start %)"] <= 100) & (df["State of Charge (End %)"] <= 100)]
    df = df.drop(columns=["User ID", "Charging Station ID", "Charging Start Time",
                          "Charging End Time"], errors="ignore")
    df = df.dropna()
    return df.reset_index(drop=True)


@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return clean(pd.read_csv(file))


st.sidebar.title("⚡ EV Charging")
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")

# Keep the data in session_state so new rows added by the user stay.
source_key = uploaded.name if uploaded else "default"
if st.session_state.get("source") != source_key:
    try:
        st.session_state.data = load_csv(uploaded if uploaded else "ev_charging_patterns.csv")
        st.session_state.source = source_key
    except FileNotFoundError:
        st.error("ev_charging_patterns.csv not found. Put it next to app.py or upload it from the sidebar.")
        st.stop()

data = st.session_state.data
data["Is Weekend"] = data["Day of Week"].isin(["Saturday", "Sunday"]).astype(int)

# ---------------------------------------------------------------
# 2. Sidebar filters - every chart and number updates from these
# ---------------------------------------------------------------
st.sidebar.header("Filters")
filters = {}
for col in ["Charging Station Location", "Charger Type", "User Type", "Vehicle Model"]:
    options = sorted(data[col].unique())
    filters[col] = st.sidebar.multiselect(col, options, default=options)

cost_min, cost_max = float(data[TARGET].min()), float(data[TARGET].max())
cost_range = st.sidebar.slider("Charging cost range (USD)", cost_min, cost_max,
                               (cost_min, cost_max))

df = data.copy()
for col, chosen in filters.items():
    df = df[df[col].isin(chosen)]
df = df[df[TARGET].between(*cost_range)]

st.sidebar.markdown(f"**Rows shown:** {len(df)} of {len(data)}")
if st.sidebar.button("Reset data to original file"):
    st.session_state.pop("source", None)
    st.rerun()

page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Explore", "🔗 Relationships",
                                  "🤖 ML Models", "🔮 Predict Cost", "➕ Add Data"])

if df.empty:
    st.warning("No rows match these filters. Change the filters in the sidebar.")
    st.stop()


# ---------------------------------------------------------------
# Helper: train all models on the current (filtered) data
# ---------------------------------------------------------------
@st.cache_data
def train_models(frame: pd.DataFrame, test_size: float, k: int, depth: int, trees: int):
    X = pd.get_dummies(frame.drop(columns=[TARGET]), drop_first=True, dtype=int)
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler().fit(X_train)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=depth, random_state=42),
        f"KNN (k={k})": KNeighborsRegressor(n_neighbors=k),
        "Random Forest": RandomForestRegressor(n_estimators=trees, random_state=42),
    }

    def score(name, pred):
        return {"Model": name,
                "MAE": round(mean_absolute_error(y_test, pred), 2),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
                "R2": round(r2_score(y_test, pred), 3)}

    results = [score("Baseline (average)", np.full(len(y_test), y_train.mean()))]
    preds = {}
    for name, m in models.items():
        if name.startswith("KNN"):
            m.fit(scaler.transform(X_train), y_train)
            p = m.predict(scaler.transform(X_test))
        else:
            m.fit(X_train, y_train)
            p = m.predict(X_test)
        preds[name] = p
        results.append(score(name, p))

    return models, scaler, list(X.columns), pd.DataFrame(results), y_test, preds


# ---------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------
if page == "🏠 Home":
    st.title("⚡ Analysis and Cost Prediction of EV Charging Patterns")
    st.write("**Kunwar Prakhar Singh** · Reg. No. 12502064 · Section K2P24QR  \n"
             "INT234 – Predictive Analytics · Guide: Dr. Mrinalini Rana · Lovely Professional University")
    st.write("This app looks at electric vehicle charging sessions and tests whether machine "
             "learning can predict the cost of a session. Use the **filters** in the sidebar and "
             "every number and chart will update.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", f"{len(df):,}")
    c2.metric("Average cost", f"${df[TARGET].mean():.2f}")
    c3.metric("Average energy", f"{df['Energy Consumed (kWh)'].mean():.1f} kWh")
    c4.metric("Average duration", f"{df['Charging Duration (hours)'].mean():.2f} h")

    st.subheader("Objectives")
    st.markdown("""
1. Which cars, users and chargers appear most often?
2. How does charging cost differ by city and charger type?
3. When do people usually charge their cars?
4. Are cost, energy and other numeric values related?
5. Can machine learning predict charging cost?
""")
    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.download_button("Download filtered data (CSV)", df.to_csv(index=False),
                       "filtered_ev_data.csv", "text/csv")

elif page == "📊 Explore":
    st.title("📊 Explore the Data")

    st.subheader("Objective 1: Cars, users and chargers")
    c1, c2, c3 = st.columns(3)
    c1.write("**Sessions by vehicle model**")
    c1.bar_chart(df["Vehicle Model"].value_counts())
    c2.write("**Sessions by user type**")
    c2.bar_chart(df["User Type"].value_counts())
    c3.write("**Sessions by charger type**")
    c3.bar_chart(df["Charger Type"].value_counts())
    st.write("User type vs charger type:")
    st.dataframe(pd.crosstab(df["User Type"], df["Charger Type"]), use_container_width=True)

    st.subheader("Objective 2: Cost by city and charger type")
    c1, c2 = st.columns(2)
    c1.write("**Average cost by city (USD)**")
    c1.bar_chart(df.groupby("Charging Station Location")[TARGET].mean().sort_values(ascending=False))
    c2.write("**Average cost by charger type (USD)**")
    c2.bar_chart(df.groupby("Charger Type")[TARGET].mean())

    st.subheader("Objective 3: When do people charge?")
    c1, c2 = st.columns(2)
    c1.write("**Sessions by time of day**")
    c1.bar_chart(df["Time of Day"].value_counts().reindex(TIME_ORDER).fillna(0))
    c2.write("**Sessions by day of week**")
    c2.bar_chart(df["Day of Week"].value_counts().reindex(DAY_ORDER).fillna(0))

    wk = df.groupby("Is Weekend")[[TARGET, "Energy Consumed (kWh)", "Charging Duration (hours)"]].mean()
    wk.index = wk.index.map({0: "Weekday", 1: "Weekend"})
    st.write("Weekday vs weekend averages:")
    st.dataframe(wk.round(2), use_container_width=True)

    st.subheader("Distribution of any column")
    col = st.selectbox("Choose a column", [TARGET] + NUMERIC_COLS)
    bins = st.slider("Number of bins", 5, 60, 25)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(df[col], bins=bins, color="seagreen", edgecolor="white")
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean = {df[col].mean():.2f}")
    ax.set_xlabel(col); ax.set_ylabel("Sessions"); ax.legend()
    st.pyplot(fig)

elif page == "🔗 Relationships":
    st.title("🔗 Objective 4: Relationships between numeric columns")
    st.write("Correlation is between -1 and +1. Near 0 means no clear relationship.")
    corr = df[NUMERIC_COLS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("**Correlation with charging cost:**")
    st.dataframe(corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3),
                 use_container_width=True)

    st.subheader("Scatter plot")
    c1, c2 = st.columns(2)
    x = c1.selectbox("X axis", NUMERIC_COLS, index=1)
    y = c2.selectbox("Y axis", [TARGET] + NUMERIC_COLS, index=0)
    st.scatter_chart(df, x=x, y=y, color="Charger Type")

elif page == "🤖 ML Models":
    st.title("🤖 Objective 5: Predict Charging Cost")
    st.write("Change the settings and the models are trained again on the **filtered data**.")
    c1, c2, c3, c4 = st.columns(4)
    test_size = c1.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    k = c2.slider("KNN neighbours (k)", 1, 25, 5)
    depth = c3.slider("Decision tree depth", 1, 20, 5)
    trees = c4.slider("Random forest trees", 10, 300, 100, 10)

    if len(df) < 30:
        st.warning("Need at least 30 rows to train models. Widen the filters.")
        st.stop()

    with st.spinner("Training models..."):
        models, scaler, cols, results, y_test, preds = train_models(df, test_size, k, depth, trees)

    best = results.iloc[1:].sort_values("MAE").iloc[0]
    st.dataframe(results.style.highlight_min(subset=["MAE", "RMSE"], color="#c8e6c9")
                 .highlight_max(subset=["R2"], color="#c8e6c9"), use_container_width=True)
    base_mae = results.iloc[0]["MAE"]
    if best["MAE"] < base_mae:
        st.success(f"Best model: **{best['Model']}** (MAE ${best['MAE']}) – better than the baseline (${base_mae}).")
    else:
        st.warning(f"No model beat the simple baseline (MAE ${base_mae}). The data does not hold a strong pattern for cost.")

    c1, c2 = st.columns(2)
    c1.write("**MAE by model (lower is better)**")
    c1.bar_chart(results.set_index("Model")["MAE"])
    c2.write("**R² by model (higher is better)**")
    c2.bar_chart(results.set_index("Model")["R2"])

    st.subheader("Actual vs predicted")
    choice = st.selectbox("Model", list(preds))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, preds[choice], alpha=0.5, color="purple")
    lim = [0, max(y_test.max(), preds[choice].max())]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual cost (USD)"); ax.set_ylabel("Predicted cost (USD)"); ax.legend()
    st.pyplot(fig, use_container_width=False)

    st.subheader("Top 10 feature importances (Random Forest)")
    imp = pd.Series(models["Random Forest"].feature_importances_, index=cols).nlargest(10)
    st.bar_chart(imp)

elif page == "🔮 Predict Cost":
    st.title("🔮 Predict the Cost of a Charging Session")
    st.write("Fill in the details of a session and pick a model.")

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        inp = {}
        inp["Vehicle Model"] = c1.selectbox("Vehicle model", sorted(data["Vehicle Model"].unique()))
        inp["Charging Station Location"] = c1.selectbox("City", sorted(data["Charging Station Location"].unique()))
        inp["Charger Type"] = c1.selectbox("Charger type", sorted(data["Charger Type"].unique()))
        inp["User Type"] = c1.selectbox("User type", sorted(data["User Type"].unique()))
        inp["Time of Day"] = c2.selectbox("Time of day", TIME_ORDER)
        inp["Day of Week"] = c2.selectbox("Day of week", DAY_ORDER)
        inp["Battery Capacity (kWh)"] = c2.number_input("Battery capacity (kWh)", 1.0, 200.0, 75.0)
        inp["Energy Consumed (kWh)"] = c2.number_input("Energy consumed (kWh)", 0.0, 200.0, 42.0)
        inp["Charging Duration (hours)"] = c3.number_input("Duration (hours)", 0.0, 10.0, 2.3)
        inp["Charging Rate (kW)"] = c3.number_input("Charging rate (kW)", 0.0, 100.0, 26.0)
        inp["State of Charge (Start %)"] = c3.slider("Battery at start (%)", 0, 100, 30)
        inp["State of Charge (End %)"] = c3.slider("Battery at end (%)", 0, 100, 80)
        inp["Distance Driven (since last charge) (km)"] = c1.number_input("Distance driven (km)", 0.0, 1000.0, 150.0)
        inp["Temperature (°C)"] = c2.number_input("Temperature (°C)", -20.0, 50.0, 15.0)
        inp["Vehicle Age (years)"] = c3.number_input("Vehicle age (years)", 0.0, 20.0, 3.0)
        model_name = st.selectbox("Model", ["Linear Regression", "Decision Tree", "KNN (k=5)", "Random Forest"])
        go = st.form_submit_button("Predict cost")

    if go:
        models, scaler, cols, results, *_ = train_models(df, 0.2, 5, 5, 100)
        row = pd.DataFrame([inp])
        row["Is Weekend"] = int(inp["Day of Week"] in ["Saturday", "Sunday"])
        full = pd.concat([df.drop(columns=[TARGET]), row], ignore_index=True)
        X_row = pd.get_dummies(full, drop_first=True, dtype=int).reindex(columns=cols, fill_value=0).tail(1)
        model = models[model_name]
        pred = model.predict(scaler.transform(X_row) if model_name.startswith("KNN") else X_row)[0]
        mae = results.set_index("Model").loc[model_name, "MAE"]
        st.success(f"### Predicted cost: ${pred:.2f}")
        st.caption(f"This model is off by about ${mae} on average, so treat this as a rough estimate.")

elif page == "➕ Add Data":
    st.title("➕ Add a New Charging Session")
    st.write("Add your own session. It joins the dataset and all charts, stats and models update.")

    with st.form("add", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        new = {}
        new["Vehicle Model"] = c1.text_input("Vehicle model", "Tesla Model 3")
        new["Charging Station Location"] = c1.text_input("City", "Chicago")
        new["Charger Type"] = c1.selectbox("Charger type", sorted(data["Charger Type"].unique()))
        new["User Type"] = c1.selectbox("User type", sorted(data["User Type"].unique()))
        new["Time of Day"] = c1.selectbox("Time of day", TIME_ORDER)
        new["Day of Week"] = c2.selectbox("Day of week", DAY_ORDER)
        new["Battery Capacity (kWh)"] = c2.number_input("Battery capacity (kWh)", 1.0, 200.0, 75.0)
        new["Energy Consumed (kWh)"] = c2.number_input("Energy consumed (kWh)", 0.0, 200.0, 40.0)
        new["Charging Duration (hours)"] = c2.number_input("Duration (hours)", 0.0, 10.0, 2.0)
        new["Charging Rate (kW)"] = c2.number_input("Charging rate (kW)", 0.0, 100.0, 25.0)
        new[TARGET] = c3.number_input("Charging cost (USD)", 0.0, 200.0, 20.0)
        new["State of Charge (Start %)"] = c3.slider("Battery at start (%)", 0, 100, 25)
        new["State of Charge (End %)"] = c3.slider("Battery at end (%)", 0, 100, 85)
        new["Distance Driven (since last charge) (km)"] = c3.number_input("Distance driven (km)", 0.0, 1000.0, 120.0)
        new["Temperature (°C)"] = c3.number_input("Temperature (°C)", -20.0, 50.0, 20.0)
        new["Vehicle Age (years)"] = c3.number_input("Vehicle age (years)", 0.0, 20.0, 2.0)
        add = st.form_submit_button("Add session")

    if add:
        if new["State of Charge (End %)"] < new["State of Charge (Start %)"]:
            st.error("Battery at end should be higher than battery at start.")
        else:
            new["Is Weekend"] = int(new["Day of Week"] in ["Saturday", "Sunday"])
            st.session_state.data = pd.concat([data, pd.DataFrame([new])], ignore_index=True)
            st.success(f"Session added! The dataset now has {len(st.session_state.data)} rows.")
            st.rerun()

    st.subheader("Last 10 rows")
    st.dataframe(st.session_state.data.tail(10), use_container_width=True)

    st.subheader("Delete a row")
    idx = st.number_input("Row number to delete", 0, len(data) - 1, len(data) - 1)
    if st.button("Delete row"):
        st.session_state.data = data.drop(index=idx).reset_index(drop=True)
        st.rerun()

    st.download_button("Download updated dataset (CSV)", st.session_state.data.to_csv(index=False),
                       "ev_charging_updated.csv", "text/csv")
