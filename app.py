import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="WHO Physical Activity — EDA", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

st.title("WHO Physical Activity (Insufficient Activity) — Global EDA")

DATA_PATH = "data/insufficient_activity.csv"
df_raw = load_data(DATA_PATH)

# Sidebar filters
years = sorted(df_raw["Period"].dropna().unique())
default_year = 2019 if 2019 in years else years[-1]
year = st.sidebar.selectbox("Year", years, index=years.index(default_year))

sex_options = ["Both sexes", "Female", "Male"]
sex = st.sidebar.selectbox("Sex", sex_options, index=0)

df = (
    df_raw[(df_raw["Period"] == year) & (df_raw["Dim1"] == sex)]
    .loc[:, ["Location", "FactValueNumeric"]]
    .rename(columns={"Location": "country", "FactValueNumeric": "insufficient_activity"})
    .dropna(subset=["insufficient_activity"])
)

df["sufficient_activity"] = 100 - df["insufficient_activity"]

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Countries", f"{len(df):,}")
c2.metric("Avg insufficient activity (%)", f"{df['insufficient_activity'].mean():.2f}")
c3.metric("Avg sufficient activity (%)", f"{df['sufficient_activity'].mean():.2f}")

st.subheader("Preview")
st.dataframe(df.sort_values("sufficient_activity", ascending=False).head(20), use_container_width=True)

# Charts
st.subheader("Distribution")
colA, colB = st.columns(2)

with colA:
    fig = plt.figure(figsize=(8, 5))
    plt.hist(df["sufficient_activity"], bins=20)
    plt.xlabel("Sufficient physical activity (%)")
    plt.ylabel("Number of countries")
    plt.title("Histogram")
    st.pyplot(fig)
    plt.close(fig)

with colB:
    fig = plt.figure(figsize=(8, 3))
    sns.boxplot(x=df["sufficient_activity"])
    plt.xlabel("Sufficient physical activity (%)")
    plt.title("Boxplot")
    st.pyplot(fig)
    plt.close(fig)

# Top/Bottom
st.subheader("Top / Bottom countries")
colC, colD = st.columns(2)
with colC:
    st.write("Bottom 10 (lowest sufficient activity)")
    st.dataframe(df.nsmallest(10, "sufficient_activity"), use_container_width=True)
with colD:
    st.write("Top 10 (highest sufficient activity)")
    st.dataframe(df.nlargest(10, "sufficient_activity"), use_container_width=True)

# Region breakdown (optional)
if "ParentLocation" in df_raw.columns:
    st.subheader("By Region (Both sexes)")
    df_region = (
        df_raw[(df_raw["Period"] == year) & (df_raw["Dim1"] == "Both sexes")]
        .groupby("ParentLocation")["FactValueNumeric"]
        .mean()
        .reset_index()
        .rename(columns={"ParentLocation": "region", "FactValueNumeric": "avg_insufficient_activity"})
        .dropna()
    )
    df_region["avg_sufficient_activity"] = 100 - df_region["avg_insufficient_activity"]

    fig = plt.figure(figsize=(8, 5))
    sns.barplot(data=df_region, x="avg_sufficient_activity", y="region")
    plt.xlabel("Avg sufficient activity (%)")
    plt.ylabel("Region")
    plt.title("Average sufficient activity by region")
    st.pyplot(fig)
    plt.close(fig)
