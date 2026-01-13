import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="WHO Physical Activity — EDA", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

st.title("WHO Physical Activity (Insufficient Activity) — Global EDA")
st.markdown(
    "This dashboard explores WHO (World Health Organization) estimates of *insufficient physical activity among adults (18+)*. "
    "Use the filters to compare years and genders across countries."
)

DATA_PATH = "data/insufficient_activity.csv"
df_raw = load_data(DATA_PATH)

# Sidebar filters
st.sidebar.markdown(
    "Use the controls below to explore how physical activity levels "
    "change across years and between genders."
)

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

st.subheader("Preview (sample)")
st.dataframe(df.sample(20, random_state=42), use_container_width=True)

#st.subheader("Preview")
#st.dataframe(df.sort_values("sufficient_activity", ascending=False).head(20), use_container_width=True)

st.subheader("Trend over time (global average)")

# Глобальний тренд по роках для вибраної статі (Both sexes / Male / Female)
df_trend = (
    df_raw[df_raw["Dim1"] == sex]
    .groupby("Period")["FactValueNumeric"]
    .mean()
    .reset_index()
    .rename(columns={
        "Period": "year",
        "FactValueNumeric": "avg_insufficient_activity"
    })
    .dropna()
)

df_trend["avg_sufficient_activity"] = 100 - df_trend["avg_insufficient_activity"]

fig = plt.figure(figsize=(10, 4))
plt.plot(df_trend["year"], df_trend["avg_sufficient_activity"], marker="o")
plt.xlabel("Year")
plt.ylabel("Avg sufficient physical activity (%)")
plt.title("Global trend in physical activity over time")
st.pyplot(fig)
plt.close(fig)

st.caption(
    "This line chart shows how the global average level of physical activity changes over time. "
    "Use the 'Sex' filter to compare trends for males, females, and both sexes."
)

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
st.caption(
    "Most countries have a sufficient physical activity level between 60% and 85%. "
    "However, a noticeable group of countries falls below this range, indicating "
    "substantial global disparities in physical activity."
)

with colB:
    fig = plt.figure(figsize=(8, 3))
    sns.boxplot(x=df["sufficient_activity"])
    plt.xlabel("Sufficient physical activity (%)")
    plt.title("Boxplot")
    st.pyplot(fig)
    plt.close(fig)
st.caption(
    "The boxplot shows a relatively compact distribution of physical activity levels. "
    "Only a few countries exhibit very low activity levels, while extreme high values "
    "are rare, suggesting limited statistical outliers."
)
st.subheader("Clustering: groups of countries by activity level")

# Кластеризацію робимо на snapshot df (обраний year + sex)
X = df[["sufficient_activity"]].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Number of clusters (k)", min_value=2, max_value=6, value=3)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
st.markdown("### Countries in each cluster")

# 1) Вибір кластера
selected_cluster = st.selectbox(
    "Select cluster to view countries",
    sorted(df["cluster"].unique())
)

cluster_df = (
    df[df["cluster"] == selected_cluster]
    .sort_values("sufficient_activity", ascending=False)
    .reset_index(drop=True)
)

st.write(f"Countries in cluster {selected_cluster}: **{len(cluster_df)}**")

# 2) Таблиця країн в кластері
st.dataframe(
    cluster_df[["country", "sufficient_activity", "insufficient_activity"]],
    use_container_width=True
)

# 3) Пошук конкретної країни і показ її кластера
st.markdown("### Find a country")
country_search = st.selectbox("Choose a country", sorted(df["country"].unique()))
row = df[df["country"] == country_search].iloc[0]

st.info(
    f"**{country_search}** → **cluster {int(row['cluster'])}**, "
    f"**{row['sufficient_activity']:.1f}%** sufficient activity "
    f"({row['insufficient_activity']:.1f}% insufficient)."
)

# 4) Top/Bottom всередині кластера — дуже корисно і не перевантажує
top_n = st.slider("Show top/bottom N within selected cluster", 5, 30, 10)

col1, col2 = st.columns(2)
with col1:
    st.write("Top countries in this cluster")
    st.dataframe(
        cluster_df.nlargest(top_n, "sufficient_activity")[["country", "sufficient_activity"]],
        use_container_width=True
    )

with col2:
    st.write("Bottom countries in this cluster")
    st.dataframe(
        cluster_df.nsmallest(top_n, "sufficient_activity")[["country", "sufficient_activity"]],
        use_container_width=True
    )

st.caption(
    "Use the selector to inspect which countries belong to each cluster. "
    "This makes clustering results interpretable and directly answers the question about grouping countries."
)

# Скільки країн у кожному кластері
cluster_counts = df["cluster"].value_counts().sort_index().reset_index()
cluster_counts.columns = ["cluster", "countries"]

c1, c2 = st.columns(2)
with c1:
    st.write("Countries per cluster")
    st.dataframe(cluster_counts, use_container_width=True)

with c2:
    # Центри кластерів у вихідних % (повертаємо scale назад)
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled).flatten()
    centers_df = pd.DataFrame({"cluster": range(k), "center_sufficient_activity": centers}).sort_values("center_sufficient_activity")
    st.write("Cluster centers (sufficient activity %)")
    st.dataframe(centers_df, use_container_width=True)

# Візуалізація: stripplot по кластерах (наочніше для 1 фічі)
fig = plt.figure(figsize=(10, 4))
sns.stripplot(data=df, x="cluster", y="sufficient_activity", jitter=0.25)
plt.xlabel("Cluster")
plt.ylabel("Sufficient physical activity (%)")
plt.title("Clusters of countries by physical activity level")
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Clustering groups countries with similar physical activity levels. "
    "This answers whether we can identify distinct patterns across countries for the selected year and sex."
)

st.subheader("Heatmap: Activity by Region and Year")

if "ParentLocation" in df_raw.columns:
    # беремо Both sexes для стабільності порівняння
    heat_df = (
        df_raw[df_raw["Dim1"] == "Both sexes"]
        .groupby(["ParentLocation", "Period"])["FactValueNumeric"]
        .mean()
        .reset_index()
        .rename(columns={
            "ParentLocation": "region",
            "Period": "year",
            "FactValueNumeric": "avg_insufficient_activity"
        })
    )

    heat_df["avg_sufficient_activity"] = 100 - heat_df["avg_insufficient_activity"]

    pivot = heat_df.pivot(index="region", columns="year", values="avg_sufficient_activity")

    fig = plt.figure(figsize=(12, 4))
    sns.heatmap(pivot, annot=False)
    plt.xlabel("Year")
    plt.ylabel("Region")
    plt.title("Average sufficient physical activity (%) by region and year")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "This heatmap shows how average physical activity levels vary across regions over time. "
        "It helps spot persistent regional differences and long-term trends."
    )
else:
    st.info("Region information (ParentLocation) is not available in this dataset.")

# Top/Bottom
st.subheader("Top / Bottom countries")
st.markdown(
    # "### Top and bottom countries\n"
    "The tables below highlight countries with the highest and lowest levels of sufficient "
    "physical activity, helping identify extreme cases rather than statistical outliers."
)

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
