import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 설정 (브라우저 탭 제목 및 아이콘 지정)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

# 화면 상단 제목 및 어울리는 아이콘
st.title("🎬 영화 유형 나누기 (K-Means 클러스터링)")
st.caption("KOBIS 영화 데이터를 바탕으로 속성별 표준화 및 K-Means 군집화를 수행하여 영화의 유형을 분석합니다.")

# 2. 데이터 불러오기 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 데이터 가공 및 필터링
# 롱런 지수 생성: 누적 관객 / 첫 주 관객 (20 초과 시 20으로 클리핑)
df_raw["long_run"] = df_raw["total_audi"] / df_raw["first_week_audi"]
df_raw["long_run"] = df_raw["long_run"].clip(upper=20.0)

# 조건 필터링: 네 속성 중 값이 없거나 첫 주 관객이 0 이하(로그 계산 불가 포함)인 데이터 제외
valid_mask = (
    df_raw["first_scrn"].notna() & (df_raw["first_scrn"] > 0) &
    df_raw["total_audi"].notna() & (df_raw["total_audi"] > 0) &
    df_raw["days_in_top10"].notna() &
    df_raw["first_week_audi"].notna() & (df_raw["first_week_audi"] > 0)
)

df_clean = df_raw[valid_mask].copy()

# 스크린 수와 누적 관객 상용로그(log10) 변환
df_clean["log_first_scrn"] = np.log10(df_clean["first_scrn"])
df_clean["log_total_audi"] = np.log10(df_clean["total_audi"])

total_count = len(df_raw)
used_count = len(df_clean)

# 전체 편수와 묶은 편수를 한 줄로 적기
st.markdown(f"**전체 영화 편수:** `{total_count:,}`편 &nbsp;|&nbsp; **분석에 사용된(묶은) 영화 편수:** `{used_count:,}`편")
st.divider()

# 4. 묶는 데 사용할 속성 선택 UI (화면 위쪽)
feature_map = {
    "스크린 수 (상용로그)": "log_first_scrn",
    "누적 관객 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run"
}

st.subheader("⚙️ 묶는 데 사용할 속성 선택")
selected_labels = st.multiselect(
    "군집화에 활용할 속성을 고르세요 (최소 2개 이상 선택, 기본 4개 모두 선택):",
    options=list(feature_map.keys()),
    default=list(feature_map.keys())
)

# 둘 이상 고르지 않은 경우 경고 후 정지
if len(selected_labels) < 2:
    st.warning("⚠️ 영화 유형을 나누려면 속성을 최소 2개 이상 선택해야 합니다.")
    st.stop()

# 5. 데이터 표준화 및 K-Means 군집화 (난수 고정 random_state=42)
selected_cols = [feature_map[label] for label in selected_labels]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[selected_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean["raw_cluster"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 묶음 순서대로 ㉮, ㉯, ㉰ 부여
audi_means = df_clean.groupby("raw_cluster")["total_audi"].mean()
sorted_clusters = audi_means.sort_values(ascending=False).index.tolist()

cluster_name_map = {
    sorted_clusters[0]: "㉮",
    sorted_clusters[1]: "㉯",
    sorted_clusters[2]: "㉰"
}

df_clean["cluster"] = df_clean["raw_cluster"].map(cluster_name_map)
cluster_order = ["㉮", "㉯", "㉰"]
color_map = {"㉮": "#EF553B", "㉯": "#00CC96", "㉰": "#636EFA"}

# 6. 2차원 산점도 (Plotly)
st.subheader("📊 2차원 산점도")
col2d_1, col2d_2 = st.columns(2)

with col2d_1:
    x_2d_label = st.selectbox("가로축(X축) 속성:", options=selected_labels, index=0, key="2d_x")
with col2d_2:
    y_2d_label = st.selectbox("세로축(Y축) 속성:", options=selected_labels, index=1 if len(selected_labels) > 1 else 0, key="2d_y")

x_2d_col = feature_map[x_2d_label]
y_2d_col = feature_map[y_2d_label]

fig_2d = px.scatter(
    df_clean,
    x=x_2d_col,
    y=y_2d_col,
    color="cluster",
    category_orders={"cluster": cluster_order},
    color_discrete_map=color_map,
    hover_name="movieNm",  # 점에 마우스를 올리면 영화 제목 표시
    hover_data={
        "total_audi": ":,",
        "first_scrn": ":,",
        "days_in_top10": True,
        "long_run": ":.2f",
        x_2d_col: False,
        y_2d_col: False
    },
    labels={
        x_2d_col: x_2d_label,
        y_2d_col: y_2d_label,
        "cluster": "묶음"
    },
    title=f"2차원 산점도 ({x_2d_label} vs {y_2d_label})"
)
fig_2d.update_layout(legend_title_text="묶음")
st.plotly_chart(fig_2d, use_container_width=True)

# 7. 3차원 산점도 (Plotly)
st.subheader("🧊 3차원 산점도")
if len(selected_labels) < 3:
    st.info("💡 3차원 산점도를 그리려면 선택된 속성이 3개 이상이어야 합니다. 위에서 속성을 3개 이상 선택해 주세요.")
else:
    col3d_1, col3d_2, col3d_3 = st.columns(3)
    with col3d_1:
        x_3d_label = st.selectbox("3차원 X축 속성:", options=selected_labels, index=0, key="3d_x")
    with col3d_2:
        y_3d_label = st.selectbox("3차원 Y축 속성:", options=selected_labels, index=1, key="3d_y")
    with col3d_3:
        z_3d_label = st.selectbox("3차원 Z축 속성:", options=selected_labels, index=2, key="3d_z")

    x_3d_col = feature_map[x_3d_label]
    y_3d_col = feature_map[y_3d_label]
    z_3d_col = feature_map[z_3d_label]

    fig_3d = px.scatter_3d(
        df_clean,
        x=x_3d_col,
        y=y_3d_col,
        z=z_3d_col,
        color="cluster",
        category_orders={"cluster": cluster_order},
        color_discrete_map=color_map,
        hover_name="movieNm",  # 마우스 호버 시 영화 제목 표시
        hover_data={
            "total_audi": ":,",
            "first_scrn": ":,",
            "days_in_top10": True,
            "long_run": ":.2f",
            x_3d_col: False,
            y_3d_col: False,
            z_3d_col: False
        },
        labels={
            x_3d_col: x_3d_label,
            y_3d_col: y_3d_label,
            z_3d_col: z_3d_label,
            "cluster": "묶음"
        },
        title=f"3차원 산점도 ({x_3d_label} × {y_3d_label} × {z_3d_label})"
    )
    # 점을 작게 찍고, 마우스 드래그로 돌릴 수 있도록 설정
    fig_3d.update_traces(marker=dict(size=3))
    fig_3d.update_layout(legend_title_text="묶음")
    st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# 8. 묶음마다 편수 및 4개 속성의 평균(원래 단위) 표
st.subheader("📋 묶음별 특징 요약 (원래 단위 평균)")

summary_rows = []
for c in cluster_order:
    sub = df_clean[df_clean["cluster"] == c]
    summary_rows.append({
        "묶음": c,
        "영화 편수": len(sub),
        "스크린 수 평균": sub["first_scrn"].mean(),
        "누적 관객 평균": sub["total_audi"].mean(),
        "10위권 일수 평균": sub["days_in_top10"].mean(),
        "롱런 지수 평균": sub["long_run"].mean()
    })

summary_df = pd.DataFrame(summary_rows)

# 표시용 포맷팅 (원래 단위 표기)
summary_df_display = summary_df.copy()
summary_df_display["영화 편수"] = summary
