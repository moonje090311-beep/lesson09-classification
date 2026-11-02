import streamlit as st
import pandas as pd
import plotly.express as px


# 페이지 설정
st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="📊",
    layout="wide"
)

st.title("📊 데이터 탐색")
st.write("뇌졸중 데이터의 분포와 변수 사이의 관계를 살펴봅니다.")

# 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/moonje090311-beep/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# 뇌졸중 발생 여부 이름 변경
df["stroke_group"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# --------------------------------------------------
# 1. 나이와 평균 혈당 분포
# --------------------------------------------------

st.header("1. 나이와 평균 혈당 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"}
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )
    st.plotly_chart(fig_glucose, use_container_width=True)


st.divider()


# --------------------------------------------------
# 2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교
# --------------------------------------------------

st.header("2. 뇌졸중 여부에 따른 비교")

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        df,
        x="stroke_group",
        y="age",
        color="stroke_group",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "stroke_group": "뇌졸중 여부",
            "age": "나이"
        }
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col2:
    fig_glucose_box = px.box(
        df,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "stroke_group": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        }
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 두 그룹의 평균값
mean_table = (
    df.groupby("stroke_group")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .rename(columns={
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당"
    })
    .round(2)
)

st.subheader("뇌졸중 여부에 따른 평균값")
st.dataframe(mean_table, use_container_width=True)


st.divider()


# --------------------------------------------------
# 3. 고혈압과 심장병에 따른 뇌졸중 비율
# --------------------------------------------------

st.header("3. 질환 여부에 따른 뇌졸중 비율")


def make_rate_table(data, column):
    result = (
        data.groupby(column)["stroke"]
        .mean()
        .mul(100)
        .reset_index(name="뇌졸중 비율")
    )

    return result


col1, col2 = st.columns(2)

# 고혈압
with col1:
    hypertension_data = make_rate_table(
        df,
        "hypertension"
    )

    hypertension_data["고혈압 여부"] = (
        hypertension_data["hypertension"]
        .map({0: "없음", 1: "있음"})
    )

    fig_hypertension = px.bar(
        hypertension_data,
        x="고혈압 여부",
        y="뇌졸중 비율",
        title="고혈압 여부에 따른 뇌졸중 비율",
        text_auto=".2f",
        labels={
            "뇌졸중 비율": "뇌졸중 비율 (%)"
        }
    )

    st.plotly_chart(
        fig_hypertension,
        use_container_width=True
    )


# 심장병
with col2:
    heart_data = make_rate_table(
        df,
        "heart_disease"
    )

    heart_data["심장병 여부"] = (
        heart_data["heart_disease"]
        .map({0: "없음", 1: "있음"})
    )

    fig_heart = px.bar(
        heart_data,
        x="심장병 여부",
        y="뇌졸중 비율",
        title="심장병 여부에 따른 뇌졸중 비율",
        text_auto=".2f",
        labels={
            "뇌졸중 비율": "뇌졸중 비율 (%)"
        }
    )

    st.plotly_chart(
        fig_heart,
        use_container_width=True
    )


st.divider()


# --------------------------------------------------
# 4. BMI 결측값과 뇌졸중 비율
# --------------------------------------------------

st.header("4. BMI 빈 값과 뇌졸중 비율")

missing_bmi = df[df["bmi"].isna()]

missing_bmi_count = len(missing_bmi)
total_count = len(df)

missing_bmi_rate = (
    missing_bmi["stroke"].mean() * 100
    if missing_bmi_count > 0 else 0
)

total_stroke_rate = df["stroke"].mean() * 100

bmi_table = pd.DataFrame({
    "구분": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],
    "사람 수": [
        missing_bmi_count,
        total_count
    ],
    "뇌졸중 비율 (%)": [
        round(missing_bmi_rate, 2),
        round(total_stroke_rate, 2)
    ]
})

st.dataframe(
    bmi_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# 5. 흡연 상태별 사람 수
# --------------------------------------------------

st.header("5. 흡연 상태별 사람 수")

smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .reset_index()
)

smoking_table.columns = [
    "흡연 상태",
    "사람 수"
]

smoking_table["흡연 상태"] = (
    smoking_table["흡연 상태"]
    .fillna("빈 값")
)

st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True
)

st.caption("※ 이 페이지의 그래프와 수치는 제공된 원본 데이터를 바탕으로 계산했습니다.")
