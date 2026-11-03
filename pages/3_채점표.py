
import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 기본 설정 및 데이터 불러오기
# --------------------------------------------------

st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="📋",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "moonje090311-beep/data/refs/heads/main/stroke.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()


# --------------------------------------------------
# 2. 입력 속성 설정
# --------------------------------------------------

KOREAN_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURES = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease"
]

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]


# --------------------------------------------------
# 3. 사용할 속성 선택
# --------------------------------------------------

st.title("📋 채점표")
st.write(
    "테스트 데이터를 이용해 두 분류 모델의 예측 결과를 비교합니다."
)

selected_features = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요.",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda x: KOREAN_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해야 합니다.")
    st.stop()


# --------------------------------------------------
# 4. 번호 순 정렬 및 데이터 분리
# --------------------------------------------------

df = df.sort_values(
    "id",
    kind="mergesort"
).reset_index(drop=True)

# 10명씩 묶어 각 묶음의 앞 3명을 테스트 데이터로 사용
positions = list(range(len(df)))

test_mask = [
    position % 10 < 3
    for position in positions
]

train_df = df.loc[
    [not value for value in test_mask]
].copy()

test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

st.write(
    f"**테스트 데이터: {len(test_df):,}명** | "
    f"그중 실제 뇌졸중인 사람: "
    f"**{int(y_test.sum()):,}명**"
)

if len(test_df) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        "데이터 행 수에 따라 1,533명과 다를 수 있습니다."
    )


# --------------------------------------------------
# 5. BMI 결측값 처리
# --------------------------------------------------

# BMI를 선택한 경우에만 훈련 데이터 중앙값으로 채움
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train.loc[:, "bmi"] = (
        X_train["bmi"].fillna(bmi_median)
    )

    X_test.loc[:, "bmi"] = (
        X_test["bmi"].fillna(bmi_median)
    )

    st.info(
        f"체질량지수의 빈 값은 훈련 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )


# --------------------------------------------------
# 6. 로지스틱 회귀 학습
# --------------------------------------------------

# 크기 맞추기는 훈련 데이터로만 진행
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)


# --------------------------------------------------
# 7. 의사결정트리 학습
# --------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 8. 테스트 데이터 예측
# --------------------------------------------------

logistic_predictions = logistic_model.predict(
    X_test_scaled
)

tree_predictions = tree_model.predict(
    X_test
)

# 훈련 데이터에서 많은 쪽으로만 답하는 기준 모델
majority_class = int(
    y_train.value_counts().idxmax()
)

baseline_predictions = [
    majority_class
] * len(y_test)


# --------------------------------------------------
# 9. 혼동행렬 및 지표 계산 함수
# --------------------------------------------------

def calculate_confusion(actual, predicted):
    """
    행: 실제
    열: 예측

    TN: 실제 0, 예측 0
    FP: 실제 0, 예측 1
    FN: 실제 1, 예측 0
    TP: 실제 1, 예측 1
    """

    actual = list(actual)
    predicted = list(predicted)

    tn = sum(
        a == 0 and p == 0
        for a, p in zip(actual, predicted)
    )

    fp = sum(
        a == 0 and p == 1
        for a, p in zip(actual, predicted)
    )

    fn = sum(
        a == 1 and p == 0
        for a, p in zip(actual, predicted)
    )

    tp = sum(
        a == 1 and p == 1
        for a, p in zip(actual, predicted)
    )

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def calculate_metrics(confusion):
    tn = confusion["TN"]
    fp = confusion["FP"]
    fn = confusion["FN"]
    tp = confusion["TP"]

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0
        else None
    )

    # 재현율 = TP / 실제 뇌졸중인 사람 수
    actual_positive = tp + fn

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else None
    )

    # 정밀도 = TP / 뇌졸중이라고 예측한 사람 수
    predicted_positive = tp + fp

    if predicted_positive == 0:
        precision = None
        f1 = None
    else:
        precision = tp / predicted_positive

        # F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)
        if recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# --------------------------------------------------
# 10. 각 모델의 혼동행렬 계산
# --------------------------------------------------

logistic_confusion = calculate_confusion(
    y_test,
    logistic_predictions
)

tree_confusion = calculate_confusion(
    y_test,
    tree_predictions
)

baseline_confusion = calculate_confusion(
    y_test,
    baseline_predictions
)


# --------------------------------------------------
# 11. 혼동행렬 표시 함수
# --------------------------------------------------

def make_confusion_table(confusion):
    tn = confusion["TN"]
    fp = confusion["FP"]
    fn = confusion["FN"]
    tp = confusion["TP"]

    actual_0_total = tn + fp
    actual_1_total = fn + tp

    predicted_0_total = tn + fn
    predicted_1_total = fp + tp

    total = tn + fp + fn + tp

    return pd.DataFrame(
        [
            [
                f"TN: {tn:,}명",
                f"FP: {fp:,}명",
                f"{actual_0_total:,}명"
            ],
            [
                f"FN: {fn:,}명",
                f"TP: {tp:,}명",
                f"{actual_1_total:,}명"
            ],
            [
                f"{predicted_0_total:,}명",
                f"{predicted_1_total:,}명",
                f"{total:,}명"
            ]
        ],
        index=[
            "실제 뇌졸중 아님(0)",
            "실제 뇌졸중 있음(1)",
            "열 합계"
        ],
        columns=[
            "예측 뇌졸중 아님(0)",
            "예측 뇌졸중 있음(1)",
            "행 합계"
        ]
    )


# --------------------------------------------------
# 12. 로지스틱 회귀 혼동행렬
# --------------------------------------------------

st.divider()
st.header("1. 로지스틱 회귀(확률로 답하는 모델)")

st.write(
    "행은 실제 뇌졸중 여부이고, 열은 모델이 예측한 뇌졸중 여부입니다."
)

st.dataframe(
    make_confusion_table(logistic_confusion),
    use_container_width=True
)


# --------------------------------------------------
# 13. 의사결정트리 혼동행렬
# --------------------------------------------------

st.divider()
st.header("2. 의사결정트리(질문으로 답하는 모델)")

st.dataframe(
    make_confusion_table(tree_confusion),
    use_container_width=True
)


# --------------------------------------------------
# 14. 네 가지 지표 비교
# --------------------------------------------------

st.divider()
st.header("3. 모델 성능 지표 비교")

logistic_metrics = calculate_metrics(
    logistic_confusion
)

tree_metrics = calculate_metrics(
    tree_confusion
)

baseline_metrics = calculate_metrics(
    baseline_confusion
)

metrics_table = pd.DataFrame(
    [
        {
            "모델": "로지스틱 회귀(확률로 답하는 모델)",
            "정확도": format_metric(
                logistic_metrics["정확도"]
            ),
            "재현율": format_metric(
                logistic_metrics["재현율"]
            ),
            "정밀도": format_metric(
                logistic_metrics["정밀도"]
            ),
            "F1": format_metric(
                logistic_metrics["F1"]
            )
        },
        {
            "모델": "의사결정트리(질문으로 답하는 모델)",
            "정확도": format_metric(
                tree_metrics["정확도"]
            ),
            "재현율": format_metric(
                tree_metrics["재현율"]
            ),
            "정밀도": format_metric(
                tree_metrics["정밀도"]
            ),
            "F1": format_metric(
                tree_metrics["F1"]
            )
        },
        {
            "모델": "훈련용 다수 클래스만 답하는 모델",
            "정확도": format_metric(
                baseline_metrics["정확도"]
            ),
            "재현율": format_metric(
                baseline_metrics["재현율"]
            ),
            "정밀도": format_metric(
                baseline_metrics["정밀도"]
            ),
            "F1": format_metric(
                baseline_metrics["F1"]
            )
        }
    ]
)

st.dataframe(
    metrics_table,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "정확도 = (TP + TN) / 전체 인원, "
    "재현율 = TP / (TP + FN), "
    "정밀도 = TP / (TP + FP)입니다. "
    "F1은 정밀도와 재현율의 조화평균입니다."
)


# --------------------------------------------------
# 15. 놓친 환자와 헛짚은 사람 목록을 위한 모델 선택
# --------------------------------------------------

st.divider()
st.header("4. 잘못 예측한 사람 확인")

model_options = {
    "로지스틱 회귀(확률로 답하는 모델)": (
        logistic_predictions,
        logistic_confusion
    ),
    "의사결정트리(질문으로 답하는 모델)": (
        tree_predictions,
        tree_confusion
    ),
    "훈련용 다수 클래스만 답하는 모델": (
        baseline_predictions,
        baseline_confusion
    )
}

selected_model_name = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    options=list(model_options.keys())
)

selected_predictions, selected_confusion = (
    model_options[selected_model_name]
)


# --------------------------------------------------
# 16. 실제값과 예측값을 비교해 목록 만들기
# --------------------------------------------------

result_df = test_df[
    [
        "id",
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ]
].copy()

result_df["actual"] = y_test.to_numpy()
result_df["predicted"] = selected_predictions

# 실제 뇌졸중인데 아니라고 예측한 사람(FN)
missed_people = result_df[
    (result_df["actual"] == 1)
    & (result_df["predicted"] == 0)
].copy()

# 실제 뇌졸중이 아닌데 뇌졸중이라고 예측한 사람(FP)
false_alarm_people = result_df[
    (result_df["actual"] == 0)
    & (result_df["predicted"] == 1)
].copy()

# 나이가 많은 순서로 정렬
missed_people = missed_people.sort_values(
    "age",
    ascending=False
)

false_alarm_people = false_alarm_people.sort_values(
    "age",
    ascending=False
)


# --------------------------------------------------
# 17. 목록 표시용 데이터 정리
# --------------------------------------------------

def prepare_display_table(data):
    display = data[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease"
        ]
    ].copy()

    display = display.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부"
        }
    )

    display["고혈압 여부"] = display[
        "고혈압 여부"
    ].map({
        0: "없음",
        1: "있음"
    })

    display["심장병 여부"] = display[
        "심장병 여부"
    ].map({
        0: "없음",
        1: "있음"
    })

    return display.head(20)


# --------------------------------------------------
# 18. 두 목록 나란히 표시
# --------------------------------------------------

left_col, right_col = st.columns(2)

with left_col:
    st.subheader(
        f"놓친 환자(FN) · 전체 {len(missed_people):,}명"
    )

    st.write(
        "실제로 뇌졸중이 있지만 모델이 아니라고 예측한 사람입니다."
    )

    if len(missed_people) > 0:
        st.dataframe(
            prepare_display_table(missed_people),
            hide_index=True,
            use_container_width=True
        )

        if len(missed_people) > 20:
            st.caption("나이가 많은 순서로 앞의 20명만 표시했습니다.")
    else:
        st.info("놓친 환자가 없습니다.")

with right_col:
    st.subheader(
        f"헛짚은 사람(FP) · 전체 {len(false_alarm_people):,}명"
    )

    st.write(
        "실제로는 뇌졸중이 아니지만 모델이 있다고 예측한 사람입니다."
    )

    if len(false_alarm_people) > 0:
        st.dataframe(
            prepare_display_table(false_alarm_people),
            hide_index=True,
            use_container_width=True
        )

        if len(false_alarm_people) > 20:
            st.caption("나이가 많은 순서로 앞의 20명만 표시했습니다.")
    else:
        st.info("헛짚은 사람이 없습니다.")


# --------------------------------------------------
# 19. 혼동행렬과 목록 인원 비교
# --------------------------------------------------

st.divider()
st.subheader("목록과 혼동행렬 비교")

fn_count = selected_confusion["FN"]
fp_count = selected_confusion["FP"]

st.write(
    f"놓친 환자 목록 **{len(missed_people):,}명** "
    f"= 혼동행렬의 FN **{fn_count:,}명**"
)

st.write(
    f"헛짚은 사람 목록 **{len(false_alarm_people):,}명** "
    f"= 혼동행렬의 FP **{fp_count:,}명**"
)

if (
    len(missed_people) == fn_count
    and len(false_alarm_people) == fp_count
):
    st.success(
        "두 목록의 인원수가 혼동행렬의 FN·FP 값과 일치합니다."
    )
else:
    st.error(
        "목록과 혼동행렬의 인원수가 일치하지 않습니다."
    )


# --------------------------------------------------
# 20. 안내
# --------------------------------------------------

st.divider()

st.caption(
    "이 페이지는 뇌졸중 데이터로 분류 모델의 성능을 비교하는 "
    "교육용 실습입니다. 실제 진단이나 건강 상태 판단에 "
    "사용할 수 없습니다."
)
