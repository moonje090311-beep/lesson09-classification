
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 기본 설정 및 데이터 불러오기
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기 | 뇌졸중 예측 실습실",
    page_icon="📈",
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

st.title("📈 성능 높이기")

st.write(
    "뇌졸중을 겪은 사람은 전체 데이터에서 적은 비율을 차지합니다. "
    "클래스 가중치를 적용하기 전과 적용한 후의 결과를 비교해 보세요."
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
# 4. 번호 순 정렬 및 훈련·테스트 데이터 분리
# --------------------------------------------------

df = df.sort_values(
    "id",
    kind="mergesort"
).reset_index(drop=True)

positions = list(range(len(df)))

# 열 명씩 묶어 앞의 세 명은 테스트용
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
    f"훈련 데이터: **{len(train_df):,}명** | "
    f"테스트 데이터: **{len(test_df):,}명** | "
    f"테스트 데이터 중 실제 뇌졸중: "
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

if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train.loc[:, "bmi"] = (
        X_train["bmi"].fillna(bmi_median)
    )

    X_test.loc[:, "bmi"] = (
        X_test["bmi"].fillna(bmi_median)
    )

    st.info(
        f"체질량지수의 빈 값은 훈련 데이터 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )


# --------------------------------------------------
# 6. 크기 맞추기
# --------------------------------------------------

# 훈련 데이터로만 평균과 표준편차를 계산
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 7. 로지스틱 회귀 모델 학습
# --------------------------------------------------

# 가중치 적용 전
logistic_before = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_before.fit(
    X_train_scaled,
    y_train
)

# 가중치 적용 후
logistic_after = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_after.fit(
    X_train_scaled,
    y_train
)


# --------------------------------------------------
# 8. 의사결정트리 모델 학습
# --------------------------------------------------

# 가중치 적용 전
tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_before.fit(
    X_train,
    y_train
)

# 가중치 적용 후
tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

tree_after.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 9. 테스트 데이터 예측
# --------------------------------------------------

predictions = {
    "로지스틱 회귀(확률로 답하는 모델) - 적용 전":
        logistic_before.predict(X_test_scaled),

    "로지스틱 회귀(확률로 답하는 모델) - 적용 후":
        logistic_after.predict(X_test_scaled),

    "의사결정트리(질문으로 답하는 모델) - 적용 전":
        tree_before.predict(X_test),

    "의사결정트리(질문으로 답하는 모델) - 적용 후":
        tree_after.predict(X_test)
}


# --------------------------------------------------
# 10. 혼동행렬 계산
# --------------------------------------------------

def calculate_confusion(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tp = int(np.sum((actual == 1) & (predicted == 1)))

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


confusions = {
    name: calculate_confusion(y_test, prediction)
    for name, prediction in predictions.items()
}


# --------------------------------------------------
# 11. 네 가지 지표 계산
# --------------------------------------------------

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


metrics = {
    name: calculate_metrics(confusion)
    for name, confusion in confusions.items()
}


def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# --------------------------------------------------
# 12. 혼동행렬 전후 비교
# --------------------------------------------------

st.divider()
st.header("1. 혼동행렬 비교")

st.write(
    "행은 실제 뇌졸중 여부이고, 열은 모델의 예측 결과입니다."
)

def make_confusion_comparison(before, after):
    tn_b = before["TN"]
    fp_b = before["FP"]
    fn_b = before["FN"]
    tp_b = before["TP"]

    tn_a = after["TN"]
    fp_a = after["FP"]
    fn_a = after["FN"]
    tp_a = after["TP"]

    return pd.DataFrame(
        [
            [
                f"TN: {tn_b:,}명 → {tn_a:,}명",
                f"FP: {fp_b:,}명 → {fp_a:,}명",
                f"{tn_b + fp_b:,}명"
            ],
            [
                f"FN: {fn_b:,}명 → {fn_a:,}명",
                f"TP: {tp_b:,}명 → {tp_a:,}명",
                f"{fn_b + tp_b:,}명"
            ],
            [
                f"{tn_b + fn_b:,}명 → {tn_a + fn_a:,}명",
                f"{fp_b + tp_b:,}명 → {fp_a + tp_a:,}명",
                f"{tn_b + fp_b + fn_b + tp_b:,}명"
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


col1, col2 = st.columns(2)

with col1:
    st.subheader("로지스틱 회귀(확률로 답하는 모델)")

    st.caption("가중치 적용 전 → 적용 후")

    st.dataframe(
        make_confusion_comparison(
            confusions[
                "로지스틱 회귀(확률로 답하는 모델) - 적용 전"
            ],
            confusions[
                "로지스틱 회귀(확률로 답하는 모델) - 적용 후"
            ]
        ),
        use_container_width=True
    )

with col2:
    st.subheader("의사결정트리(질문으로 답하는 모델)")

    st.caption("가중치 적용 전 → 적용 후")

    st.dataframe(
        make_confusion_comparison(
            confusions[
                "의사결정트리(질문으로 답하는 모델) - 적용 전"
            ],
            confusions[
                "의사결정트리(질문으로 답하는 모델) - 적용 후"
            ]
        ),
        use_container_width=True
    )


# --------------------------------------------------
# 13. 네 가지 지표 전후 비교
# --------------------------------------------------

st.divider()
st.header("2. 성능 지표 비교")

def make_metric_comparison(model_prefix):
    before_name = model_prefix + " - 적용 전"
    after_name = model_prefix + " - 적용 후"

    before_metrics = metrics[before_name]
    after_metrics = metrics[after_name]

    rows = []

    for metric_name in [
        "정확도",
        "재현율",
        "정밀도",
        "F1"
    ]:
        rows.append({
            "지표": metric_name,
            "가중치 적용 전": format_metric(
                before_metrics[metric_name]
            ),
            "가중치 적용 후": format_metric(
                after_metrics[metric_name]
            )
        })

    return pd.DataFrame(rows)


metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.subheader("로지스틱 회귀(확률로 답하는 모델)")

    st.dataframe(
        make_metric_comparison(
            "로지스틱 회귀(확률로 답하는 모델)"
        ),
        hide_index=True,
        use_container_width=True
    )

with metric_col2:
    st.subheader("의사결정트리(질문으로 답하는 모델)")

    st.dataframe(
        make_metric_comparison(
            "의사결정트리(질문으로 답하는 모델)"
        ),
        hide_index=True,
        use_container_width=True
    )


# --------------------------------------------------
# 14. 찾아낸 환자·놓친 환자·헛짚은 사람 변화
# --------------------------------------------------

st.divider()
st.header("3. 뇌졸중 예측 결과 변화")

def count_changes(model_prefix):
    before = confusions[model_prefix + " - 적용 전"]
    after = confusions[model_prefix + " - 적용 후"]

    return (
        f"찾아낸 환자(TP): {before['TP']:,}명 → {after['TP']:,}명  |  "
        f"놓친 환자(FN): {before['FN']:,}명 → {after['FN']:,}명  |  "
        f"헛짚은 사람(FP): {before['FP']:,}명 → {after['FP']:,}명"
    )


st.subheader("로지스틱 회귀(확률로 답하는 모델)")
st.write(
    count_changes("로지스틱 회귀(확률로 답하는 모델)")
)

st.subheader("의사결정트리(질문으로 답하는 모델)")
st.write(
    count_changes("의사결정트리(질문으로 답하는 모델)")
)


# --------------------------------------------------
# 15. 축 선택
# --------------------------------------------------

st.divider()
st.header("4. 가중치 적용 전후의 분류 경계 비교")

st.write(
    "테스트 데이터의 실제 뇌졸중 여부를 점으로 표시하고, "
    "로지스틱 회귀의 0.5 경계선을 가중치 적용 전후로 비교합니다."
)

if len(selected_features) == 2:
    axis_col1, axis_col2 = st.columns(2)

    with axis_col1:
        x_feature = st.selectbox(
            "가로축",
            selected_features,
            index=0,
            format_func=lambda x: KOREAN_NAMES[x]
        )

    y_options = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    with axis_col2:
        y_feature = st.selectbox(
            "세로축",
            y_options,
            format_func=lambda x: KOREAN_NAMES[x]
        )

    axes = [x_feature, y_feature]

else:
    st.write("세 가지 속성을 골라 3차원 산점도의 축으로 사용합니다.")

    axis_col1, axis_col2, axis_col3 = st.columns(3)

    with axis_col1:
        x_feature = st.selectbox(
            "가로축",
            selected_features,
            index=0,
            format_func=lambda x: KOREAN_NAMES[x]
        )

    remaining_1 = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    with axis_col2:
        y_feature = st.selectbox(
            "세로축",
            remaining_1,
            index=0,
            format_func=lambda x: KOREAN_NAMES[x]
        )

    remaining_2 = [
        feature
        for feature in selected_features
        if feature not in [x_feature, y_feature]
    ]

    with axis_col3:
        z_feature = st.selectbox(
            "높이축",
            remaining_2,
            index=0,
            format_func=lambda x: KOREAN_NAMES[x]
        )

    axes = [x_feature, y_feature, z_feature]


# --------------------------------------------------
# 16. 나머지 속성의 고정값
# --------------------------------------------------

fixed_values = {}

for feature in selected_features:
    if feature not in axes:
        fixed_values[feature] = float(
            X_test[feature].median()
        )

if fixed_values:
    fixed_text = ", ".join(
        [
            f"{KOREAN_NAMES[feature]}: {value:.2f}"
            for feature, value in fixed_values.items()
        ]
    )

    st.write(
        "**축으로 사용하지 않은 속성은 테스트 데이터의 중앙값으로 고정했습니다.**"
    )
    st.write(fixed_text)
else:
    st.write("선택한 모든 속성을 축으로 사용합니다.")


# --------------------------------------------------
# 17. 그림에 사용할 축 범위
# --------------------------------------------------

def make_axis_values(series, count=45):
    minimum = float(series.min())
    maximum = float(series.max())

    if minimum == maximum:
        return np.linspace(
            minimum - 1,
            maximum + 1,
            count
        )

    margin = (maximum - minimum) * 0.05

    return np.linspace(
        minimum - margin,
        maximum + margin,
        count
    )


# --------------------------------------------------
# 18. 로지스틱 회귀의 확률 계산 함수
# --------------------------------------------------

def make_grid_2d(x_values, y_values):
    rows = []

    for y_value in y_values:
        for x_value in x_values:
            row = {}

            for feature in selected_features:
                if feature == x_feature:
                    row[feature] = x_value
                elif feature == y_feature:
                    row[feature] = y_value
                else:
                    row[feature] = fixed_values[feature]

            rows.append(row)

    return pd.DataFrame(
        rows,
        columns=selected_features
    )


def get_logistic_probabilities(model, grid):
    scaled_grid = scaler.transform(grid)

    positive_index = list(model.classes_).index(1)

    return model.predict_proba(
        scaled_grid
    )[:, positive_index]


# --------------------------------------------------
# 19. 2차원 산점도와 경계선
# --------------------------------------------------

if len(selected_features) == 2:
    x_values = make_axis_values(X_test[x_feature])
    y_values = make_axis_values(X_test[y_feature])

    grid = make_grid_2d(
        x_values,
        y_values
    )

    probabilities_before = get_logistic_probabilities(
        logistic_before,
        grid
    )

    probabilities_after = get_logistic_probabilities(
        logistic_after,
        grid
    )

    z_before = probabilities_before.reshape(
        len(y_values),
        len(x_values)
    )

    z_after = probabilities_after.reshape(
        len(y_values),
        len(x_values)
    )

    fig = go.Figure()

    # 가중치 적용 전 경계선
    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=z_before,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="none",
                showlabels=False
            ),
            line=dict(
                color="blue",
                width=3
            ),
            showscale=False,
            hoverinfo="skip",
            name="가중치 적용 전 0.5 경계선"
        )
    )

    # 가중치 적용 후 경계선
    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=z_after,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring="none",
                showlabels=False
            ),
            line=dict(
                color="orange",
                width=3
            ),
            showscale=False,
            hoverinfo="skip",
            name="가중치 적용 후 0.5 경계선"
        )
    )

    # 실제 뇌졸중 여부에 따라 테스트 데이터 표시
    no_stroke = test_df.loc[
        test_df["stroke"] == 0
    ]

    stroke = test_df.loc[
        test_df["stroke"] == 1
    ]

    fig.add_trace(
        go.Scatter(
            x=no_stroke[x_feature],
            y=no_stroke[y_feature],
            mode="markers",
            name="뇌졸중 아님(0)",
            marker=dict(
                color="royalblue",
                size=6,
                opacity=0.65
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stroke[x_feature],
            y=stroke[y_feature],
            mode="markers",
            name="뇌졸중 있음(1)",
            marker=dict(
                color="red",
                size=7,
                opacity=0.8
            )
        )
    )

    fig.update_layout(
        title="가중치 적용 전후의 0.5 경계선",
        xaxis_title=KOREAN_NAMES[x_feature],
        yaxis_title=KOREAN_NAMES[y_feature],
        height=650,
        legend_title="실제 뇌졸중 여부",
        hovermode="closest"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # 경계선이 그림 범위 안에 있는지 확인
    before_inside = (
        np.min(probabilities_before) <= 0.5
        and np.max(probabilities_before) >= 0.5
    )

    after_inside = (
        np.min(probabilities_after) <= 0.5
        and np.max(probabilities_after) >= 0.5
    )

    st.write(
        "파란색: 가중치 적용 전 / 주황색: 가중치 적용 후"
    )

    if before_inside:
        st.success("가중치 적용 전 경계선은 그림 범위 안에 있습니다.")
    else:
        st.warning("가중치 적용 전 경계선은 그림 범위 밖에 있습니다.")

    if after_inside:
        st.success("가중치 적용 후 경계선은 그림 범위 안에 있습니다.")
    else:
        st.warning("가중치 적용 후 경계선은 그림 범위 밖에 있습니다.")


# --------------------------------------------------
# 20. 3차원 산점도와 경계면
# --------------------------------------------------

else:
    x_values = make_axis_values(X_test[x_feature], count=24)
    y_values = make_axis_values(X_test[y_feature], count=24)
    z_values = make_axis_values(X_test[z_feature], count=24)

    grid_rows = []

    for z_value in z_values:
        for y_value in y_values:
            for x_value in x_values:
                row = {}

                for feature in selected_features:
                    if feature == x_feature:
                        row[feature] = x_value
                    elif feature == y_feature:
                        row[feature] = y_value
                    elif feature == z_feature:
                        row[feature] = z_value
                    else:
                        row[feature] = fixed_values[feature]

                grid_rows.append(row)

    grid_3d = pd.DataFrame(
        grid_rows,
        columns=selected_features
    )

    prob_before_3d = get_logistic_probabilities(
        logistic_before,
        grid_3d
    )

    prob_after_3d = get_logistic_probabilities(
        logistic_after,
        grid_3d
    )

    # 로지스틱 회귀의 결정 경계는
    # 표준화된 공간에서 계수와 절편으로 표현되는 평면입니다.
    # 원래 단위의 좌표로 변환해 평면을 계산합니다.

    def make_plane(model, color, name):
        coefficients = (
            model.coef_[0] / scaler.scale_
        )

        intercept = (
            model.intercept_[0]
            - np.sum(
                model.coef_[0]
                * scaler.mean_
                / scaler.scale_
            )
        )

        feature_indices = {
            feature: selected_features.index(feature)
            for feature in selected_features
        }

        axis_coefficients = [
            coefficients[feature_indices[feature]]
            for feature in axes
        ]

        # 계수의 절댓값이 가장 큰 축을 높이 계산 축으로 선택
        dependent_index = int(
            np.argmax(np.abs(axis_coefficients))
        )

        dependent_coefficient = axis_coefficients[
            dependent_index
        ]

        if abs(dependent_coefficient) < 1e-12:
            return None

        other_indices = [
            index
            for index in range(3)
            if index != dependent_index
        ]

        first_axis = axes[other_indices[0]]
        second_axis = axes[other_indices[1]]
        dependent_axis = axes[dependent_index]

        first_values = make_axis_values(
            X_test[first_axis],
            count=30
        )

        second_values = make_axis_values(
            X_test[second_axis],
            count=30
        )

        first_grid, second_grid = np.meshgrid(
            first_values,
            second_values
        )

        fixed_sum = intercept

        for feature in selected_features:
            if feature not in axes:
                fixed_sum += (
                    coefficients[feature_indices[feature]]
                    * fixed_values[feature]
                )

        first_coefficient = coefficients[
            feature_indices[first_axis]
        ]

        second_coefficient = coefficients[
            feature_indices[second_axis]
        ]

        dependent_values = -(
            fixed_sum
            + first_coefficient * first_grid
            + second_coefficient * second_grid
        ) / dependent_coefficient

        # x, y, z 순서로 표면 좌표 구성
        coordinate_grids = {
            first_axis: first_grid,
            second_axis: second_grid,
            dependent_axis: dependent_values
        }

        surface = go.Surface(
            x=coordinate_grids[x_feature],
            y=coordinate_grids[y_feature],
            z=coordinate_grids[z_feature],
            surfacecolor=np.zeros_like(dependent_values),
            colorscale=[
                [0, color],
                [1, color]
            ],
            opacity=0.45,
            showscale=False,
            name=name,
            hoverinfo="skip"
        )

        return surface, dependent_values

    fig = go.Figure()

    before_plane = make_plane(
        logistic_before,
        "blue",
        "가중치 적용 전 0.5 평면"
    )

    after_plane = make_plane(
        logistic_after,
        "orange",
        "가중치 적용 후 0.5 평면"
    )

    if before_plane is not None:
        fig.add_trace(before_plane[0])

    if after_plane is not None:
        fig.add_trace(after_plane[0])

    # 실제 뇌졸중 여부에 따라 점을 표시
    no_stroke = test_df.loc[
        test_df["stroke"] == 0
    ]

    stroke = test_df.loc[
        test_df["stroke"] == 1
    ]

    fig.add_trace(
        go.Scatter3d(
            x=no_stroke[x_feature],
            y=no_stroke[y_feature],
            z=no_stroke[z_feature],
            mode="markers",
            name="뇌졸중 아님(0)",
            marker=dict(
                color="royalblue",
                size=3,
                opacity=0.6
            )
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=stroke[x_feature],
            y=stroke[y_feature],
            z=stroke[z_feature],
            mode="markers",
            name="뇌졸중 있음(1)",
            marker=dict(
                color="red",
                size=4,
                opacity=0.8
            )
        )
    )

    fig.update_layout(
        title="가중치 적용 전후의 0.5 경계 평면",
        scene=dict(
            xaxis_title=KOREAN_NAMES[x_feature],
            yaxis_title=KOREAN_NAMES[y_feature],
            zaxis_title=KOREAN_NAMES[z_feature]
        ),
        height=750,
        legend_title="실제 뇌졸중 여부"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "마우스로 드래그하면 3차원 그림을 회전할 수 있습니다. "
        "파란색은 가중치 적용 전, 주황색은 적용 후의 0.5 경계 평면입니다."
    )

    if before_plane is None:
        st.warning(
            "가중치 적용 전 모델의 경계면을 현재 축으로 표현하기 어렵습니다."
        )

    if after_plane is None:
        st.warning(
            "가중치 적용 후 모델의 경계면을 현재 축으로 표현하기 어렵습니다."
        )


# --------------------------------------------------
# 21. 안내
# --------------------------------------------------

st.divider()

st.caption(
    "클래스 가중치 적용은 적은 수의 뇌졸중 사례를 학습 과정에서 "
    "더 중요하게 반영하는 방법입니다. 가중치를 적용했다고 해서 "
    "모델의 성능이 모든 지표에서 반드시 높아지는 것은 아닙니다. "
    "이 페이지는 교육용 실습이며 실제 진단에 사용할 수 없습니다."
)
