
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 기본 설정 및 데이터 불러오기
# --------------------------------------------------

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧠",
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
# 2. 변수 이름 설정
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
# 3. 입력 속성 선택
# --------------------------------------------------

st.title("🧠 분류 모델")
st.write(
    "입력 속성을 선택하고 로지스틱 회귀와 "
    "의사결정트리 모델의 결과를 비교해 보세요."
)

selected_features = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요.",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda x: KOREAN_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해야 모델을 만들 수 있습니다.")
    st.stop()


# --------------------------------------------------
# 4. 데이터 정렬 및 훈련용·테스트용 분리
# --------------------------------------------------

df = df.sort_values("id", kind="mergesort").reset_index(drop=True)

# 10명씩 묶었을 때 앞의 3명을 테스트용으로 사용
positions = list(range(len(df)))
test_mask = [position % 10 < 3 for position in positions]

train_df = df.loc[
    [not value for value in test_mask]
].copy()

test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

st.write(
    f"훈련 데이터: **{len(train_df):,}명** / "
    f"테스트 데이터: **{len(test_df):,}명**"
)

if len(test_df) != 1533:
    st.warning(
        f"현재 데이터의 테스트 인원은 {len(test_df):,}명입니다. "
        "데이터 행 수에 따라 1,533명과 다를 수 있습니다."
    )


# --------------------------------------------------
# 5. BMI 결측값 처리
# --------------------------------------------------

# BMI를 선택한 경우에만 훈련 데이터의 중앙값으로 채움
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train.loc[:, "bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test.loc[:, "bmi"] = X_test["bmi"].fillna(bmi_median)

    st.info(
        f"체질량지수의 빈 값은 훈련 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )


# --------------------------------------------------
# 6. 로지스틱 회귀 모델
# --------------------------------------------------

# 크기 맞추기는 훈련 데이터로만 진행
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# 7. 의사결정트리 모델
# --------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)


# --------------------------------------------------
# 8. 기준 모델
# --------------------------------------------------

# 입력 속성을 사용하지 않고 훈련 데이터에서 많은 쪽으로만 예측
majority_class = int(y_train.value_counts().idxmax())

baseline_train_predictions = [
    majority_class
] * len(y_train)

baseline_test_predictions = [
    majority_class
] * len(y_test)


# --------------------------------------------------
# 9. 정확도 계산
# --------------------------------------------------

logistic_train_accuracy = logistic_model.score(
    X_train_scaled,
    y_train
)

logistic_test_accuracy = logistic_model.score(
    X_test_scaled,
    y_test
)

tree_train_accuracy = tree_model.score(
    X_train,
    y_train
)

tree_test_accuracy = tree_model.score(
    X_test,
    y_test
)

baseline_train_accuracy = sum(
    actual == predicted
    for actual, predicted in zip(
        y_train.tolist(),
        baseline_train_predictions
    )
) / len(y_train)

baseline_test_accuracy = sum(
    actual == predicted
    for actual, predicted in zip(
        y_test.tolist(),
        baseline_test_predictions
    )
) / len(y_test)


# --------------------------------------------------
# 10. 정확도 카드
# --------------------------------------------------

st.divider()
st.header("모델 정확도 비교")

col1, col2, col3 = st.columns(3)


def show_accuracy_card(
    column,
    title,
    test_accuracy,
    train_accuracy
):
    with column:
        st.subheader(title)

        st.metric(
            label="테스트 데이터 정확도",
            value=f"{test_accuracy * 100:.2f}%"
        )

        small_col1, small_col2 = st.columns(2)

        small_col1.metric(
            "훈련 데이터",
            f"{train_accuracy * 100:.2f}%"
        )

        small_col2.metric(
            "테스트 데이터",
            f"{test_accuracy * 100:.2f}%"
        )


show_accuracy_card(
    col1,
    "로지스틱 회귀(확률로 답하는 모델)",
    logistic_test_accuracy,
    logistic_train_accuracy
)

show_accuracy_card(
    col2,
    "의사결정트리(질문으로 답하는 모델)",
    tree_test_accuracy,
    tree_train_accuracy
)

show_accuracy_card(
    col3,
    "입력을 하나도 보지 않는 기준 모델",
    baseline_test_accuracy,
    baseline_train_accuracy
)


# --------------------------------------------------
# 11. 가로축과 세로축 선택
# --------------------------------------------------

st.divider()
st.header("두 속성으로 살펴보는 분류 결과")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: KOREAN_NAMES[x]
    )

with axis_col2:
    y_feature_options = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_feature_options,
        format_func=lambda x: KOREAN_NAMES[x]
    )


# --------------------------------------------------
# 12. 두 축 이외의 속성은 테스트 중앙값으로 고정
# --------------------------------------------------

fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = X_test[feature].median()

if fixed_values:
    fixed_text = ", ".join(
        [
            f"{KOREAN_NAMES[feature]}: {value:.2f}"
            for feature, value in fixed_values.items()
        ]
    )

    st.write(
        "**두 축 이외의 속성은 테스트 데이터의 중앙값으로 고정했습니다.**"
    )
    st.write(fixed_text)
else:
    st.write("선택한 속성이 두 개이므로 고정할 나머지 속성이 없습니다.")


# --------------------------------------------------
# 13. 그림을 그리기 위한 범위 설정
# --------------------------------------------------

def make_axis_values(series, count=80):
    minimum = float(series.min())
    maximum = float(series.max())

    if minimum == maximum:
        return [
            minimum - 1,
            minimum,
            minimum + 1
        ]

    margin = (maximum - minimum) * 0.05

    start = minimum - margin
    end = maximum + margin

    return [
        start + (end - start) * i / (count - 1)
        for i in range(count)
    ]


x_values = make_axis_values(X_test[x_feature])
y_values = make_axis_values(X_test[y_feature])

grid_rows = []

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

        grid_rows.append(row)

grid_df = pd.DataFrame(
    grid_rows,
    columns=selected_features
)


# --------------------------------------------------
# 14. 로지스틱 회귀의 0.5 경계선
# --------------------------------------------------

grid_scaled = scaler.transform(grid_df)

logistic_probabilities = logistic_model.predict_proba(
    grid_scaled
)[:, list(logistic_model.classes_).index(1)]

logistic_z = []

for row_index in range(len(y_values)):
    start = row_index * len(x_values)
    end = start + len(x_values)

    logistic_z.append(
        logistic_probabilities[start:end].tolist()
    )


# --------------------------------------------------
# 15. 의사결정트리의 영역 예측
# --------------------------------------------------

tree_grid_predictions = tree_model.predict(grid_df)

tree_z = []

for row_index in range(len(y_values)):
    start = row_index * len(x_values)
    end = start + len(x_values)

    tree_z.append(
        tree_grid_predictions[start:end].tolist()
    )


# --------------------------------------------------
# 16. Plotly 산점도와 분류 영역
# --------------------------------------------------

fig = go.Figure()

# 의사결정트리의 분류 영역
fig.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=tree_z,
        colorscale=[
            [0, "rgba(100, 149, 237, 0.18)"],
            [1, "rgba(255, 120, 120, 0.18)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 분류 영역"
    )
)

# 로지스틱 회귀의 0.5 경계선
fig.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=logistic_z,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="none",
            showlabels=False
        ),
        line=dict(
            color="black",
            width=2
        ),
        showscale=False,
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계선"
    )
)


# 실제 뇌졸중 여부에 따라 점을 나누어 표시
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
            opacity=0.7
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
    title="테스트 데이터와 두 모델의 분류 영역",
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


# --------------------------------------------------
# 17. 로지스틱 회귀 경계선이 그림 안에 있는지 확인
# --------------------------------------------------

has_boundary = (
    min(min(row) for row in logistic_z) <= 0.5
    and max(max(row) for row in logistic_z) >= 0.5
)

if has_boundary:
    st.success(
        "로지스틱 회귀의 0.5 경계선이 그림의 범위 안에 있습니다."
    )
else:
    st.warning(
        "로지스틱 회귀의 0.5 경계선이 현재 그림의 범위 밖에 있습니다."
    )


# --------------------------------------------------
# 18. 의사결정트리 가지 그림
# --------------------------------------------------

st.divider()
st.header("의사결정트리가 던지는 질문")

tree_structure = tree_model.tree_
tree_classes = list(tree_model.classes_)

positive_index = tree_classes.index(1)

# 각 노드에 도달한 훈련 데이터의 실제 인원과 양성 인원 계산
node_total = [0] * tree_structure.node_count
node_positive = [0] * tree_structure.node_count

train_paths = tree_model.decision_path(X_train)

for sample_index in range(len(X_train)):
    reached_nodes = train_paths.indices[
        train_paths.indptr[sample_index]:
        train_paths.indptr[sample_index + 1]
    ]

    actual_label = int(y_train.iloc[sample_index])

    for node_id in reached_nodes:
        node_total[node_id] += 1

        if actual_label == 1:
            node_positive[node_id] += 1


# DOT 문자열 만들기
dot_lines = [
    "digraph Tree {",
    'graph [rankdir=TB, bgcolor="white"];',
    'node [shape=box, style="rounded,filled", '
    'fontname="Malgun Gothic", fontsize=10];',
    'edge [fontname="Malgun Gothic", fontsize=10];'
]

leaf_count = 0
negative_leaf_count = 0
used_features = set()


def escape_dot_text(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def make_node(node_id):
    global leaf_count, negative_leaf_count

    total = node_total[node_id]
    positive = node_positive[node_id]

    positive_ratio = (
        positive / total * 100
        if total > 0
        else 0
    )

    is_leaf = (
        tree_structure.children_left[node_id]
        == tree_structure.children_right[node_id]
    )

    if is_leaf:
        leaf_count += 1

        prediction_index = int(
            tree_structure.value[node_id][0].argmax()
        )

        predicted_class = int(
            tree_classes[prediction_index]
        )

        if predicted_class == 0:
            negative_leaf_count += 1
            fill_color = "#DCEBFA"
            answer_text = "답: 뇌졸중 아님"
        else:
            fill_color = "#FADDDD"
            answer_text = "답: 뇌졸중 있음"

        label = (
            f"마디 {node_id}\\n"
            f"훈련 인원: {total}명\\n"
            f"실제 뇌졸중: {positive}명\\n"
            f"실제 뇌졸중 비율: {positive_ratio:.1f}%\\n"
            f"{answer_text}"
        )

        dot_lines.append(
            f'{node_id} [label="{escape_dot_text(label)}", '
            f'fillcolor="{fill_color}"];'
        )

        return

    feature_index = tree_structure.feature[node_id]
    feature_name = selected_features[feature_index]

    used_features.add(feature_name)

    threshold = tree_structure.threshold[node_id]

    label = (
        f"마디 {node_id}\\n"
        f"{KOREAN_NAMES[feature_name]} "
        f"≤ {threshold:.2f}?\\n"
        f"훈련 인원: {total}명\\n"
        f"실제 뇌졸중: {positive}명\\n"
        f"실제 뇌졸중 비율: {positive_ratio:.1f}%"
    )

    dot_lines.append(
        f'{node_id} [label="{escape_dot_text(label)}", '
        'fillcolor="#FFF2CC"];'
    )

    left_child = tree_structure.children_left[node_id]
    right_child = tree_structure.children_right[node_id]

    make_node(left_child)
    make_node(right_child)

    # 왼쪽 가지는 조건을 만족하는 경우
    dot_lines.append(
        f'{node_id} -> {left_child} [label="예"];'
    )

    # 오른쪽 가지는 조건을 만족하지 않는 경우
    dot_lines.append(
        f'{node_id} -> {right_child} [label="아니요"];'
    )


make_node(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# --------------------------------------------------
# 19. 의사결정트리 요약
# --------------------------------------------------

st.subheader("의사결정트리 요약")

st.write(
    f"- 답을 내리는 마지막 마디: **{leaf_count}개**"
)

st.write(
    f"- 그중 뇌졸중이 아니라고 답하는 마디: "
    f"**{negative_leaf_count}개**"
)

used_feature_names = [
    KOREAN_NAMES[feature]
    for feature in selected_features
    if feature in used_features
]

if used_feature_names:
    st.write(
        "- 실제로 질문에 사용한 속성: **"
        + ", ".join(used_feature_names)
        + "**"
    )
else:
    st.write("- 실제로 질문에 사용한 속성: 없음")


# --------------------------------------------------
# 20. 안내
# --------------------------------------------------

st.divider()

st.caption(
    "이 결과는 학습용 데이터로 분류 모델의 작동 원리를 살펴보는 "
    "교육용 실습입니다. 실제 뇌졸중 진단이나 건강 상태 판단에 "
    "사용할 수 없습니다."
)
