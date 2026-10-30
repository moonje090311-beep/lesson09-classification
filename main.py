import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고 각 변수의 의미를 직접 정리하는 공간입니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/moonje090311-beep/"
    "data/refs/heads/main/stroke.csv"
)
DATA_DIR = Path("내_기록")
RECORD_FILE = DATA_DIR / "stroke_기록.json"

EXPECTED_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]


# ------------------------------------------------------------
# 기록 파일 읽기 / 저장
# ------------------------------------------------------------
def load_record():
    """저장된 우리말 뜻과 데이터 출처를 불러옵니다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            record = json.load(file)
        if not isinstance(record, dict):
            raise ValueError("기록 파일 형식이 올바르지 않습니다.")
        record.setdefault("meanings", {})
        record.setdefault("source", "")
        return record
    except (json.JSONDecodeError, OSError, ValueError):
        st.warning("기록 파일을 읽지 못해 새 기록으로 시작합니다.")
        return {"meanings": {}, "source": ""}


def save_record(meanings, source):
    """입력한 내용을 내_기록 폴더의 JSON 파일에 저장합니다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "meanings": meanings,
        "source": source,
    }
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    return data


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()

record = load_record()

# 필수 열이 있는지 확인
missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
if missing_columns:
    st.error(f"데이터에서 다음 열을 찾을 수 없습니다: {', '.join(missing_columns)}")
    st.stop()

# stroke 값이 1인 사람 수와 비율
stroke_count = int((df["stroke"] == 1).sum())
total_count = len(df)
stroke_ratio = (stroke_count / total_count * 100) if total_count else 0


# ------------------------------------------------------------
# 데이터 소개 및 요약 카드
# ------------------------------------------------------------
st.header("1. 데이터 소개")
st.write(
    "이 데이터는 사람들의 기본 정보와 건강 관련 특성을 열로 정리하고, "
    "뇌졸중 발생 여부를 나타내는 `stroke` 값을 포함한 자료입니다. "
    "각 열이 어떤 정보를 담고 있는지 살펴보세요."
)

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_count:,}명")
card2.metric("열 개수", f"{df.shape[1]:,}개")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
card4.metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")

st.divider()


# ------------------------------------------------------------
# 열 정보 표
# ------------------------------------------------------------
st.header("2. 열 정보 정리")
st.write("‘우리말 뜻’ 칸은 교재를 참고해 직접 채워 보세요.")

def value_kind(column_name, series):
    """열의 역할과 값 형태를 간단히 구분합니다."""
    if column_name == "id":
        return "숫자형(식별자)"
    if column_name == "stroke":
        return "이진형(0 또는 1)"
    if column_name in ["hypertension", "heart_disease"]:
        return "이진형(0 또는 1)"
    if pd.api.types.is_numeric_dtype(series):
        return "숫자형"
    return "범주형"


table_rows = []
for column in df.columns:
    table_rows.append(
        {
            "열 이름": column,
            "우리말 뜻": record["meanings"].get(column, ""),
            "값의 종류": value_kind(column, df[column]),
            "빈 값 개수": int(df[column].isna().sum()),
        }
    )

info_df = pd.DataFrame(table_rows)
st.dataframe(
    info_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻"),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
)

st.caption("참고: 빈 값은 해당 칸에 값이 기록되지 않은 경우를 뜻합니다.")
st.divider()


# ------------------------------------------------------------
# 처음 다섯 줄
# ------------------------------------------------------------
st.header("3. 데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)
st.divider()


# ------------------------------------------------------------
# 교재를 보고 직접 작성하는 기록 영역
# ------------------------------------------------------------
st.header("4. 나의 기록")
st.write(
    "아래 입력란에 교재에서 확인한 각 열의 우리말 뜻과 데이터 출처를 적어 주세요. "
    "저장하면 `내_기록/stroke_기록.json` 파일에 기록됩니다."
)

with st.form("record_form"):
    st.subheader("열 이름의 우리말 뜻")
    meanings_input = {}

    # 2열 배치로 입력 공간을 효율적으로 사용합니다.
    for start in range(0, len(EXPECTED_COLUMNS), 2):
        cols = st.columns(2)
        for offset, column in enumerate(EXPECTED_COLUMNS[start:start + 2]):
            with cols[offset]:
                meanings_input[column] = st.text_input(
                    column,
                    value=record["meanings"].get(column, ""),
                    key=f"meaning_{column}",
                    placeholder="교재를 보고 우리말 뜻 입력",
                )

    st.subheader("데이터 출처")
    source_input = st.text_area(
        "출처 내용",
        value=record.get("source", ""),
        placeholder="교재에 적힌 데이터 출처를 입력하세요.",
        height=100,
    )

    submitted = st.form_submit_button("기록 저장", type="primary")

    if submitted:
        save_record(meanings_input, source_input)
        st.success("기록을 저장했습니다. 다음에 앱을 열어도 입력한 내용이 유지됩니다.")
        st.rerun()

if RECORD_FILE.exists():
    st.caption(f"저장 위치: `{RECORD_FILE}`")

st.divider()
st.caption("데이터 출처")
saved_source = record.get("source", "").strip()
if saved_source:
    st.write(saved_source)
else:
    st.info("아직 출처를 입력하지 않았습니다. 위의 ‘데이터 출처’ 칸에 교재 내용을 적어 주세요.")
