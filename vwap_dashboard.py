# coding=utf-8
"""
================================================================================
VWAP 주문 배분 스케줄 대시보드 (Streamlit)
================================================================================

[시스템 구조]
1. 데이터 레이어
   - load_schedule(): CSV에서 VWAP 스케줄 데이터 로드
   - 주 1회 배치로 업데이트되는 read-only 데이터

2. 비즈니스 로직
   - filter_by_ticker(): 종목코드/종목명으로 필터링
   - redistribute_remaining_weight(): 잔여 주문 재배분 계산

3. 프레젠테이션 레이어
   - render_tables(): Streamlit 테이블 렌더링
   - 메인 UI 컴포넌트

[사용법]
1. 배치 실행: python generate_vwap_schedule_batch.py
2. 대시보드 실행: streamlit run vwap_dashboard.py
3. 브라우저에서 http://localhost:8501 접속

[배포]
- Streamlit Cloud: https://streamlit.io/cloud
- 또는 내부 서버에 Docker로 배포

================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

# =============================================================================
# 한국 시간대 설정 (UTC+9)
# =============================================================================
KST = timezone(timedelta(hours=9))

def get_kst_now() -> datetime:
    """한국 시간(KST) 반환"""
    return datetime.now(KST)

# =============================================================================
# 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="VWAP 주문 배분 스케줄",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 스타일 설정
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2C5282;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F7FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3182CE;
    }
    .warning-box {
        background-color: #FFFAF0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ED8936;
    }
    .success-box {
        background-color: #F0FFF4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #38A169;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 데이터 레이어 함수
# =============================================================================

@st.cache_data(ttl=3600)  # 1시간 캐시
def load_schedule(file_path: str = None) -> Tuple[pd.DataFrame, str]:
    """
    VWAP 스케줄 데이터 로드

    Args:
        file_path: CSV 파일 경로 (None이면 기본 경로 사용)

    Returns:
        (DataFrame, 업데이트 시간 문자열)

    Notes:
        - 주 1회 배치로 업데이트되는 데이터
        - TTL 캐시로 불필요한 재로드 방지
    """
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'db', 'vwap_schedule', 'vwap_schedule.csv')

    # 메타데이터 로드
    meta_path = file_path.replace('vwap_schedule.csv', 'schedule_meta.txt')
    update_time = "알 수 없음"
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '생성 시간' in line:
                    update_time = line.split(': ')[1].strip()
                    break

    # 데이터 로드
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, dtype={'ticker': str})
        return df, update_time
    else:
        # 테스트용 샘플 데이터 생성
        return create_sample_data(), "샘플 데이터"


def create_sample_data() -> pd.DataFrame:
    """
    테스트용 샘플 데이터 생성

    Returns:
        샘플 VWAP 스케줄 DataFrame
    """
    # 삼성전자 샘플
    sample_records = []
    buckets = [
        ('08:30', '09:00', 1.0),
        ('09:00', '09:15', 12.5),
        ('09:15', '09:30', 9.8),
        ('09:30', '10:00', 11.2),
        ('10:00', '10:30', 9.5),
        ('10:30', '11:00', 7.3),
        ('11:00', '11:30', 6.1),
        ('11:30', '12:00', 5.2),
        ('12:00', '12:30', 4.0),
        ('12:30', '13:00', 4.8),
        ('13:00', '13:30', 5.5),
        ('13:30', '14:00', 6.0),
        ('14:00', '14:30', 5.8),
        ('14:30', '15:00', 6.3),
        ('15:00', '15:30', 5.0),
    ]

    # 삼성전자
    cum = 0.0
    for start, end, weight in buckets:
        cum += weight
        sample_records.append({
            'ticker': '005930',
            'name': '삼성전자',
            'start_time': start,
            'end_time': end,
            'weight': weight,
            'cum_weight': round(cum, 2)
        })

    # SK하이닉스 (다른 패턴)
    buckets2 = [
        ('08:30', '09:00', 0.8),
        ('09:00', '09:15', 14.2),
        ('09:15', '09:30', 10.5),
        ('09:30', '10:00', 10.0),
        ('10:00', '10:30', 8.5),
        ('10:30', '11:00', 6.8),
        ('11:00', '11:30', 5.5),
        ('11:30', '12:00', 4.8),
        ('12:00', '12:30', 3.5),
        ('12:30', '13:00', 4.2),
        ('13:00', '13:30', 5.0),
        ('13:30', '14:00', 5.8),
        ('14:00', '14:30', 6.2),
        ('14:30', '15:00', 7.0),
        ('15:00', '15:30', 7.2),
    ]

    cum = 0.0
    for start, end, weight in buckets2:
        cum += weight
        sample_records.append({
            'ticker': '000660',
            'name': 'SK하이닉스',
            'start_time': start,
            'end_time': end,
            'weight': weight,
            'cum_weight': round(cum, 2)
        })

    return pd.DataFrame(sample_records)


# =============================================================================
# 비즈니스 로직 함수
# =============================================================================

def normalize_time_str(time_str: str) -> str:
    """
    시간 문자열을 HH:MM 형식으로 정규화

    Args:
        time_str: 시간 문자열 (예: "9:00", "09:00", "9:0")

    Returns:
        정규화된 시간 문자열 (예: "09:00")
    """
    if not time_str or ':' not in str(time_str):
        return time_str

    parts = str(time_str).strip().split(':')
    if len(parts) >= 2:
        hour = parts[0].zfill(2)
        minute = parts[1].zfill(2)
        return f"{hour}:{minute}"
    return time_str


def filter_by_ticker(df: pd.DataFrame, search_term: str) -> Optional[pd.DataFrame]:
    """
    종목코드 또는 종목명으로 필터링 (단일 종목)

    Args:
        df: 전체 스케줄 DataFrame
        search_term: 검색어 (종목코드 또는 종목명)

    Returns:
        필터링된 DataFrame 또는 None (검색 결과 없음)
    """
    if not search_term:
        return None

    search_term = search_term.strip()

    # 종목코드로 검색 (정확히 일치)
    mask_code = df['ticker'] == search_term

    # 종목명으로 검색 (부분 일치)
    mask_name = df['name'].str.contains(search_term, case=False, na=False)

    result = df[mask_code | mask_name]

    if result.empty:
        return None

    first_ticker = result['ticker'].iloc[0]
    return df[df['ticker'] == first_ticker].copy()


def filter_by_multiple_tickers(df: pd.DataFrame, search_terms: str) -> List[Tuple[str, str, pd.DataFrame]]:
    """
    여러 종목코드/종목명으로 필터링 (최대 4종목)

    Args:
        df: 전체 스케줄 DataFrame
        search_terms: 검색어들 (쉼표, 공백으로 구분)

    Returns:
        [(ticker, name, DataFrame), ...] 리스트 (최대 4개)
    """
    if not search_terms:
        return []

    # 구분자로 분리 (쉼표, 공백, 줄바꿈)
    terms = re.split(r'[,\s\n]+', search_terms.strip())
    terms = [t.strip() for t in terms if t.strip()]

    results = []
    found_tickers = set()

    for term in terms:
        if len(results) >= 4:
            break

        # 종목코드로 검색
        mask_code = df['ticker'] == term
        # 종목명으로 검색
        mask_name = df['name'].str.contains(term, case=False, na=False)

        matched = df[mask_code | mask_name]

        if not matched.empty:
            ticker = matched['ticker'].iloc[0]
            if ticker not in found_tickers:
                found_tickers.add(ticker)
                name = matched['name'].iloc[0]
                stock_df = df[df['ticker'] == ticker].copy()
                results.append((ticker, name, stock_df))

    return results


def get_current_bucket(current_time: str = None) -> Tuple[str, str]:
    """
    현재 시간의 bucket 반환

    Args:
        current_time: 'HH:MM' 형식 (None이면 현재 시간)

    Returns:
        (start_time, end_time) tuple
    """
    if current_time is None:
        now = get_kst_now()
        current_time = now.strftime('%H:%M')
    else:
        current_time = normalize_time_str(current_time)

    buckets = [
        ('08:30', '09:00'),
        ('09:00', '09:15'),
        ('09:15', '09:30'),
        ('09:30', '10:00'),
        ('10:00', '10:30'),
        ('10:30', '11:00'),
        ('11:00', '11:30'),
        ('11:30', '12:00'),
        ('12:00', '12:30'),
        ('12:30', '13:00'),
        ('13:00', '13:30'),
        ('13:30', '14:00'),
        ('14:00', '14:30'),
        ('14:30', '15:00'),
        ('15:00', '15:30'),
    ]

    for start, end in buckets:
        if start <= current_time < end:
            return (start, end)

    # 장 시간 외
    if current_time < '08:30':
        return ('08:30', '09:00')
    else:
        return ('15:00', '15:30')


def apply_twap_cap_with_carry(
    weights: pd.Series,
    cap: float = 20.0
) -> Tuple[pd.Series, Optional[str]]:
    """
    TWAP 상한(cap) 적용 - 초과분을 이후 구간으로 이월

    Args:
        weights: 각 bucket의 비중 (% 단위)
        cap: 최대 비중 제한 (기본값 20%)

    Returns:
        (조정된 weights Series, 경고 메시지 또는 None)
    """
    if weights.empty:
        return weights, None

    result = weights.copy().astype(float)
    n = len(result)
    warning_msg = None
    total_original = result.sum()

    max_iterations = n * 2

    for iteration in range(max_iterations):
        excess_total = 0.0
        has_excess = False

        for i in range(n):
            if result.iloc[i] > cap:
                excess = result.iloc[i] - cap
                result.iloc[i] = cap
                excess_total += excess
                has_excess = True

        if not has_excess or excess_total < 0.001:
            break

        remaining_indices = [i for i in range(n) if result.iloc[i] < cap]

        if not remaining_indices:
            current_sum = result.sum()
            shortfall = total_original - current_sum
            if shortfall > 0.01:
                warning_msg = (
                    f"[WARNING] {cap}% cap으로는 남은 비중을 모두 배정할 수 없습니다. "
                    f"미배정 비중: {shortfall:.1f}%"
                )
            break

        remaining_weights = [result.iloc[i] for i in remaining_indices]
        remaining_sum = sum(remaining_weights)

        if remaining_sum <= 0:
            per_bucket = excess_total / len(remaining_indices)
            for i in remaining_indices:
                result.iloc[i] += per_bucket
        else:
            for i in remaining_indices:
                ratio = result.iloc[i] / remaining_sum
                result.iloc[i] += excess_total * ratio

    final_sum = result.sum()
    if abs(final_sum - total_original) > 0.01 and warning_msg is None:
        diff = total_original - final_sum
        result.iloc[-1] += diff

    return result, warning_msg


def redistribute_remaining_weight(
    df_schedule: pd.DataFrame,
    actual_filled_pct: float,
    current_time: str = None,
    apply_cap: bool = True,
    cap_pct: float = 20.0
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    잔여 주문 재배분 계산 (TWAP cap 적용)

    Args:
        df_schedule: 종목의 VWAP 스케줄 DataFrame
        actual_filled_pct: 실제 체결률 (%, 0-100)
        current_time: 현재 시간 'HH:MM' (None이면 시스템 시간)
        apply_cap: TWAP cap 적용 여부 (기본값 True)
        cap_pct: 최대 비중 제한 (기본값 20%)

    Returns:
        (재배분된 스케줄 DataFrame, 경고 메시지 또는 None)
    """
    df = df_schedule.copy()
    warning_msg = None

    # 시간 문자열 정규화
    df['start_time'] = df['start_time'].apply(normalize_time_str)
    df['end_time'] = df['end_time'].apply(normalize_time_str)

    # 현재 bucket 확인
    current_bucket = get_current_bucket(current_time)
    current_start = current_bucket[0]

    # 남은 구간 필터링 (현재 시간대 포함)
    df_remaining = df[df['start_time'] >= current_start].copy()

    if df_remaining.empty:
        return pd.DataFrame(), None

    # 원래 weight 합계
    original_remaining_sum = df_remaining['weight'].sum()

    # 남은 비중
    remaining_to_fill = 100.0 - actual_filled_pct

    if remaining_to_fill <= 0:
        df_remaining['new_weight'] = 0.0
        df_remaining['new_cum_weight'] = 100.0
    elif original_remaining_sum <= 0:
        equal_weight = remaining_to_fill / len(df_remaining)
        df_remaining['new_weight'] = equal_weight
    else:
        df_remaining['new_weight'] = (
            df_remaining['weight'] / original_remaining_sum * remaining_to_fill
        )

    # TWAP cap 적용
    if apply_cap and remaining_to_fill > 0:
        weights_series = df_remaining['new_weight'].reset_index(drop=True)
        capped_weights, warning_msg = apply_twap_cap_with_carry(weights_series, cap=cap_pct)
        df_remaining['new_weight'] = capped_weights.values

    # 새 누적 체결률 계산
    df_remaining['new_cum_weight'] = actual_filled_pct + df_remaining['new_weight'].cumsum()

    # 결과 정리
    df_result = df_remaining[[
        'start_time', 'end_time', 'weight', 'new_weight', 'new_cum_weight'
    ]].copy()

    df_result.columns = ['시작시간', '종료시간', '기존비율', '재배분비율', '재배분누적']

    # 반올림
    df_result['기존비율'] = df_result['기존비율'].round(1)
    df_result['재배분비율'] = df_result['재배분비율'].round(1)
    df_result['재배분누적'] = df_result['재배분누적'].round(1)

    return df_result, warning_msg


# =============================================================================
# 프레젠테이션 레이어 함수
# =============================================================================

def render_schedule_table(df: pd.DataFrame, title: str = "VWAP 스케줄"):
    """
    스케줄 테이블 렌더링

    Args:
        df: 스케줄 DataFrame
        title: 테이블 제목
    """
    st.subheader(title)

    # 표시용 DataFrame
    display_df = df[['start_time', 'end_time', 'weight', 'cum_weight']].copy()
    display_df.columns = ['시작시간', '종료시간', '비율(%)', '누적체결률(%)']

    # 시간 정규화
    display_df['시작시간'] = display_df['시작시간'].apply(normalize_time_str)
    display_df['종료시간'] = display_df['종료시간'].apply(normalize_time_str)

    # 비율 포맷팅
    display_df['비율(%)'] = display_df['비율(%)'].round(0).astype(int)
    display_df['누적체결률(%)'] = display_df['누적체결률(%)'].round(0).astype(int)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(len(display_df) * 40 + 40, 600)
    )


def render_redistribution_table(df: pd.DataFrame):
    """
    재배분 결과 테이블 렌더링

    Args:
        df: 재배분된 스케줄 DataFrame
    """
    if df.empty:
        st.warning("남은 구간이 없습니다.")
        return

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(len(df) * 40 + 40, 400)
    )


def render_copy_format(df: pd.DataFrame, ticker: str, name: str):
    """
    복사용 텍스트 포맷 렌더링

    Args:
        df: 재배분된 스케줄 DataFrame
        ticker: 종목코드
        name: 종목명
    """
    if df.empty:
        return

    st.markdown("##### 📋 복사용 텍스트")

    lines = [f"[{ticker}] {name} - 재배분 스케줄", ""]
    lines.append("시간대\t\t비율(%)\t누적(%)")
    lines.append("-" * 40)

    for _, row in df.iterrows():
        time_range = f"{row['시작시간']}-{row['종료시간']}"
        lines.append(f"{time_range}\t{row['재배분비율']:.1f}\t{row['재배분누적']:.1f}")

    text = "\n".join(lines)
    st.code(text, language=None)


# =============================================================================
# 메인 UI
# =============================================================================

def main():
    """메인 애플리케이션"""

    # ---------------------------------------------------------------------
    # 헤더
    # ---------------------------------------------------------------------
    st.markdown('<p class="main-header">📊 VWAP 주문 배분 스케줄</p>', unsafe_allow_html=True)

    # 데이터 로드
    df_schedule, update_time = load_schedule()

    # 사이드바: 메타 정보
    with st.sidebar:
        st.markdown("### ℹ️ 시스템 정보")
        st.info(f"**스케줄 업데이트:** {update_time}")

        if not df_schedule.empty:
            n_stocks = df_schedule['ticker'].nunique()
            st.metric("등록 종목 수", f"{n_stocks:,}개")

        st.markdown("---")
        st.markdown("### 📖 사용법")
        st.markdown("""
        1. 종목코드 또는 종목명 입력
        2. 원본 스케줄 확인
        3. 실제 체결률 입력
        4. 재배분 결과 확인 및 복사
        """)

        st.markdown("---")
        st.markdown("### ⏰ 현재 시간")
        now = get_kst_now()
        st.write(f"**{now.strftime('%H:%M:%S')}**")
        current_bucket = get_current_bucket()
        st.write(f"현재 구간: {current_bucket[0]} - {current_bucket[1]}")

    # ---------------------------------------------------------------------
    # 기능 1: 종목 검색 (최대 4종목 동시)
    # ---------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<p class="sub-header">🔍 종목 검색 (최대 4종목)</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        search_term = st.text_input(
            "종목코드 또는 종목명 입력 (쉼표/공백으로 구분)",
            placeholder="예: 005930, 000660 또는 삼성전자 SK하이닉스",
            help="최대 4개 종목을 쉼표 또는 공백으로 구분하여 입력하세요"
        )

    with col2:
        st.write("")
        search_clicked = st.button("🔍 검색", use_container_width=True)

    # 검색 실행
    if search_term or search_clicked:
        search_results = filter_by_multiple_tickers(df_schedule, search_term)

        if search_results:
            st.success(f"**{len(search_results)}개 종목** 스케줄을 불러왔습니다: " +
                      ", ".join([f"[{t}] {n}" for t, n, _ in search_results]))

            # -----------------------------------------------------------------
            # 공통 설정: 체결률, 시간, TWAP cap
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown('<p class="sub-header">✏️ 공통 설정</p>', unsafe_allow_html=True)

            col_input1, col_input2, col_input3, col_input4 = st.columns([2, 2, 2, 2])

            with col_input1:
                actual_filled = st.number_input(
                    "현재까지 실제 체결률 (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=1.0,
                    help="현재까지 체결된 비율을 입력하세요 (0-100%)"
                )

            with col_input2:
                use_current_time = st.checkbox("현재 시간 사용", value=True)
                if use_current_time:
                    current_time = None
                    time_display = get_kst_now().strftime('%H:%M')
                else:
                    time_input = st.time_input("시간 지정", value=get_kst_now().time())
                    current_time = time_input.strftime('%H:%M')
                    time_display = current_time

            with col_input3:
                apply_twap_cap = st.checkbox("TWAP 제한 적용", value=True,
                                             help="각 구간 최대 비중 제한")
                if apply_twap_cap:
                    cap_value = st.slider("최대 비중 (%)", 10, 50, 20, 5)
                else:
                    cap_value = 100.0

            with col_input4:
                st.write("")
                calculate_clicked = st.button("📊 재배분 계산", use_container_width=True, type="primary")

            # 요약 메트릭
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("실제 체결률", f"{actual_filled:.1f}%")
            with col_m2:
                remaining = 100.0 - actual_filled
                st.metric("잔여 비중", f"{remaining:.1f}%")
            with col_m3:
                st.metric("기준 시간", time_display)
            with col_m4:
                current_bucket = get_current_bucket(current_time)
                st.metric("현재 구간", f"{current_bucket[0]}-{current_bucket[1]}")

            # -----------------------------------------------------------------
            # 종목별 결과 표시 (2x2 그리드)
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown('<p class="sub-header">📊 종목별 VWAP 스케줄</p>', unsafe_allow_html=True)

            # 2x2 그리드 레이아웃
            n_stocks = len(search_results)

            if n_stocks == 1:
                cols = [st.columns(1)[0]]
            elif n_stocks == 2:
                cols = st.columns(2)
            else:
                row1 = st.columns(2)
                row2 = st.columns(2)
                cols = row1 + row2

            for idx, (ticker, name, df_filtered) in enumerate(search_results):
                with cols[idx]:
                    st.markdown(f"#### [{ticker}] {name}")

                    # 원본 스케줄 (접기)
                    with st.expander("원본 스케줄", expanded=False):
                        display_df = df_filtered[['start_time', 'end_time', 'weight', 'cum_weight']].copy()
                        display_df.columns = ['시작', '종료', '비율(%)', '누적(%)']
                        st.dataframe(display_df, use_container_width=True, hide_index=True,
                                    height=min(len(display_df) * 35 + 35, 300))

                    # 재배분 계산
                    if calculate_clicked or actual_filled > 0:
                        df_redistribution, twap_warning = redistribute_remaining_weight(
                            df_filtered,
                            actual_filled,
                            current_time,
                            apply_cap=apply_twap_cap,
                            cap_pct=cap_value
                        )

                        if twap_warning:
                            st.warning(twap_warning, icon="⚠️")

                        if not df_redistribution.empty:
                            st.markdown("**재배분 결과:**")
                            st.dataframe(
                                df_redistribution,
                                use_container_width=True,
                                hide_index=True,
                                height=min(len(df_redistribution) * 35 + 35, 300)
                            )

                            # 검증
                            total_new = df_redistribution['재배분비율'].sum()
                            expected = 100.0 - actual_filled

                            if abs(total_new - expected) < 1:
                                st.success(f"합계: {total_new:.1f}%")
                            else:
                                st.warning(f"합계: {total_new:.1f}% (예상: {expected:.1f}%)")

                            # 복사용 텍스트
                            with st.expander("복사용 텍스트"):
                                lines = [f"[{ticker}] {name}", "시간\t비율\t누적"]
                                for _, row in df_redistribution.iterrows():
                                    lines.append(f"{row['시작시간']}-{row['종료시간']}\t{row['재배분비율']}\t{row['재배분누적']}")
                                st.code("\n".join(lines), language=None)
                        else:
                            st.info("남은 구간 없음")

        else:
            st.warning(f"'{search_term}'에 해당하는 종목을 찾을 수 없습니다.")

    # ---------------------------------------------------------------------
    # 전체 종목 목록 (접기)
    # ---------------------------------------------------------------------
    st.markdown("---")
    with st.expander("📋 전체 종목 목록 보기"):
        if not df_schedule.empty:
            stock_list = df_schedule[['ticker', 'name']].drop_duplicates()
            stock_list = stock_list.sort_values('ticker')
            st.dataframe(stock_list, use_container_width=True, hide_index=True)
        else:
            st.info("등록된 종목이 없습니다.")


# =============================================================================
# 실행
# =============================================================================

if __name__ == '__main__':
    main()
