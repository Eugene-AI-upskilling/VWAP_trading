# VWAP 주문 배분 스케줄 대시보드

종목별 VWAP 기반 주문 배분 스케줄 조회 및 잔여 주문 재배분 계산 도구

## 기능

1. **종목 검색**: 종목코드 또는 종목명으로 VWAP 스케줄 조회
2. **스케줄 조회**: 시간대별 주문 배분 비율 확인
3. **재배분 계산**: 실제 체결률 입력 시 남은 구간 자동 재배분

## 사용법

1. 종목코드(예: 005930) 또는 종목명(예: 삼성전자) 입력
2. 원본 VWAP 스케줄 확인
3. 현재까지 실제 체결률(%) 입력
4. 재배분된 스케줄 확인 및 복사

## 데이터 업데이트

- 스케줄 데이터는 주 1회 배치로 업데이트
- 과거 60거래일 5분봉 거래량 기반

---

## 종목 리스트 변경 방법

### 1. 소수 종목 추가/삭제 (수동)

`db/vwap_schedule/vwap_schedule.csv` 파일을 직접 편집합니다.

**CSV 형식:**
```csv
ticker,name,start_time,end_time,weight,cum_weight
005930,삼성전자,08:30,09:00,0.0,0.0
005930,삼성전자,09:00,09:15,13.66,13.66
...
```

**컬럼 설명:**
| 컬럼 | 설명 |
|------|------|
| ticker | 종목코드 (6자리) |
| name | 종목명 |
| start_time | 시간대 시작 |
| end_time | 시간대 종료 |
| weight | 해당 구간 배분 비율 (%) |
| cum_weight | 누적 체결률 (%) |

**시간대 구간 (15개):**
- 08:30-09:00 (동시호가)
- 09:00-09:15, 09:15-09:30, 09:30-10:00
- 10:00-10:30, 10:30-11:00, 11:00-11:30
- 11:30-12:00, 12:00-12:30, 12:30-13:00
- 13:00-13:30, 13:30-14:00, 14:00-14:30
- 14:30-15:00, 15:00-15:30

### 2. 전체 종목 갱신 (배치)

로컬 PC에서 Creon API를 사용하여 배치 실행:

```bash
# 1. 종목 리스트 수정
#    Creon-Datareader/db/code_list.csv 편집

# 2. 5분봉 데이터 수집 (관리자 권한, Creon 로그인 필요)
run_collect_5min.bat

# 3. VWAP 스케줄 생성
run_schedule_batch.bat

# 4. 생성된 파일을 이 리포지토리에 복사
#    db/vwap_schedule/vwap_schedule.csv

# 5. GitHub에 push하면 자동 배포
git add .
git commit -m "Update schedule data"
git push
```

### 3. 변경 후 배포

GitHub에 push하면 Streamlit Cloud가 **자동으로 재배포**됩니다. (1-2분 소요)

---

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run vwap_dashboard.py
```

## 배포 URL

https://vwaptrading-gplfbis7ugttk8dmeia6da.streamlit.app/
