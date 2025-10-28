# -*- coding: utf-8 -*-
"""
개선된 스키마 기반 건강 검진 데이터베이스 생성 스크립트

=== 데이터베이스 스키마 ===

1. patients (환자 기본정보)
   - patient_id: 환자 고유 번호 (자동증가, 기본키)
   - name: 환자 성명
   - sex: 성별 ('남', '여')
   - age: 나이
   - rrn_masked: 주민등록번호 마스킹 (예: 990101-3******)
   - registered_at: 환자 등록 일시

2. health_exams (검진 정보 + 측정 데이터 통합)
   - exam_id: 검진 고유 번호 (자동증가, 기본키)
   - patient_id: 환자 고유 번호 (외래키)
   - exam_at: 검진 일시
   - facility_name: 검진 기관명
   - doc_registered_on: 검진 결과 등록 날짜
   - height_cm, weight_kg, bmi: 신체 측정
   - waist_cm: 허리둘레
   - systolic_mmHg, diastolic_mmHg: 혈압
   - fbg_mg_dl: 공복혈당
   - tg_mg_dl: 중성지방
   - hdl_mg_dl: HDL 콜레스테롤
   - tc_mg_dl: 총콜레스테롤
   - ldl_mg_dl: LDL 콜레스테롤

※ 대사증후군 진단기준 (한국인 기준):
  - 복부비만: 허리둘레 남성 ≥90cm, 여성 ≥85cm
  - 고혈압: 수축기 ≥130mmHg 또는 이완기 ≥85mmHg
  - 공복혈당장애: 공복혈당 ≥100mg/dL
  - 고중성지방: 중성지방 ≥150mg/dL
  - 저HDL콜레스테롤: HDL 남성 <40mg/dL, 여성 <50mg/dL
  (5개 항목 중 3개 이상 해당 시 대사증후군 진단)
"""

import sqlite3
import json
from pathlib import Path

DB_PATH = Path("metabolic_health.sqlite")
CASES_JSON = Path("health_cases.json")

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- 환자 기본 정보
CREATE TABLE IF NOT EXISTS patients (
  patient_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  name           TEXT NOT NULL,
  sex            TEXT CHECK (sex IN ('남','여')),
  age            INTEGER,
  rrn_masked     TEXT,
  registered_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 검진 정보 + 측정 데이터 통합
CREATE TABLE IF NOT EXISTS health_exams (
  exam_id            INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id         INTEGER NOT NULL,
  exam_at            TIMESTAMP NOT NULL,
  
  -- 검진 메타데이터
  facility_name      TEXT,
  doc_registered_on  DATE,
  
  -- 신체 측정
  height_cm          REAL,
  weight_kg          REAL,
  bmi                REAL,
  
  -- 대사증후군 진단 기준 5가지
  waist_cm           REAL,
  systolic_mmHg      INTEGER,
  diastolic_mmHg     INTEGER,
  fbg_mg_dl          REAL,
  tg_mg_dl           REAL,
  hdl_mg_dl          REAL,
  
  -- 추가 지질 검사
  tc_mg_dl           REAL,
  ldl_mg_dl          REAL,
  
  FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

-- 인덱스: 환자별 검진 이력 조회 최적화
CREATE INDEX IF NOT EXISTS idx_health_exams_patient 
ON health_exams(patient_id, exam_at DESC);
"""


def calculate_bmi(height_cm, weight_kg):
    """BMI 계산 (키는 cm, 몸무게는 kg)"""
    height_m = height_cm / 100
    return round(weight_kg / (height_m**2), 2)


def load_cases():
    """JSON 파일에서 케이스 로드"""
    if not CASES_JSON.exists():
        raise FileNotFoundError(
            f"{CASES_JSON} 파일을 찾을 수 없습니다. health_cases.json 파일이 필요합니다."
        )

    try:
        with open(CASES_JSON, "r", encoding="utf-8") as f:
            cases = json.load(f)
        print(f"✅ {CASES_JSON}에서 {len(cases)}개의 케이스를 로드했습니다.")
        return cases
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파일 파싱 오류: {e}")
        raise
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        raise


def main():
    """메인 실행 함수"""
    # 기존 DB 삭제
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"🗑️  기존 데이터베이스 삭제: {DB_PATH}")

    # DB 연결 및 스키마 생성
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    print(f"✅ 데이터베이스 생성: {DB_PATH}")

    # JSON 케이스 로드
    cases = load_cases()

    print(f"\n📊 데이터 삽입 중...")

    for i, case in enumerate(cases, 1):
        # BMI 자동 계산
        calculated_bmi = calculate_bmi(case["height"], case["weight"])

        # 1. 환자 정보 삽입
        cur.execute(
            """INSERT INTO patients (name, sex, age, rrn_masked, registered_at) 
               VALUES (?, ?, ?, ?, ?)""",
            (case["name"], case["sex"], case["age"], case["rrn"], case["reg"]),
        )
        patient_id = cur.lastrowid

        # 2. 검진 정보 + 측정 데이터 통합 삽입
        cur.execute(
            """INSERT INTO health_exams (
               patient_id, exam_at, facility_name, doc_registered_on,
               height_cm, weight_kg, bmi, waist_cm,
               systolic_mmHg, diastolic_mmHg, fbg_mg_dl,
               tg_mg_dl, hdl_mg_dl, tc_mg_dl, ldl_mg_dl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                patient_id,
                case["exam_at"],
                case["facility"],
                case["doc_reg"],
                case["height"],
                case["weight"],
                calculated_bmi,
                case["waist"],
                case["sys"],
                case["dia"],
                case["fbg"],
                case["tg"],
                case["hdl"],
                case["tc"],
                case["ldl"],
            ),
        )

        if i % 5 == 0:
            print(f"  ✓ {i}/{len(cases)} 환자 데이터 삽입 완료")

    conn.commit()
    conn.close()

    print(f"\n✅ 데이터베이스 생성 완료!")
    print(f"\n📈 통계 정보:")
    print(f"  - 총 환자 수: {len(cases)}명")

    # 연령대별 통계
    age_groups = {}
    for case in cases:
        age_group = f"{case['age']//10*10}대"
        age_groups[age_group] = age_groups.get(age_group, 0) + 1

    for age_group in sorted(age_groups.keys()):
        print(f"  - {age_group}: {age_groups[age_group]}명")

    # 성별 통계
    sex_count = {"남": 0, "여": 0}
    for case in cases:
        sex_count[case["sex"]] += 1
    print(f"  - 남성: {sex_count['남']}명, 여성: {sex_count['여']}명")


if __name__ == "__main__":
    main()
