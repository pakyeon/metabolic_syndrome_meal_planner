"""SQLite-backed patient repository and metabolic syndrome evaluation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class PatientDatabase:
    """데이터베이스 조회 및 대사증후군 평가를 담당한다."""

    CRITERIA = {
        "waist": {"male": 90, "female": 85},
        "blood_pressure": {"systolic": 130, "diastolic": 85},
        "fasting_glucose": 100,
        "triglycerides": 150,
        "hdl": {"male": 40, "female": 50},
    }

    def __init__(self, db_path: str | Path = "metabolic_health.sqlite"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"데이터베이스 파일을 찾을 수 없습니다: {self.db_path}\n"
                "build_health_scenarios_v2.py를 먼저 실행하세요."
            )

    # ------------------------------------------------------------------ #
    # 기본 조회
    # ------------------------------------------------------------------ #

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_patient(self, patient_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT patient_id, name, sex, age, rrn_masked, registered_at
                FROM patients
                WHERE patient_id = ?
                """,
                (patient_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_patients(self) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT patient_id, name, sex, age, rrn_masked, registered_at
                FROM patients
                ORDER BY patient_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_latest_exam(self, patient_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM health_exams
                WHERE patient_id = ?
                ORDER BY exam_at DESC
                LIMIT 1
                """,
                (patient_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_exam_history(self, patient_id: int) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM health_exams
                WHERE patient_id = ?
                ORDER BY exam_at DESC
                """,
                (patient_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ #
    # 평가 로직
    # ------------------------------------------------------------------ #

    def check_metabolic_syndrome(self, patient_id: int) -> Optional[Dict]:
        patient = self.get_patient(patient_id)
        if not patient:
            return None

        exam = self.get_latest_exam(patient_id)
        if not exam:
            return None

        sex = patient["sex"]
        risk_factors = {
            "abdominal_obesity": self._check_abdominal_obesity(exam["waist_cm"], sex),
            "high_blood_pressure": self._check_blood_pressure(
                exam["systolic_mmHg"], exam["diastolic_mmHg"]
            ),
            "high_fasting_glucose": self._check_fasting_glucose(exam["fbg_mg_dl"]),
            "high_triglycerides": self._check_triglycerides(exam["tg_mg_dl"]),
            "low_hdl": self._check_hdl(exam["hdl_mg_dl"], sex),
        }
        criteria_met = sum(risk_factors.values())
        has_metabolic_syndrome = criteria_met >= 3

        return {
            "patient_id": patient_id,
            "name": patient["name"],
            "sex": sex,
            "age": patient["age"],
            "exam_at": exam["exam_at"],
            "criteria_met": criteria_met,
            "has_metabolic_syndrome": has_metabolic_syndrome,
            "risk_factors": risk_factors,
            "measurements": {
                "waist_cm": exam["waist_cm"],
                "systolic_mmHg": exam["systolic_mmHg"],
                "diastolic_mmHg": exam["diastolic_mmHg"],
                "fbg_mg_dl": exam["fbg_mg_dl"],
                "tg_mg_dl": exam["tg_mg_dl"],
                "hdl_mg_dl": exam["hdl_mg_dl"],
                "bmi": exam["bmi"],
            },
        }

    def get_patients_with_metabolic_syndrome(self) -> List[Dict]:
        results: List[Dict] = []
        for patient in self.get_all_patients():
            diagnosis = self.check_metabolic_syndrome(patient["patient_id"])
            if diagnosis and diagnosis["has_metabolic_syndrome"]:
                results.append(diagnosis)
        return results

    def get_statistics(self) -> Dict:
        all_patients = self.get_all_patients()
        ms_patients = self.get_patients_with_metabolic_syndrome()
        male = sum(1 for p in all_patients if p["sex"] == "남")
        female = sum(1 for p in all_patients if p["sex"] == "여")

        age_groups: Dict[str, int] = {}
        for p in all_patients:
            bucket = f"{p['age']//10*10}대"
            age_groups[bucket] = age_groups.get(bucket, 0) + 1

        return {
            "total_patients": len(all_patients),
            "male_patients": male,
            "female_patients": female,
            "metabolic_syndrome_patients": len(ms_patients),
            "metabolic_syndrome_rate": (
                len(ms_patients) / len(all_patients) * 100 if all_patients else 0
            ),
            "age_distribution": age_groups,
        }

    # ------------------------------------------------------------------ #
    # 위험도/보고서
    # ------------------------------------------------------------------ #

    def evaluate_risk_level(self, patient_id: int) -> Optional[Dict]:
        diagnosis = self.check_metabolic_syndrome(patient_id)
        if not diagnosis:
            return None

        score = diagnosis["criteria_met"]
        if score == 0:
            level, label = "low", "저위험"
            desc = "현재 대사증후군 위험 요인이 없습니다. 건강한 생활습관을 유지하세요."
        elif score == 1:
            level, label = "low", "저위험"
            desc = "1개의 위험 요인이 있습니다. 예방적 관리가 필요합니다."
        elif score == 2:
            level, label = "medium", "중위험"
            desc = "2개의 위험 요인이 있어 대사증후군 전단계입니다. 적극적인 생활습관 개선이 필요합니다."
        elif score == 3:
            level, label = "high", "고위험"
            desc = "대사증후군으로 진단됩니다. 의료진 상담 및 치료적 개입이 필요합니다."
        else:
            level, label = "high", "고위험"
            desc = (
                f"{score}개의 위험 요인이 있어 심혈관질환 위험이 매우 높습니다. "
                "즉각적인 의학적 관리가 필요합니다."
            )

        return {
            "risk_level": level,
            "risk_score": score,
            "risk_label": label,
            "risk_description": desc,
        }

    def interpret_risk_factors(self, patient_id: int) -> Optional[Dict]:
        diagnosis = self.check_metabolic_syndrome(patient_id)
        if not diagnosis:
            return None

        patient = self.get_patient(patient_id)
        if not patient:
            return None

        sex = patient["sex"]
        measurements = diagnosis["measurements"]
        factors = diagnosis["risk_factors"]

        interpretations: Dict[str, Dict[str, str]] = {}

        waist = measurements["waist_cm"]
        waist_threshold = (
            self.CRITERIA["waist"]["male"]
            if sex == "남"
            else self.CRITERIA["waist"]["female"]
        )
        interpretations["abdominal_obesity"] = (
            {
                "status": "위험",
                "value": f"{waist:.1f}cm",
                "threshold": f"{waist_threshold}cm",
                "interpretation": f"허리둘레가 {waist:.1f}cm로 기준({waist_threshold}cm)을 초과합니다.",
                "recommendation": "내장지방 감소를 위해 유산소 운동과 식이조절이 필요합니다.",
            }
            if factors["abdominal_obesity"]
            else {
                "status": "정상",
                "value": f"{waist:.1f}cm",
                "threshold": f"{waist_threshold}cm",
                "interpretation": f"허리둘레가 {waist:.1f}cm로 정상 범위입니다.",
                "recommendation": "현재 상태를 유지하세요.",
            }
        )

        sys = measurements["systolic_mmHg"]
        dia = measurements["diastolic_mmHg"]
        interpretations["high_blood_pressure"] = (
            {
                "status": "위험",
                "value": f"{sys}/{dia}mmHg",
                "threshold": "130/85mmHg",
                "interpretation": f"혈압이 {sys}/{dia}mmHg로 고혈압 기준을 충족합니다.",
                "recommendation": "저염식이, 규칙적인 운동, 필요시 약물 치료가 필요합니다.",
            }
            if factors["high_blood_pressure"]
            else {
                "status": "정상",
                "value": f"{sys}/{dia}mmHg",
                "threshold": "130/85mmHg",
                "interpretation": f"혈압이 {sys}/{dia}mmHg로 정상 범위입니다.",
                "recommendation": "저염식이와 규칙적인 운동으로 혈압을 유지하세요.",
            }
        )

        fbg = measurements["fbg_mg_dl"]
        if factors["high_fasting_glucose"]:
            if fbg >= 126:
                interpretation = (
                    f"공복혈당이 {fbg:.1f}mg/dL로 당뇨병 진단 기준(126mg/dL)에 해당합니다."
                )
                recommendation = "즉시 의료진 상담이 필요하며 혈당 관리를 시작해야 합니다."
            else:
                interpretation = (
                    f"공복혈당이 {fbg:.1f}mg/dL로 공복혈당장애(전단계) 상태입니다."
                )
                recommendation = "탄수화물 조절과 체중 관리가 필요합니다."
            interpretations["high_fasting_glucose"] = {
                "status": "위험",
                "value": f"{fbg:.1f}mg/dL",
                "threshold": "100mg/dL",
                "interpretation": interpretation,
                "recommendation": recommendation,
            }
        else:
            interpretations["high_fasting_glucose"] = {
                "status": "정상",
                "value": f"{fbg:.1f}mg/dL",
                "threshold": "100mg/dL",
                "interpretation": f"공복혈당이 {fbg:.1f}mg/dL로 정상 범위입니다.",
                "recommendation": "혈당이 상승하지 않도록 식단을 관리하세요.",
            }

        tg = measurements["tg_mg_dl"]
        interpretations["high_triglycerides"] = (
            {
                "status": "위험",
                "value": f"{tg:.1f}mg/dL",
                "threshold": "150mg/dL",
                "interpretation": f"중성지방이 {tg:.1f}mg/dL로 높습니다.",
                "recommendation": "포화지방을 제한하고 규칙적인 운동을 권장합니다.",
            }
            if factors["high_triglycerides"]
            else {
                "status": "정상",
                "value": f"{tg:.1f}mg/dL",
                "threshold": "150mg/dL",
                "interpretation": f"중성지방이 {tg:.1f}mg/dL로 정상 범위입니다.",
                "recommendation": "현재 식단을 유지하세요.",
            }
        )

        hdl = measurements["hdl_mg_dl"]
        hdl_threshold = (
            self.CRITERIA["hdl"]["male"]
            if sex == "남"
            else self.CRITERIA["hdl"]["female"]
        )
        interpretations["low_hdl"] = (
            {
                "status": "위험",
                "value": f"{hdl:.1f}mg/dL",
                "threshold": f"{hdl_threshold}mg/dL",
                "interpretation": f"HDL 콜레스테롤이 {hdl:.1f}mg/dL로 낮습니다.",
                "recommendation": "유산소 운동과 불포화지방 섭취를 권장합니다.",
            }
            if factors["low_hdl"]
            else {
                "status": "정상",
                "value": f"{hdl:.1f}mg/dL",
                "threshold": f"{hdl_threshold}mg/dL",
                "interpretation": f"HDL 콜레스테롤이 {hdl:.1f}mg/dL로 정상 범위입니다.",
                "recommendation": "건강한 식단과 운동을 지속하세요.",
            }
        )

        return interpretations

    def generate_diagnostic_report(self, patient_id: int) -> Optional[str]:
        diagnosis = self.check_metabolic_syndrome(patient_id)
        if not diagnosis:
            return None

        risk_eval = self.evaluate_risk_level(patient_id)
        interpretations = self.interpret_risk_factors(patient_id) or {}

        lines = [
            f"[환자 정보 - ID: {patient_id}]",
            f"이름: {diagnosis['name']} ({diagnosis['sex']}, {diagnosis['age']}세)",
            f"검진일: {diagnosis['exam_at']}",
            "",
            "=" * 60,
            "대사증후군 진단 평가 결과",
            "=" * 60,
            "",
        ]

        if diagnosis["has_metabolic_syndrome"]:
            lines.append(
                f"⚠️ 대사증후군: 진단 ({diagnosis['criteria_met']}/5 기준 충족)"
            )
        else:
            lines.append(f"✅ 대사증후군: 해당 없음 ({diagnosis['criteria_met']}/5 기준)")

        if risk_eval:
            lines.append(f"위험도: {risk_eval['risk_label']} ({risk_eval['risk_score']}점)")
            lines.append(f"평가: {risk_eval['risk_description']}")
        lines.append("")
        lines.append("세부 평가:")
        lines.append("-" * 60)

        risk_labels = {
            "abdominal_obesity": "1. 복부비만",
            "high_blood_pressure": "2. 고혈압",
            "high_fasting_glucose": "3. 공복혈당장애",
            "high_triglycerides": "4. 고중성지방혈증",
            "low_hdl": "5. 저HDL콜레스테롤혈증",
        }

        for key, label in risk_labels.items():
            interp = interpretations.get(key)
            if not interp:
                continue
            status_icon = "⚠️" if interp["status"] == "위험" else "✅"
            lines.extend(
                [
                    "",
                    label,
                    f"  {status_icon} 상태: {interp['status']}",
                    f"  측정값: {interp['value']} (기준: {interp['threshold']})",
                    f"  해석: {interp['interpretation']}",
                    f"  권장사항: {interp['recommendation']}",
                ]
            )

        lines.extend(
            [
                "",
                "-" * 60,
                "종합 권장사항:",
                "-" * 60,
            ]
        )

        if diagnosis["has_metabolic_syndrome"]:
            lines.extend(
                [
                    "• 대사증후군으로 진단되어 의료진과의 정기적인 상담이 필요합니다.",
                    "• 생활습관 개선: 규칙적인 운동(주 5회, 30분 이상), 균형잡힌 식단",
                    "• 체중 감량: 현재 체중의 5-10% 감량 목표",
                    "• 금연 및 절주",
                    "• 스트레스 관리 및 충분한 수면",
                ]
            )
        else:
            if diagnosis["criteria_met"] >= 2:
                lines.extend(
                    [
                        "• 대사증후군 전단계로 예방적 관리가 중요합니다.",
                        "• 위험 요인 개선을 위한 생활습관 교정이 필요합니다.",
                    ]
                )
            elif diagnosis["criteria_met"] == 1:
                lines.extend(
                    [
                        "• 1개의 위험 요인이 있어 예방적 관리가 필요합니다.",
                        "• 현재 상태가 악화되지 않도록 건강한 생활습관을 유지하세요.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "• 현재 대사증후군 위험이 없으나 정기적인 검진을 권장합니다.",
                        "• 건강한 생활습관을 꾸준히 유지하세요.",
                    ]
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 내부 헬퍼
    # ------------------------------------------------------------------ #

    def _check_abdominal_obesity(self, waist_cm: float, sex: str) -> bool:
        if waist_cm is None:
            return False
        threshold = self.CRITERIA["waist"]["male"] if sex == "남" else self.CRITERIA["waist"]["female"]
        return waist_cm >= threshold

    def _check_blood_pressure(self, systolic_mmHg: int, diastolic_mmHg: int) -> bool:
        if systolic_mmHg is None or diastolic_mmHg is None:
            return False
        return (
            systolic_mmHg >= self.CRITERIA["blood_pressure"]["systolic"]
            or diastolic_mmHg >= self.CRITERIA["blood_pressure"]["diastolic"]
        )

    def _check_fasting_glucose(self, fbg_mg_dl: float) -> bool:
        if fbg_mg_dl is None:
            return False
        return fbg_mg_dl >= self.CRITERIA["fasting_glucose"]

    def _check_triglycerides(self, tg_mg_dl: float) -> bool:
        if tg_mg_dl is None:
            return False
        return tg_mg_dl >= self.CRITERIA["triglycerides"]

    def _check_hdl(self, hdl_mg_dl: float, sex: str) -> bool:
        if hdl_mg_dl is None:
            return False
        threshold = self.CRITERIA["hdl"]["male"] if sex == "남" else self.CRITERIA["hdl"]["female"]
        return hdl_mg_dl < threshold


__all__ = ["PatientDatabase"]
