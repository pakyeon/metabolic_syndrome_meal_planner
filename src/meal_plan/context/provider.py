"""Human-readable patient context builders for LLM prompts."""

from __future__ import annotations

from typing import List, Optional

from meal_plan.data import PatientDatabase


class PatientContextProvider:
    """환자 정보를 다양한 포맷으로 제공한다."""

    def __init__(self, db: PatientDatabase):
        self.db = db

    def get_patient_context(self, patient_id: int, format: str = "standard") -> Optional[str]:
        if format == "standard":
            return self._format_standard(patient_id)
        if format == "detailed":
            return self._format_detailed(patient_id)
        if format == "compact":
            return self._format_compact(patient_id)
        return self._format_standard(patient_id)

    def list_patients(self, format: str = "compact") -> List[str]:
        results: List[str] = []
        for patient in self.db.get_all_patients():
            context = self.get_patient_context(patient["patient_id"], format=format)
            if context:
                results.append(context)
        return results

    def get_metabolic_syndrome_patients(self, format: str = "compact") -> List[str]:
        results: List[str] = []
        for diagnosis in self.db.get_patients_with_metabolic_syndrome():
            context = self.get_patient_context(diagnosis["patient_id"], format=format)
            if context:
                results.append(context)
        return results

    def format_for_llm_context(self, patient_id: Optional[int]) -> str:
        if patient_id is None:
            return ""
        context = self.get_patient_context(patient_id, format="standard")
        if not context:
            return ""
        return (
            "다음은 현재 상담 중인 환자의 정보입니다. 이 정보를 참고하여 답변해주세요:\n\n"
            f"{context}\n\n"
            "※ 위 환자 정보를 고려하되, 검색된 문서 내용을 우선으로 답변하세요."
        )

    # ------------------------------------------------------------------ #
    # 내부 포맷터
    # ------------------------------------------------------------------ #

    def _format_standard(self, patient_id: int) -> Optional[str]:
        diagnosis = self.db.check_metabolic_syndrome(patient_id)
        if not diagnosis:
            return None
        risk_eval = self.db.evaluate_risk_level(patient_id)
        lines = [
            f"[환자 정보 - ID: {patient_id}]",
            f"이름: {diagnosis['name']} ({diagnosis['sex']}, {diagnosis['age']}세)",
            f"대사증후군 진단: {'있음' if diagnosis['has_metabolic_syndrome'] else '없음'} "
            f"({diagnosis['criteria_met']}/5 기준 충족)",
            f"위험도: {risk_eval['risk_label'] if risk_eval else '미평가'}",
            "",
        ]
        risk_labels = {
            "abdominal_obesity": "복부비만",
            "high_blood_pressure": "고혈압",
            "high_fasting_glucose": "공복혈당장애",
            "high_triglycerides": "고중성지방",
            "low_hdl": "저HDL콜레스테롤",
        }
        measurements = diagnosis["measurements"]
        has_risk = False
        for key, label in risk_labels.items():
            if diagnosis["risk_factors"][key]:
                if not has_risk:
                    lines.append("위험 요인:")
                    has_risk = True
                if key == "abdominal_obesity":
                    lines.append(f"⚠️ {label} (허리둘레 {measurements['waist_cm']:.1f}cm)")
                elif key == "high_blood_pressure":
                    lines.append(
                        f"⚠️ {label} ({measurements['systolic_mmHg']}/{measurements['diastolic_mmHg']}mmHg)"
                    )
                elif key == "high_fasting_glucose":
                    lines.append(f"⚠️ {label} (공복혈당 {measurements['fbg_mg_dl']:.1f}mg/dL)")
                elif key == "high_triglycerides":
                    lines.append(f"⚠️ {label} (중성지방 {measurements['tg_mg_dl']:.1f}mg/dL)")
                elif key == "low_hdl":
                    lines.append(f"⚠️ {label} (HDL {measurements['hdl_mg_dl']:.1f}mg/dL)")
        if not has_risk:
            lines.append("위험 요인: 없음 ✅")
        return "\n".join(lines)

    def _format_detailed(self, patient_id: int) -> Optional[str]:
        return self.db.generate_diagnostic_report(patient_id)

    def _format_compact(self, patient_id: int) -> Optional[str]:
        diagnosis = self.db.check_metabolic_syndrome(patient_id)
        if not diagnosis:
            return None
        risk_eval = self.db.evaluate_risk_level(patient_id)
        return (
            f"환자 {patient_id}번 ({diagnosis['name']}, {diagnosis['sex']}, {diagnosis['age']}세): "
            f"대사증후군 {'진단' if diagnosis['has_metabolic_syndrome'] else '없음'}, "
            f"{risk_eval['risk_label'] if risk_eval else '미평가'}, "
            f"위험요인 {diagnosis['criteria_met']}개"
        )


class PatientSession:
    """선택된 환자 정보를 유지하는 세션 컨테이너."""

    def __init__(self, provider: PatientContextProvider):
        self.provider = provider
        self.current_patient_id: Optional[int] = None

    def select_patient(self, patient_id: int) -> Optional[str]:
        context = self.provider.get_patient_context(patient_id, format="standard")
        if context:
            self.current_patient_id = patient_id
            return context
        return None

    def get_current_patient_id(self) -> Optional[int]:
        return self.current_patient_id

    def get_current_context(self) -> str:
        if self.current_patient_id is None:
            return ""
        return self.provider.format_for_llm_context(self.current_patient_id)

    def is_patient_selected(self) -> bool:
        return self.current_patient_id is not None

    def clear_selection(self):
        self.current_patient_id = None


__all__ = ["PatientContextProvider", "PatientSession"]
