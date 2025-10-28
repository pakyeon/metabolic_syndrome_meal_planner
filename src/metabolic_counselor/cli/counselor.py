"""Command-line interface for metabolic syndrome meal planning."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from os import getenv

# Ensure the src directory is discoverable when invoking this module as a script.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv()

def _resolve_temperature(default: float = 0.3) -> float:
    raw = getenv("OPENAI_CHAT_TEMPERATURE")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            "OPENAI_CHAT_TEMPERATURE must be a numeric value."
        ) from exc


def _resolve_chunk_days(default: int = 14) -> int:
    raw = getenv("MEAL_PLAN_CHUNK_DAYS")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MEAL_PLAN_CHUNK_DAYS must be an integer.") from exc
    return max(1, value)


DEFAULT_CHAT_MODEL = getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_CHAT_TEMPERATURE = _resolve_temperature()
MAX_CHUNK_DAYS = _resolve_chunk_days()

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from metabolic_counselor.agents import MealPlanAgent, MealPlanResult
from metabolic_counselor.context import PatientContextProvider
from metabolic_counselor.data import PatientDatabase
from metabolic_counselor.services import (
    CounselorProfile,
    MealPlanRequest,
    RequestNormalizer,
    RevisionInstruction,
)


@dataclass
class SessionState:
    patient_id: Optional[int] = None
    request: Optional[MealPlanRequest] = None
    latest_plan: Optional["MealPlanResult"] = None  # forward reference
    latest_path: Optional[Path] = None


class MealPlanCLI:
    """대사증후군 상담 지원 CLI."""

    def __init__(
        self,
        db: Optional[PatientDatabase] = None,
        model: Optional[BaseChatModel] = None,
        output_root: Path = Path("meal_plans"),
    ):
        self.db = db or PatientDatabase()
        self.context_provider = PatientContextProvider(self.db)
        self.normalizer = RequestNormalizer()
        self.output_root = output_root
        self.state = SessionState()
        self.counselor_profile = CounselorProfile.MEDICAL
        self.agent = MealPlanAgent(model or self._bootstrap_model())

    def _bootstrap_model(self) -> BaseChatModel:
        if not getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Store it in your environment or a .env file."
            )
        model_name = DEFAULT_CHAT_MODEL
        temperature = DEFAULT_CHAT_TEMPERATURE
        return ChatOpenAI(model=model_name, temperature=temperature)

    # ------------------------------------------------------------------ #
    # CLI flow
    # ------------------------------------------------------------------ #

    def run(self):
        self._print_banner()

        while True:
            patient_id = self._prompt_patient()
            if patient_id is None:
                print("프로그램을 종료합니다.")
                return

            self.state = SessionState(patient_id=patient_id)
            patient_context = self.context_provider.get_patient_context(
                patient_id, format="standard"
            )
            if not patient_context:
                print("환자 정보를 불러올 수 없습니다. 다른 환자를 선택해주세요.")
                continue

            print("\n=== 환자 위험 요약 ===")
            print(patient_context)

            self._interaction_loop(patient_id, patient_context)

    def _interaction_loop(self, patient_id: int, patient_context: str):
        while True:
            print("\n명령을 입력하세요: [plan | modify | show | history | back | exit]")
            command = input("> ").strip().lower()

            if command == "plan":
                self._handle_plan(patient_id, patient_context)
            elif command == "modify":
                self._handle_modify(patient_id, patient_context)
            elif command == "show":
                self._handle_show()
            elif command == "history":
                self._handle_history(patient_id)
            elif command == "back":
                break
            elif command == "exit":
                print("세션을 종료합니다.")
                raise SystemExit(0)
            else:
                print("지원하지 않는 명령입니다.")

    # ------------------------------------------------------------------ #
    # Command handlers
    # ------------------------------------------------------------------ #

    def _handle_plan(self, patient_id: int, patient_context: str):
        try:
            request = self._collect_plan_request(patient_id)
            plan = self._generate_sequence(patient_context, request)
            filepath = self._persist_plan(plan, request)
            self.state.request = request
            self.state.latest_plan = plan
            self.state.latest_path = filepath

            print("\n=== 생성된 식단 ===")
            print(plan.markdown)
            print(f"\n저장 완료: {filepath}")
        except ValueError as exc:
            print(f"요청이 올바르지 않습니다: {exc}")
        except Exception as exc:  # pragma: no cover
            print(f"식단 생성 중 오류가 발생했습니다: {exc}")

    def _generate_sequence(
        self, patient_context: str, request: MealPlanRequest
    ) -> MealPlanResult:
        max_days = max(1, MAX_CHUNK_DAYS)
        if request.duration_days <= max_days:
            return self.agent.generate_plan(patient_context, request)

        segments: List[tuple[MealPlanRequest, MealPlanResult]] = []
        previous_markdown: Optional[str] = None
        current_start = request.start_date
        while current_start <= request.end_date:
            current_end = min(
                current_start + timedelta(days=max_days - 1), request.end_date
            )
            chunk_request = replace(
                request, start_date=current_start, end_date=current_end
            )
            chunk_result = self.agent.generate_plan(
                patient_context,
                chunk_request,
                previous_plan=previous_markdown,
            )
            segments.append((chunk_request, chunk_result))
            previous_markdown = chunk_result.markdown
            current_start = current_end + timedelta(days=1)

        combined_sections = [
            "\n".join(
                [
                    f"### {chunk_request.start_date.isoformat()} ~ {chunk_request.end_date.isoformat()}",
                    "",
                    chunk_result.markdown,
                ]
            )
            for chunk_request, chunk_result in segments
        ]
        combined_markdown = "\n\n".join(combined_sections)

        metadata = {
            "status": "created",
            "chunks": str(len(segments)),
            "chunk_span_days": str(max_days),
        }
        return MealPlanResult(
            markdown=combined_markdown,
            start_date=request.start_date,
            end_date=request.end_date,
            patient_id=request.patient_id,
            mode="create",
            metadata=metadata,
        )

    def _handle_modify(self, patient_id: int, patient_context: str):
        if not self.state.latest_plan or not self.state.request:
            print("먼저 plan 명령으로 식단을 생성한 뒤 수정할 수 있습니다.")
            return

        try:
            revision = self._collect_revision()
            plan = self.agent.revise_plan(
                patient_context,
                self.state.request,
                revision,
                self.state.latest_plan.markdown,  # type: ignore[union-attr]
            )
            filepath = self._persist_plan(plan, self.state.request, revision)
            self.state.latest_plan = plan
            self.state.latest_path = filepath

            print("\n=== 수정된 식단 ===")
            print(plan.markdown)
            print(f"\n저장 완료: {filepath}")
        except ValueError as exc:
            print(f"수정 요청이 올바르지 않습니다: {exc}")
        except Exception as exc:  # pragma: no cover
            print(f"식단 수정 중 오류가 발생했습니다: {exc}")

    def _handle_show(self):
        if not self.state.latest_plan:
            print("표시할 식단이 없습니다. plan 명령으로 먼저 생성해주세요.")
            return
        print("\n=== 최신 식단 ===")
        print(self.state.latest_plan.markdown)
        if self.state.latest_path:
            print(f"\n저장 위치: {self.state.latest_path}")

    def _handle_history(self, patient_id: int):
        folder = self.output_root / str(patient_id)
        if not folder.exists():
            print("저장된 식단이 없습니다.")
            return
        files = sorted(folder.glob("mealplan_*.md"))
        if not files:
            print("저장된 식단이 없습니다.")
            return
        print("\n=== 저장 이력 ===")
        for path in files[-10:]:
            print(path.name)

    # ------------------------------------------------------------------ #
    # Input helpers
    # ------------------------------------------------------------------ #

    def _collect_plan_request(self, patient_id: int) -> MealPlanRequest:
        print("\n=== 식단 요청 입력 ===")
        start_date = input("시작일 (YYYY-MM-DD): ").strip()
        end_date = input("종료일 (YYYY-MM-DD): ").strip()
        calories = input("일일 칼로리 목표 (예: 1800kcal, 미입력 가능): ").strip()
        preferred = input("선호 식품 (쉼표 구분, 미입력 가능): ").split(",")
        avoided = input("기피 식품 (쉼표 구분, 미입력 가능): ").split(",")
        snack_policy = input("간식 정책 (포함/제외, 기본=포함): ").strip()
        notes = input("특이 요청 (선택): ")

        return self.normalizer.normalize_plan_request(
            counselor_profile=self.counselor_profile,
            patient_id=patient_id,
            start_date_str=start_date,
            end_date_str=end_date,
            calorie_text=calories,
            preferred_tokens=preferred,
            avoided_tokens=avoided,
            snack_policy=snack_policy,
            notes=notes,
        )

    def _collect_revision(self) -> RevisionInstruction:
        print("\n=== 수정 요청 입력 ===")
        dates = input("수정할 날짜 (쉼표로 구분, YYYY-MM-DD): ").split(",")
        meals = input("식사 구분 (예: 아침,저녁 / 비우면 전체): ").split(",")
        notes = input("수정 지시 사항: ")
        return self.normalizer.normalize_revision(dates, meals, notes)

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def _persist_plan(
        self,
        plan: MealPlanResult,
        request: MealPlanRequest,
        revision: Optional[RevisionInstruction] = None,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.output_root / str(plan.patient_id)
        folder.mkdir(parents=True, exist_ok=True)

        markdown_path = folder / f"mealplan_{timestamp}.md"
        markdown_path.write_text(plan.markdown, encoding="utf-8")

        log_payload = {
            "mode": plan.mode,
            "request": {
                "counselor_profile": request.counselor_profile.value,
                "patient_id": request.patient_id,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "target_calories": request.target_calories,
                "preferred_foods": request.preferred_foods,
                "avoided_foods": request.avoided_foods,
                "snack_policy": request.snack_policy,
                "special_notes": request.special_notes,
            },
            "revision": revision.describe() if revision else None,
            "saved_at": timestamp,
            "metadata": plan.metadata,
        }
        (folder / f"mealplan_{timestamp}.json").write_text(
            json.dumps(log_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return markdown_path

    # ------------------------------------------------------------------ #
    # Selection helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _print_banner():
        print("=" * 64)
        print("대사증후군 상담 지원 식단 에이전트 (LangGraph 기반)")
        print("=" * 64)

    def _prompt_patient(self) -> Optional[int]:
        print("\n=== 환자 목록 ===")
        entries = []
        for patient in self.db.get_all_patients():
            diagnosis = self.db.check_metabolic_syndrome(patient["patient_id"])
            if not diagnosis:
                continue
            exam_raw = diagnosis.get("exam_at")

            def _parse_exam(value: str) -> datetime:
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                        try:
                            return datetime.strptime(value, fmt)
                        except ValueError:
                            continue
                    return datetime.min

            parsed_exam = _parse_exam(exam_raw) if exam_raw else datetime.min
            entries.append((patient, diagnosis, parsed_exam, exam_raw))

        entries.sort(key=lambda item: item[2], reverse=True)

        for display_idx, (patient, diagnosis, parsed_exam, exam_raw) in enumerate(
            entries, start=1
        ):
            status = "🔴 진단" if diagnosis["has_metabolic_syndrome"] else "🟢 정상"
            exam_label = parsed_exam.date().isoformat() if exam_raw else "정보 없음"
            print(
                f"{display_idx:2d}. {patient['name']} "
                f"({patient['sex']}, {patient['age']}세, ID {patient['patient_id']}) "
                f"- {status} | 최근 검진일: {exam_label}"
            )

        while True:
            raw = input("\n환자 ID 입력 (종료: exit): ").strip().lower()
            if raw in {"exit", "quit"}:
                return None
            try:
                patient_id = int(raw)
            except ValueError:
                print("숫자로 된 환자 ID를 입력해주세요.")
                continue
            if self.db.get_patient(patient_id):
                return patient_id
            print("해당 환자를 찾을 수 없습니다.")


def main():
    cli = MealPlanCLI()
    cli.run()


if __name__ == "__main__":  # pragma: no cover
    main()
