"""Normalization utilities for counselor input."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Sequence


class CounselorProfile(str, Enum):
    """상담사 역할 분류."""

    MEDICAL = "medical"
    EXERCISE = "exercise"

    @property
    def label(self) -> str:
        return "의료 상담사" if self is CounselorProfile.MEDICAL else "운동 상담사"


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("날짜는 YYYY-MM-DD 형식으로 입력해주세요.") from exc


def _parse_calories(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        raise ValueError("칼로리 값에서 숫자를 추출할 수 없습니다.")
    calories = int(digits)
    if not 800 <= calories <= 4000:
        raise ValueError("칼로리는 800kcal 이상 4000kcal 이하 범위로 입력해주세요.")
    return calories


def _parse_list(tokens: Sequence[str]) -> List[str]:
    items: List[str] = []
    for token in tokens:
        for chunk in token.replace(";", ",").split(","):
            name = chunk.strip()
            if name:
                items.append(name)
    return items


@dataclass(frozen=True)
class MealPlanRequest:
    """식단 생성 요청."""

    counselor_profile: CounselorProfile
    patient_id: int
    start_date: date
    end_date: date
    target_calories: Optional[int]
    preferred_foods: List[str] = field(default_factory=list)
    avoided_foods: List[str] = field(default_factory=list)
    snack_policy: str = "포함"
    special_notes: str = ""

    def __post_init__(self):
        if self.start_date > self.end_date:
            raise ValueError("기간의 시작일은 종료일보다 앞서야 합니다.")

    @property
    def duration_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    def summary_lines(self) -> List[str]:
        cal_text = (
            f"{self.target_calories}kcal"
            if self.target_calories is not None
            else "칼로리 자유"
        )
        lines = [
            f"- 기간: {self.start_date.isoformat()} ~ {self.end_date.isoformat()} "
            f"(총 {self.duration_days}일)",
            f"- 일일 목표 칼로리: {cal_text}",
            f"- 간식 정책: {self.snack_policy}",
        ]
        if self.preferred_foods:
            lines.append("- 선호 식품: " + ", ".join(self.preferred_foods))
        if self.avoided_foods:
            lines.append("- 기피 식품: " + ", ".join(self.avoided_foods))
        if self.special_notes:
            lines.append(f"- 특이 요청: {self.special_notes}")
        return lines


@dataclass(frozen=True)
class RevisionInstruction:
    """식단 수정 지시."""

    target_dates: List[date]
    meals_to_update: List[str]
    change_notes: str

    def describe(self) -> str:
        date_text = ", ".join(d.isoformat() for d in self.target_dates)
        meals = ", ".join(self.meals_to_update) if self.meals_to_update else "전체"
        return f"{date_text}의 {meals} 수정 요청: {self.change_notes}"


class RequestNormalizer:
    """상담사 입력을 구조화하는 파서."""

    MEAL_LABELS = ("아침", "점심", "저녁", "간식")

    def __init__(self, default_snack_policy: str = "포함"):
        self.default_snack_policy = default_snack_policy

    def normalize_plan_request(
        self,
        counselor_profile: CounselorProfile,
        patient_id: int,
        start_date_str: str,
        end_date_str: str,
        calorie_text: str,
        preferred_tokens: Sequence[str],
        avoided_tokens: Sequence[str],
        snack_policy: Optional[str],
        notes: str,
    ) -> MealPlanRequest:
        start = _parse_date(start_date_str)
        end = _parse_date(end_date_str)
        calories = _parse_calories(calorie_text)
        preferred = _parse_list(preferred_tokens)
        avoided = _parse_list(avoided_tokens)
        policy = snack_policy.strip() if snack_policy else self.default_snack_policy
        return MealPlanRequest(
            counselor_profile=counselor_profile,
            patient_id=patient_id,
            start_date=start,
            end_date=end,
            target_calories=calories,
            preferred_foods=preferred,
            avoided_foods=avoided,
            snack_policy=policy or self.default_snack_policy,
            special_notes=notes.strip(),
        )

    def normalize_revision(
        self,
        date_tokens: Sequence[str],
        meal_tokens: Sequence[str],
        change_notes: str,
    ) -> RevisionInstruction:
        if not change_notes.strip():
            raise ValueError("수정 지시 내용을 입력해주세요.")

        target_dates = [_parse_date(token) for token in date_tokens if token.strip()]
        if not target_dates:
            raise ValueError("수정 대상 날짜를 최소 1개 이상 입력해주세요.")

        normalized_meals: List[str] = []
        for token in meal_tokens:
            for label in token.replace(";", ",").split(","):
                candidate = label.strip()
                if not candidate:
                    continue
                if candidate not in self.MEAL_LABELS:
                    raise ValueError(
                        f"지원하지 않는 식사 구분입니다: {candidate} "
                        f"(허용: {', '.join(self.MEAL_LABELS)})"
                    )
                normalized_meals.append(candidate)

        return RevisionInstruction(
            target_dates=target_dates,
            meals_to_update=normalized_meals,
            change_notes=change_notes.strip(),
        )


__all__ = [
    "CounselorProfile",
    "MealPlanRequest",
    "RequestNormalizer",
    "RevisionInstruction",
]
