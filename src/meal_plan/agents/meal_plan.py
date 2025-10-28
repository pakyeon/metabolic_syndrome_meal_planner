"""LangGraph agent for metabolic syndrome meal planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from textwrap import dedent
from typing import Dict, List, Literal, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from meal_plan.services import MealPlanRequest, RevisionInstruction

SYSTEM_PROMPT = dedent(
    """
    당신은 대사증후군 상담을 지원하는 식단 전문가이다.
    환자의 임상 정보를 해석해 한식 중심의 저당·저염·고식이섬유 식단을 설계한다.
    - 필요한 경우 혈당 관리, 체중 조절, 운동 후 회복을 위해 가공당과 포화지방을 최소화한다.
    - 각 식사는 구체적인 음식명과 1인분 분량(예: g, 컵, 작은접시 등)을 제시한다.
    - 재료 선택 시 현지 조리법과 계절 채소를 우선 고려한다.
    - 요청된 칼로리 목표와 간식 정책을 벗어나지 않도록 한다.
    - 환자의 기저질환/위험요인을 고려해 나트륨을 제한하고 수분 섭취 지침을 포함한다.
    출력은 반드시 날짜 × (아침, 점심, 저녁, 간식) 4열을 가진 마크다운 표 형식이어야 한다.
    표 아래에는 칼로리 요약과 상담 포인트를 항목으로 정리한다.
    """
).strip()


class MealPlanState(TypedDict, total=False):
    mode: Literal["create", "revise"]
    request: MealPlanRequest
    patient_context: str
    revision: RevisionInstruction
    existing_plan: str
    messages: List[BaseMessage]
    llm_markdown: str
    previous_plan: str


@dataclass
class MealPlanResult:
    markdown: str
    start_date: date
    end_date: date
    patient_id: int
    mode: Literal["create", "revise"]
    metadata: Dict[str, str]


class MealPlanAgent:
    """LangGraph 기반 식단 생성/수정 에이전트."""

    def __init__(self, chat_model: BaseChatModel):
        self.chat_model = chat_model
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(MealPlanState)
        builder.add_node("build_messages", self._build_messages)
        builder.add_node("invoke_llm", self._invoke_llm)
        builder.add_edge(START, "build_messages")
        builder.add_edge("build_messages", "invoke_llm")
        builder.add_edge("invoke_llm", END)
        return builder.compile()

    def generate_plan(
        self,
        patient_context: str,
        request: MealPlanRequest,
        previous_plan: Optional[str] = None,
    ) -> MealPlanResult:
        payload: Dict[str, object] = {
            "mode": "create",
            "patient_context": patient_context.strip(),
            "request": request,
        }
        if previous_plan:
            payload["previous_plan"] = previous_plan.strip()

        state = self.graph.invoke(payload)
        return MealPlanResult(
            markdown=state["llm_markdown"],
            start_date=request.start_date,
            end_date=request.end_date,
            patient_id=request.patient_id,
            mode="create",
            metadata={"status": "created"},
        )

    def revise_plan(
        self,
        patient_context: str,
        request: MealPlanRequest,
        revision: RevisionInstruction,
        existing_plan: str,
    ) -> MealPlanResult:
        state = self.graph.invoke(
            {
                "mode": "revise",
                "patient_context": patient_context.strip(),
                "request": request,
                "revision": revision,
                "existing_plan": existing_plan.strip(),
            }
        )
        return MealPlanResult(
            markdown=state["llm_markdown"],
            start_date=request.start_date,
            end_date=request.end_date,
            patient_id=request.patient_id,
            mode="revise",
            metadata={"status": "revised", "change_notes": revision.change_notes},
        )

    # ------------------------------------------------------------------ #
    # LangGraph nodes
    # ------------------------------------------------------------------ #

    def _build_messages(self, state: MealPlanState) -> Dict[str, List[BaseMessage]]:
        request = state["request"]
        user_sections: List[str] = [
            "환자 컨텍스트:",
            state.get("patient_context", "").strip() or "환자 정보가 제공되지 않았습니다.",
            "",
            "요청 파라미터:",
            *request.summary_lines(),
            "",
            "출력 규칙:",
            "- 날짜는 YYYY-MM-DD 형식으로 표에 기입한다.",
            "- 표는 날짜 오름차순으로 정렬하고 각 행에 아침/점심/저녁/간식을 채운다.",
            "- 각 셀에는 음식, 조리법, 분량(예: g, 컵, 큰술)을 명시한다.",
            "- 표 아래에 총 칼로리와 상담사가 확인할 주의사항을 bullet으로 정리한다.",
            "- 요청된 시작일과 종료일을 모두 포함해 기간 내 모든 날짜를 빠짐없이 기입한다.",
        ]

        previous_plan = state.get("previous_plan")
        if previous_plan:
            user_sections.extend(
                [
                    "",
                    "이전 기간 식단(연속성 참고용):",
                    previous_plan,
                    "",
                    "이번 기간 식단은 위 내용을 기반으로 날짜가 겹치지 않게 이어서 제안한다. "
                    "음식 구성을 다양화하되 칼로리, 간식 정책 등 기준은 유지한다.",
                ]
            )

        if state.get("mode") == "revise":
            revision = state["revision"]
            existing_plan = state.get("existing_plan", "")
            user_sections.extend(
                [
                    "",
                    "기존 식단 요약:",
                    existing_plan,
                    "",
                    "수정 지시:",
                    revision.describe(),
                    "지시된 날짜/식사 외에는 기존 내용을 유지한다.",
                ]
            )

        system = SystemMessage(content=SYSTEM_PROMPT)
        human = HumanMessage(content="\n".join(filter(None, user_sections)))
        return {"messages": [system, human]}

    def _invoke_llm(self, state: MealPlanState) -> Dict[str, str]:
        response = self.chat_model.invoke(state["messages"])
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)
        return {"llm_markdown": content.strip()}


__all__ = ["MealPlanAgent", "MealPlanResult"]
