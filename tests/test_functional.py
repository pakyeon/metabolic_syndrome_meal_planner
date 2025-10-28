import sys
import types
import unittest
from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _ensure_stubbed_dependencies():
    """테스트 환경에서 LangChain/LangGraph 패키지가 없을 때 최소 스텁을 제공한다."""
    # --- langchain_core stubs ---
    try:
        import langchain_core  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - only executed in bare envs
        langchain_core = types.ModuleType("langchain_core")
        sys.modules["langchain_core"] = langchain_core

        lm_module = types.ModuleType("langchain_core.language_models")
        sys.modules["langchain_core.language_models"] = lm_module

        chat_models = types.ModuleType("langchain_core.language_models.chat_models")
        sys.modules["langchain_core.language_models.chat_models"] = chat_models

        class BaseChatModel:
            model_name: str = "stub-model"

            def invoke(self, messages):
                result = self._generate(messages=messages)
                return result.generations[-1].message

            def _generate(self, *args, **kwargs):  # pragma: no cover - interface
                raise NotImplementedError

        chat_models.BaseChatModel = BaseChatModel

        messages_module = types.ModuleType("langchain_core.messages")
        sys.modules["langchain_core.messages"] = messages_module

        class BaseMessage:
            def __init__(self, content: str = ""):
                self.content = content

        class SystemMessage(BaseMessage):
            pass

        class HumanMessage(BaseMessage):
            pass

        class AIMessage(BaseMessage):
            pass

        @dataclass
        class ChatGeneration:
            message: BaseMessage

        @dataclass
        class ChatResult:
            generations: list[ChatGeneration]

        messages_module.BaseMessage = BaseMessage
        messages_module.SystemMessage = SystemMessage
        messages_module.HumanMessage = HumanMessage
        messages_module.AIMessage = AIMessage
        messages_module.ChatGeneration = ChatGeneration
        messages_module.ChatResult = ChatResult

    else:
        # Ensure message classes exist; if import succeeded, no action needed
        pass

    # --- langgraph stubs ---
    try:
        import langgraph  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        langgraph = types.ModuleType("langgraph")
        sys.modules["langgraph"] = langgraph

        graph_module = types.ModuleType("langgraph.graph")
        sys.modules["langgraph.graph"] = graph_module

        START = "__start__"
        END = "__end__"

        class StateGraph:
            def __init__(self, _state_type):
                self._nodes = []

            def add_node(self, name, func):
                self._nodes.append(func)

            def add_edge(self, *_args, **_kwargs):
                return self

            def compile(self):
                class Compiled:
                    def __init__(self, nodes):
                        self._nodes = nodes

                    def invoke(self, initial_state):
                        state = dict(initial_state)
                        for func in self._nodes:
                            updates = func(state)
                            if updates:
                                state.update(updates)
                        return state

                return Compiled(self._nodes)

        graph_module.START = START
        graph_module.END = END
        graph_module.StateGraph = StateGraph


_ensure_stubbed_dependencies()

from metabolic_counselor.agents import MealPlanAgent
from metabolic_counselor.cli.counselor import OfflinePlanModel
from metabolic_counselor.context import PatientContextProvider
from metabolic_counselor.data import PatientDatabase
from metabolic_counselor.services import (
    CounselorProfile,
    MealPlanRequest,
    RequestNormalizer,
)


class RequestNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.normalizer = RequestNormalizer()

    def test_plan_request_normalization(self):
        request = self.normalizer.normalize_plan_request(
            counselor_profile=CounselorProfile.MEDICAL,
            patient_id=1,
            start_date_str="2024-05-01",
            end_date_str="2024-05-03",
            calorie_text="1800 kcal",
            preferred_tokens=["생선, 채소"],
            avoided_tokens=["튀김"],
            snack_policy="포함",
            hydration_focus=True,
            notes="운동 후 회복식 포함",
        )

        self.assertEqual(request.patient_id, 1)
        self.assertEqual(request.start_date, date(2024, 5, 1))
        self.assertEqual(request.end_date, date(2024, 5, 3))
        self.assertEqual(request.target_calories, 1800)
        self.assertEqual(request.preferred_foods, ["생선", "채소"])
        self.assertEqual(request.avoided_foods, ["튀김"])
        self.assertTrue(request.hydration_focus)
        self.assertIn("운동 후 회복식", request.special_notes)
        self.assertEqual(request.duration_days, 3)

    def test_revision_invalid_meal(self):
        with self.assertRaises(ValueError):
            self.normalizer.normalize_revision(
                date_tokens=["2024-05-01"],
                meal_tokens=["breakfast"],
                change_notes="한식 위주로 변경",
            )


class MealPlanAgentTest(unittest.TestCase):
    def setUp(self):
        self.agent = MealPlanAgent(OfflinePlanModel())
        self.request = MealPlanRequest(
            counselor_profile=CounselorProfile.MEDICAL,
            patient_id=1,
            start_date=date(2024, 5, 1),
            end_date=date(2024, 5, 2),
            target_calories=1800,
            preferred_foods=["생선", "채소"],
            avoided_foods=["튀김"],
            snack_policy="포함",
            hydration_focus=True,
            special_notes="운동 후 회복식 포함",
        )

    def test_generate_plan_offline(self):
        db = PatientDatabase(PROJECT_ROOT / "metabolic_health.sqlite")
        provider = PatientContextProvider(db)
        context = provider.get_patient_context(1, format="standard")
        self.assertIsNotNone(context)

        result = self.agent.generate_plan(context or "", self.request)
        self.assertIn("| 날짜 | 아침 | 점심 | 저녁 | 간식 |", result.markdown)
        self.assertEqual(result.metadata["status"], "created")
        self.assertEqual(result.start_date, date(2024, 5, 1))
        self.assertEqual(result.end_date, date(2024, 5, 2))

    def test_revision_flow(self):
        initial_plan = self.agent.generate_plan("환자 컨텍스트", self.request)
        revision = RequestNormalizer().normalize_revision(
            date_tokens=["2024-05-02"],
            meal_tokens=["아침,저녁"],
            change_notes="탄수화물 비중을 낮춰 주세요.",
        )

        revised = self.agent.revise_plan(
            "환자 컨텍스트",
            self.request,
            revision,
            initial_plan.markdown,
        )

        self.assertEqual(revised.metadata["status"], "revised")
        self.assertIn("탄수화물", revised.metadata["change_notes"])


if __name__ == "__main__":
    unittest.main()
