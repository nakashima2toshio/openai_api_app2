#
# chain_of_thought_patterns.py
# ==================================================
# Chain of Thought (CoT) パターン学習プログラム
# ==================================================
# 5種類の代表的なCoTパターンを実装し、
# OpenAI APIを使用して実際に動作を確認できます。
#
# [Usage] streamlit run chain_of_thought_patterns.py --server.port 8505
# ==================================================

import os
import sys
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

# プロジェクトパスの設定
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


# ==================================================
# 設定とデータクラス
# ==================================================
@dataclass
class CoTConfig:
    """CoTパターンの設定"""
    page_title: str = "Chain of Thought Patterns"
    page_icon: str = "🧠"
    default_model: str = "gpt-4o-mini"
    json_compatible_models: List[str] = None

    def __post_init__(self):
        if self.json_compatible_models is None:
            self.json_compatible_models = [
                "gpt-4o", "gpt-4o-mini",
                "gpt-4-1106-preview", "gpt-3.5-turbo-1106"
            ]


class CoTPatternType(Enum):
    """CoTパターンの種類"""
    STEP_BY_STEP = "Step-by-Step（逐次展開型）"
    HYPOTHESIS_TEST = "Hypothesis-Test（仮説検証型）"
    TREE_OF_THOUGHT = "Tree-of-Thought（分岐探索型）"
    PROS_CONS_DECISION = "Pros-Cons-Decision（賛否比較型）"
    PLAN_EXECUTE_REFLECT = "Plan-Execute-Reflect（反復改良型）"


# グローバル設定
config = CoTConfig()


# ==================================================
# 基底クラス
# ==================================================
class BaseCoTPattern(ABC):
    """
    CoTパターンの基底クラス

    全てのCoTパターンが共通して持つ機能を実装
    """

    def __init__(self, pattern_type: CoTPatternType):
        self.pattern_type = pattern_type
        self.client = OpenAI()
        self.model = self._get_model()

    def _get_model(self) -> str:
        """選択されたモデルを取得"""
        return st.session_state.get("selected_model", config.default_model)

    @abstractmethod
    def get_system_prompt(self) -> str:
        """システムプロンプトを取得（サブクラスで実装）"""
        pass

    @abstractmethod
    def get_result_model(self) -> BaseModel:
        """結果のPydanticモデルを取得（サブクラスで実装）"""
        pass

    @abstractmethod
    def create_ui(self) -> Dict[str, Any]:
        """UI要素を作成し、ユーザー入力を取得（サブクラスで実装）"""
        pass

    @abstractmethod
    def parse_response(self, response_text: str, inputs: Dict[str, Any]) -> BaseModel:
        """レスポンスをパースして結果モデルに変換（サブクラスで実装）"""
        pass

    def execute(self) -> Optional[BaseModel]:
        """CoTパターンの実行"""
        st.subheader(f"🔹 {self.pattern_type.value}")

        # UI作成とユーザー入力取得
        inputs = self.create_ui()

        # 実行ボタン
        if not st.button("実行", key=f"execute_{self.pattern_type.name}"):
            return None

        # プログレスバー
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # メッセージ作成
            status_text.text("メッセージを作成中...")
            progress_bar.progress(20)

            messages = self._create_messages(inputs)

            # API呼び出し
            status_text.text("APIを呼び出し中...")
            progress_bar.progress(50)

            response = self._call_api(messages, inputs.get("temperature", 0.3))

            # レスポンス解析
            status_text.text("レスポンスを解析中...")
            progress_bar.progress(80)

            result = self.parse_response(response, inputs)

            # 完了
            progress_bar.progress(100)
            status_text.text("完了!")

            # 結果表示
            self._display_result(result)

            return result

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            if st.checkbox("詳細を表示", key=f"error_{self.pattern_type.name}"):
                st.exception(e)
            return None
        finally:
            # プログレスバーをクリア
            progress_bar.empty()
            status_text.empty()

    def _create_messages(self, inputs: Dict[str, Any]) -> List[ChatCompletionMessageParam]:
        """メッセージリストを作成"""
        system_prompt = self.get_system_prompt()
        user_content = self._format_user_content(inputs)

        return [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(role="user", content=user_content)
        ]

    def _format_user_content(self, inputs: Dict[str, Any]) -> str:
        """ユーザー入力をフォーマット（デフォルト実装）"""
        # サブクラスでオーバーライド可能
        return "\n".join([f"{k}: {v}" for k, v in inputs.items() if k != "temperature"])

    def _call_api(self, messages: List[ChatCompletionMessageParam], temperature: float) -> str:
        """OpenAI APIを呼び出し"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _display_result(self, result: BaseModel):
        """結果を表示"""
        # JSON形式で表示
        with st.expander("📊 結果（JSON形式）", expanded=True):
            st.json(result.model_dump())

        # 整形表示
        with st.expander("📝 結果（整形表示）", expanded=True):
            self._display_formatted_result(result)

    def _display_formatted_result(self, result: BaseModel):
        """整形された結果を表示（デフォルト実装）"""
        # サブクラスでオーバーライド可能
        for field, value in result.model_dump().items():
            if isinstance(value, list):
                st.write(f"**{field}:**")
                for item in value:
                    st.write(f"- {item}")
            else:
                st.write(f"**{field}:** {value}")


# ==================================================
# 1. Step-by-Step（逐次展開型）
# ==================================================
class StepByStepResult(BaseModel):
    """Step-by-Stepパターンの結果"""
    question: str = Field(..., description="質問")
    steps: List[str] = Field(..., description="解決ステップ")
    answer: str = Field(..., description="最終的な答え")


class StepByStepPattern(BaseCoTPattern):
    """
    Step-by-Step（逐次展開型）パターン

    用途: 算数・アルゴリズム・レシピ等の手順型タスク
    """

    def __init__(self):
        super().__init__(CoTPatternType.STEP_BY_STEP)

    def get_system_prompt(self) -> str:
        return """You are a helpful tutor who thinks through problems step by step.
When given a question:
1. Break down the problem into clear, sequential steps
2. Number each step (Step 1:, Step 2:, etc.)
3. Show your work clearly
4. End with "Answer:" followed by the final answer

あなたは問題を段階的に考える親切なチューターです。
質問が与えられたら：
1. 問題を明確で順序立ったステップに分解してください
2. 各ステップに番号を付けてください（Step 1:, Step 2: など）
3. 作業を明確に示してください
4. 最後に "Answer:" に続けて最終的な答えを記載してください"""

    def get_result_model(self) -> BaseModel:
        return StepByStepResult

    def create_ui(self) -> Dict[str, Any]:
        col1, col2 = st.columns([3, 1])

        with col1:
            question = st.text_input(
                "質問を入力してください",
                value="2X + 1 = 5  Xはいくつ？",
                help="段階的に解決したい問題を入力"
            )

        with col2:
            temperature = st.slider(
                "Temperature",
                0.0, 1.0, 0.3, 0.05,
                help="低い値ほど一貫性のある回答"
            )

        return {"question": question, "temperature": temperature}

    def parse_response(self, response_text: str, inputs: Dict[str, Any]) -> StepByStepResult:
        steps = []
        answer = ""

        for line in response_text.splitlines():
            line = line.strip()
            if line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
            elif line and (line[0].isdigit() or line.lower().startswith("step")):
                steps.append(line)

        return StepByStepResult(
            question=inputs["question"],
            steps=steps,
            answer=answer
        )

    def _display_formatted_result(self, result: StepByStepResult):
        st.write(f"**質問:** {result.question}")
        st.write("**解決ステップ:**")
        for step in result.steps:
            st.write(f"  {step}")
        st.write(f"**答え:** {result.answer}")


# ==================================================
# 2. Hypothesis-Test（仮説検証型）
# ==================================================
class HypothesisTestResult(BaseModel):
    """Hypothesis-Testパターンの結果"""
    problem: str = Field(..., description="問題")
    hypothesis: str = Field(..., description="仮説")
    evidence: List[str] = Field(default_factory=list, description="証拠・実験")
    evaluation: str = Field(..., description="評価")
    conclusion: str = Field(..., description="結論")


class HypothesisTestPattern(BaseCoTPattern):
    """
    Hypothesis-Test（仮説検証型）パターン

    用途: バグ解析・科学実験・A/Bテスト
    """

    def __init__(self):
        super().__init__(CoTPatternType.HYPOTHESIS_TEST)

    def get_system_prompt(self) -> str:
        return """You are a senior QA engineer following a hypothesis-test methodology.
Given a problem and hypothesis, return a JSON object with these keys:
- "evidence": array of at least 3 concrete tests or measurements
- "evaluation": explanation of whether evidence supports/refutes the hypothesis
- "conclusion": short statement accepting or rejecting the hypothesis

Return ONLY valid JSON, no additional text.

あなたは仮説検証方法論に従う上級QAエンジニアです。
問題と仮説が与えられたら、以下のキーを持つJSONオブジェクトを返してください：
- "evidence": 少なくとも3つの具体的なテストまたは測定の配列
- "evaluation": 証拠が仮説を支持/反証するかの説明
- "conclusion": 仮説を受け入れるか拒否するかの短い声明

有効なJSONのみを返し、追加のテキストは含めないでください。"""

    def get_result_model(self) -> BaseModel:
        return HypothesisTestResult

    def create_ui(self) -> Dict[str, Any]:
        problem = st.text_area(
            "問題（バグ・実験目的）",
            value="モバイル版Webアプリの初回表示が遅い",
            height=80,
            help="解決したい問題や調査したい現象"
        )

        hypothesis = st.text_input(
            "仮説（原因・改善案）",
            value="画像サイズが大きすぎて帯域を圧迫している",
            help="問題の原因や解決策の仮説"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

        return {
            "problem"    : problem,
            "hypothesis" : hypothesis,
            "temperature": temperature
        }

    def _format_user_content(self, inputs: Dict[str, Any]) -> str:
        return f"Problem: {inputs['problem']}\nHypothesis: {inputs['hypothesis']}"

    def parse_response(self, response_text: str, inputs: Dict[str, Any]) -> HypothesisTestResult:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # JSONパースエラーの場合、簡易的なパース
            data = {
                "evidence"  : ["パースエラー: 手動で確認してください"],
                "evaluation": response_text,
                "conclusion": "不明"
            }

        return HypothesisTestResult(
            problem=inputs["problem"],
            hypothesis=inputs["hypothesis"],
            evidence=data.get("evidence", []),
            evaluation=data.get("evaluation", ""),
            conclusion=data.get("conclusion", "")
        )


# ==================================================
# 3. Tree-of-Thought（分岐探索型）
# ==================================================
class Branch(BaseModel):
    """思考の分岐"""
    state: str = Field(..., description="現在の状態")
    action: str = Field(..., description="取るべきアクション")
    score: Optional[float] = Field(None, description="評価スコア")


class TreeOfThoughtResult(BaseModel):
    """Tree-of-Thoughtパターンの結果"""
    goal: str = Field(..., description="目標")
    branches: List[Branch] = Field(default_factory=list, description="思考の分岐")
    best_path: Optional[List[int]] = Field(None, description="最適パスのインデックス")
    result: str = Field(..., description="最終結果")


class TreeOfThoughtPattern(BaseCoTPattern):
    """
    Tree-of-Thought（分岐探索型）パターン

    用途: パズル・最適化・プランニング・ゲームAI
    """

    def __init__(self):
        super().__init__(CoTPatternType.TREE_OF_THOUGHT)

    def get_system_prompt(self) -> str:
        return """You are an AI that performs Tree-of-Thoughts search.
Solve problems through branching reasoning steps.

For each problem:
1. Generate multiple candidate thoughts at each step
2. Evaluate each with a score (0-1)
3. Select the best path
4. Return a JSON with:
   - "branches": array of {state, action, score}
   - "best_path": array of selected indices
   - "result": final answer

Return ONLY valid JSON.

あなたはTree-of-Thoughts探索を実行するAIです。
分岐的な推論ステップで問題を解決します。

各問題に対して：
1. 各ステップで複数の候補思考を生成
2. それぞれを0-1のスコアで評価
3. 最適なパスを選択
4. 以下を含むJSONを返す：
   - "branches": {state, action, score}の配列
   - "best_path": 選択されたインデックスの配列
   - "result": 最終的な答え

有効なJSONのみを返してください。"""

    def get_result_model(self) -> BaseModel:
        return TreeOfThoughtResult

    def create_ui(self) -> Dict[str, Any]:
        goal = st.text_input(
            "目標（達成したいタスク）",
            value="4, 9, 10, 13 の数でGame of 24を解いてください",
            help="探索によって解決したい問題"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            num_branches = st.number_input(
                "分岐数/ステップ",
                min_value=2, max_value=6, value=3,
                help="各ステップでの候補数"
            )

        with col2:
            num_steps = st.number_input(
                "探索ステップ数",
                min_value=1, max_value=5, value=2,
                help="探索の深さ"
            )

        with col3:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.05)

        return {
            "goal"        : goal,
            "num_branches": num_branches,
            "num_steps"   : num_steps,
            "temperature" : temperature
        }

    def get_system_prompt(self) -> str:
        # 動的にプロンプトを生成する場合
        base_prompt = super().get_system_prompt()
        if hasattr(self, '_current_inputs'):
            base_prompt += f"\nUse exactly {self._current_inputs['num_branches']} branches and {self._current_inputs['num_steps']} steps."
        return base_prompt

    def _create_messages(self, inputs: Dict[str, Any]) -> List[ChatCompletionMessageParam]:
        # 入力を一時的に保存
        self._current_inputs = inputs
        return super()._create_messages(inputs)

    def parse_response(self, response_text: str, inputs: Dict[str, Any]) -> TreeOfThoughtResult:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = {
                "branches" : [],
                "best_path": [],
                "result"   : "JSONパースエラー"
            }

        branches = [Branch(**b) for b in data.get("branches", [])]

        return TreeOfThoughtResult(
            goal=inputs["goal"],
            branches=branches,
            best_path=data.get("best_path"),
            result=data.get("result", "")
        )


# ==================================================
# 4. Pros-Cons-Decision（賛否比較型）
# ==================================================
class ProsConsDecisionResult(BaseModel):
    """Pros-Cons-Decisionパターンの結果"""
    topic: str = Field(..., description="トピック")
    pros: List[str] = Field(default_factory=list, description="メリット")
    cons: List[str] = Field(default_factory=list, description="デメリット")
    decision: str = Field(..., description="決定")
    rationale: str = Field(..., description="根拠")


class ProsConsDecisionPattern(BaseCoTPattern):
    """
    Pros-Cons-Decision（賛否比較型）パターン

    用途: 技術選定・意思決定ドキュメント・企画提案
    """

    def __init__(self):
        super().__init__(CoTPatternType.PROS_CONS_DECISION)

    def get_system_prompt(self) -> str:
        return """You are a decision-making assistant.
Analyze topics by listing pros and cons, then make a reasoned decision.

Return a JSON object with:
- "pros": array of at least 3 advantages
- "cons": array of at least 3 disadvantages
- "decision": your recommendation
- "rationale": explanation for the decision

Be balanced and objective. Return ONLY valid JSON.

あなたは意思決定支援アシスタントです。
メリットとデメリットをリストアップしてトピックを分析し、理性的な決定を下します。

以下を含むJSONオブジェクトを返してください：
- "pros": 少なくとも3つの利点の配列
- "cons": 少なくとも3つの欠点の配列
- "decision": あなたの推奨
- "rationale": 決定の説明

バランスよく客観的に。有効なJSONのみを返してください。"""

    def get_result_model(self) -> BaseModel:
        return ProsConsDecisionResult

    def create_ui(self) -> Dict[str, Any]:
        topic = st.text_input(
            "意思決定したいトピック",
            value="リモートワークとオフィス出社、どちらが良い？",
            help="比較検討したいトピックや選択肢"
        )

        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.4, 0.05,
            help="高い値ほど創造的な回答"
        )

        return {"topic": topic, "temperature": temperature}

    def _format_user_content(self, inputs: Dict[str, Any]) -> str:
        return f"Topic: {inputs['topic']}"

    def parse_response(self, response_text: str, inputs: Dict[str, Any]) -> ProsConsDecisionResult:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = {
                "pros"     : ["パースエラー"],
                "cons"     : ["パースエラー"],
                "decision" : "不明",
                "rationale": response_text
            }

        return ProsConsDecisionResult(
            topic=inputs["topic"],
            pros=data.get("pros", []),
            cons=data.get("cons", []),
            decision=data.get("decision", ""),
            rationale=data.get("rationale", "")
        )

    def _display_formatted_result(self, result: ProsConsDecisionResult):
        st.write(f"**トピック:** {result.topic}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**👍 メリット:**")
            for pro in result.pros:
                st.write(f"- {pro}")

        with col2:
            st.write("**👎 デメリット:**")
            for con in result.cons:
                st.write(f"- {con}")

        st.write(f"**🎯 決定:** {result.decision}")
        st.write(f"**📝 根拠:** {result.rationale}")


# ==================================================
# 5. Plan-Execute-Reflect（反復改良型）
# ==================================================
class PlanExecuteReflectResult(BaseModel):
    """Plan-Execute-Reflectパターンの結果"""
    objective: str = Field(..., description="目標")
    plan: List[str] = Field(default_factory=list, description="計画")
    execution_log: List[str] = Field(default_factory=list, description="実行ログ")
    reflect: str = Field(..., description="振り返り")
    next_plan: List[str] = Field(default_factory=list, description="次の計画")


class PlanExecuteReflectPattern(BaseCoTPattern):
    """
    Plan-Execute-Reflect（反復改良型）パターン

    用途: 自律エージェント・長期プロジェクト管理
    """

    def __init__(self):
        super().__init__(CoTPatternType.PLAN_EXECUTE_REFLECT)

    def get_system_prompt(self) -> str:
        return """You are an AI implementing the Plan-Execute-Reflect loop.

Given an objective:
1. Plan: Create 3-5 concrete sequential steps
2. Execute: Simulate execution and log results
3. Reflect: Evaluate what worked and what didn't
4. Next Plan: Suggest 3 improved steps based on reflection

Return JSON with:
- "plan": array of initial steps
- "execution_log": array of simulated results
- "reflect": evaluation summary
- "next_plan": array of improved steps

Return ONLY valid JSON.

あなたはPlan-Execute-Reflectループを実装するAIです。

目標が与えられたら：
1. Plan: 3-5個の具体的で順序立ったステップを作成
2. Execute: 実行をシミュレートし結果を記録
3. Reflect: 何がうまくいき、何がうまくいかなかったかを評価
4. Next Plan: 振り返りに基づいて3つの改善されたステップを提案

以下を含むJSONを返してください：
- "plan": 初期ステップの配列
- "execution_log": シミュレートされた結果の配列
- "reflect": 評価の要約
- "next_plan": 改善されたステップの配列

有効なJSONのみを返してください。"""

    def get_result_model(self) -> BaseModel:
        return PlanExecuteReflectResult

    def create_ui(self) -> Dict[str, Any]:
        objective = st.text_input(
            "目標（達成したいこと）",
            value="3日以内にブログ記事を仕上げる",
            help="達成したい具体的な目標"
        )

        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.3, 0.05,
            help="低い値ほど現実的な計画"
        )

        return {"objective": objective, "temperature": temperature}

    def _format_user_content(self, inputs: Dict[str, Any]) -> str:
        return f"Objective: {inputs['objective']}"

    def parse_response(self, response_text: str, inputs: Dict[str, Any]) -> PlanExecuteReflectResult:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = {
                "plan"         : ["パースエラー"],
                "execution_log": [],
                "reflect"      : response_text,
                "next_plan"    : []
            }

        return PlanExecuteReflectResult(
            objective=inputs["objective"],
            plan=data.get("plan", []),
            execution_log=data.get("execution_log", []),
            reflect=data.get("reflect", ""),
            next_plan=data.get("next_plan", [])
        )


# ==================================================
# CoTパターンマネージャー
# ==================================================
class CoTPatternManager:
    """
    CoTパターンの統合管理クラス
    """

    def __init__(self):
        self.patterns: Dict[str, BaseCoTPattern] = {
            "Step-by-Step（逐次展開型）"        : StepByStepPattern(),
            "Hypothesis-Test（仮説検証型）"     : HypothesisTestPattern(),
            "Tree-of-Thought（分岐探索型）"     : TreeOfThoughtPattern(),
            "Pros-Cons-Decision（賛否比較型）"  : ProsConsDecisionPattern(),
            "Plan-Execute-Reflect（反復改良型）": PlanExecuteReflectPattern(),
        }
        self.client = OpenAI()

    def run(self):
        """アプリケーションの実行"""
        self._display_header()
        self._setup_sidebar()

        # 選択されたパターンの実行
        selected_pattern_name = st.session_state.get("selected_pattern")
        if selected_pattern_name and selected_pattern_name in self.patterns:
            pattern = self.patterns[selected_pattern_name]
            pattern.execute()

        self._display_footer()

    def _display_header(self):
        # ヘッダー表示
        st.title("🧠 Chain of Thought パターン学習ツール")
        st.markdown("""
        このツールでは、5種類の代表的なChain of Thought (CoT)パターンを
        実際に動かして学習できます。
        """)

    def _setup_sidebar(self):
        """サイドバーの設定"""
        with st.sidebar:
            st.header("設定")

            # モデル選択
            st.subheader("🤖 モデル選択")
            selected_model = st.selectbox(
                "使用するモデル",
                config.json_compatible_models,
                index=config.json_compatible_models.index(config.default_model),
                help="JSON出力に対応したモデルを選択"
            )
            st.session_state["selected_model"] = selected_model

            # パターン選択
            st.subheader("📚 パターン選択")
            selected_pattern = st.radio(
                "CoTパターンを選択",
                list(self.patterns.keys()),
                help="実行したいパターンを選択"
            )
            st.session_state["selected_pattern"] = selected_pattern

            # パターンの説明
            st.subheader("📖 パターン説明")
            pattern_descriptions = {
                "(1) Step-by-Step（逐次展開型）"        :
                    "問題を順序立てて段階的に解決。算数問題、アルゴリズム、レシピなどに最適。",
                "(2) Hypothesis-Test（仮説検証型）"     :
                    "仮説を立てて検証。バグ解析、科学実験、A/Bテストなどで使用。",
                "(3) Tree-of-Thought（分岐探索型）"     :
                    "複数の思考経路を探索。パズル、最適化、ゲームAIなどに適用。",
                "(4) Pros-Cons-Decision（賛否比較型）"  :
                    "メリット・デメリットを比較して決定。技術選定、意思決定に有効。",
                "(5) Plan-Execute-Reflect（反復改良型）":
                    "計画・実行・振り返りのループ。プロジェクト管理、自律エージェントに使用。"
            }

            if selected_pattern in pattern_descriptions:
                st.info(pattern_descriptions[selected_pattern])

            # リソース
            st.subheader("🔗 参考リソース")
            st.markdown("""
            - [信頼性向上テクニック](https://cookbook.openai.com/articles/techniques_to_improve_reliability)
            - [GPT-4.1プロンプトガイド](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)
            - [Reasoning モデルガイド](https://cookbook.openai.com/examples/reasoning_function_calls)
            """)

    def _display_footer(self):
        """フッター表示"""
        st.markdown("---")

        # 使用方法
        with st.expander("💡 使い方のヒント"):
            st.markdown("""
            ### 効果的な使い方

            1. **Step-by-Step**: 複雑な問題を小さなステップに分解したいとき
            2. **Hypothesis-Test**: 原因を特定したいとき、仮説を検証したいとき
            3. **Tree-of-Thought**: 複数の可能性を探索したいとき
            4. **Pros-Cons-Decision**: 選択肢を比較検討したいとき
            5. **Plan-Execute-Reflect**: 継続的な改善が必要なタスク

            ### Temperature設定
            - **0.0-0.3**: 一貫性のある論理的な回答
            - **0.4-0.6**: バランスの取れた回答
            - **0.7-1.0**: 創造的で多様な回答
            """)

        # デバッグ情報
        if st.checkbox("🐛 デバッグ情報を表示"):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("選択中のモデル", st.session_state.get("selected_model", "未選択"))
                st.metric("選択中のパターン", st.session_state.get("selected_pattern", "未選択"))

            with col2:
                st.metric("OpenAI API Key", "設定済み" if os.getenv("OPENAI_API_KEY") else "未設定")
                st.metric("セッション変数数", len(st.session_state))


# ==================================================
# ユーティリティ関数
# ==================================================
def check_environment():
    """環境チェック"""
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ 環境変数 OPENAI_API_KEY が設定されていません")
        st.info("""
        以下のコマンドで設定してください:
        ```bash
        export OPENAI_API_KEY='your-api-key'
        ```
        """)
        st.stop()


def display_examples():
    """使用例の表示"""
    with st.expander("📝 使用例"):
        st.markdown("""
        ### Step-by-Step の例
        **質問**: 「2X + 1 = 5 Xはいくつ？」
        **期待される出力**:
        - Step 1: 両辺から1を引く → 2X = 4
        - Step 2: 両辺を2で割る → X = 2
        - Answer: X = 2

        ### Hypothesis-Test の例
        **問題**: 「Webアプリが遅い」
        **仮説**: 「画像が大きすぎる」
        **期待される出力**:
        - Evidence: [画像サイズ測定, 通信量分析, 圧縮テスト]
        - Evaluation: 証拠は仮説を支持
        - Conclusion: 仮説を採択

        ### Tree-of-Thought の例
        **目標**: 「Game of 24を解く」
        **期待される出力**:
        - 複数の計算経路を探索
        - 各経路にスコア付け
        - 最適解を選択
        """)


# ==================================================
# カスタムパターンの追加例
# ==================================================
class CustomCoTPattern(BaseCoTPattern):
    """
    カスタムCoTパターンのテンプレート

    このクラスを参考に、独自のCoTパターンを作成できます
    """

    class CustomResult(BaseModel):
        """カスタム結果モデル"""
        input_data: str
        processed_data: str
        insights: List[str]

    def __init__(self):
        super().__init__(CoTPatternType.STEP_BY_STEP)  # 仮の型
        self.pattern_name = "Custom Pattern"

    def get_system_prompt(self) -> str:
        return "Your custom system prompt here..."

    def get_result_model(self) -> BaseModel:
        return self.CustomResult

    def create_ui(self) -> Dict[str, Any]:
        user_input = st.text_area("カスタム入力", height=100)
        return {"input": user_input}

    def parse_response(self, response_text: str, inputs: Dict[str, Any]) -> BaseModel:
        # カスタムパース処理
        return self.CustomResult(
            input_data=inputs["input"],
            processed_data=response_text,
            insights=["カスタム洞察1", "カスタム洞察2"]
        )


# ==================================================
# メイン実行
# ==================================================
def main():
    # メインアプリケーション
    # ページ設定
    st.set_page_config(
        page_title=config.page_title,
        page_icon=config.page_icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 環境チェック
    check_environment()

    # 使用例の表示（オプション）
    display_examples()

    # マネージャーの実行
    manager = CoTPatternManager()
    manager.run()


if __name__ == "__main__":
    main()

