# streamlit run a20_05_chain_of_thought.py --server.port=8505
#
# ─ ５種の代表的 CoT パターン
# [Menu] ------------------------------------
# 1. Step-by-Step（逐次展開型）
# 2. Hypothesis-Test（仮説検証型）
# 3. Tree-of-Thought（分岐探索型）
# 4. Pros-Cons-Decision（賛否比較型）
# 5. Plan-Execute-Reflect（反復改良型）
# ------------------------------------------
# 題名（日本語）	            題名（英語）	URL
# ------------------------------------------
# 信頼性を向上させるテクニック	                Techniques to improve reliability	https://cookbook.openai.com/articles/techniques_to_improve_reliability
# GPT-4.1 プロンプト活用ガイド	G               PT-4.1 Prompting Guide	https://cookbook.openai.com/examples/gpt4-1_prompting_guide
# Reasoningモデルでのファンクションコールの扱い方	Handling Function Calls with Reasoning Models	https://cookbook.openai.com/examples/reasoning_function_calls
# o3/o4-mini 関数呼び出しガイド	            o3/o4-mini Function Calling Guide	https://cookbook.openai.com/examples/o-series/o3o4-mini_prompting_guide
# Responses-APIを用いたReasoningモデルの性能向上	Better performance from reasoning models using the Responses API	https://cookbook.openai.com/examples/responses_api/reasoning_items
# 大規模言語モデルの活用方法	                How to work with large language models	https://cookbook.openai.com/articles/how_to_work_with_large_language_models
# 検索APIとリランキングによる質問応答	        Question answering using a search API and re-ranking	https://cookbook.openai.com/examples/question_answering_using_a_search_api
# LangChainでツール利用エージェントを構築する方法	How to build a tool-using agent with LangChain	https://cookbook.openai.com/examples/how_to_build_a_tool-using_agent_with_langchain
# Web 上の関連リソースまとめ	                Related resources from around the web	https://cookbook.openai.com/articles/related_resources
# Reasoning を活用したルーチン生成	            Using reasoning for routine generation	https://cookbook.openai.com/examples/o1/using_reasoning_for_routine_generation
# ------------------------------------------
import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from a0_common_helper.helper import (
    init_page,
    init_messages,
    select_model,
    sanitize_key,
    get_default_messages,
    extract_text_from_response, append_user_message,
)

from typing import List
from pydantic import BaseModel
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,  # ← union 型
)

import streamlit as st
st.set_page_config(
    page_title="ChatGPT API CoT",
    page_icon="2025-5 Nakashima"
)

# ------------------------------------------
# 1. Step-by-Step（逐次展開型）
#    用途: 算数・アルゴリズム・レシピ等の手順型タスク
# ------------------------------------------
class StepByStep(BaseModel):
    question: str
    steps: List[str]
    answer: str

def step_by_step(demo_name: str = "step_by_step") -> StepByStep:
    # Streamlit UI + ChatCompletion で逐次展開型 CoT を実演
    st.subheader("Step-by-Step（逐次展開型）デモ")

    # --- ユーザー入力 ---
    default_q = "2X + 1 = 5  Xはいくつ？"
    question = st.text_input("質問を入力してください", value=default_q)

    # --- オプション ---
    temperature = st.slider("temperature", 0.0, 1.0, 0.3, 0.05)

    # --- 実行 ---
    run = st.button("実行")
    if not run:
        st.stop()

    system_prompt = (
        "You are a helpful tutor.  "
        "Think through the user's question step by step.  "
        "Output each step on its own line prefixed with 'Step X:'.  "
        "After the final step, output 'Answer:' followed by the final answer only."
        "あなたは親切なチューターです。"
        "ユーザーの質問を段階的に考えてください。"
        "各段階を ‘Step X:’ の接頭辞を付けて、それぞれ新しい行に出力してください。"
        "最後の段階の後に ‘Answer:’ を出力し、その後に最終的な答えのみを記載してください。"
    )

    # messages => ChatCompletionMessageParam
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=question),
    ]

    # 選択済みモデルを session_state から取得（無ければ fallback）
    model = st.session_state.get("selected_model", select_model(demo_name))

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    content = resp.choices[0].message.content.strip()
    # --- ざっくりパース ---
    steps, answer = [], ""
    for line in content.splitlines():
        if line.lower().startswith("answer"):
            answer = line.split(":", 1)[-1].strip()
        elif line:
            steps.append(line.strip())

    result = StepByStep(question=question, steps=steps, answer=answer)
    st.json(result.model_dump(), expanded=False)
    return result

# ------------------------------------------
# 2. Hypothesis-Test（仮説検証型）
#    用途: バグ解析・科学実験・A/B テスト
# ------------------------------------------
import json

class HypothesisTest(BaseModel):
    problem: str
    hypothesis: str
    evidence: List[str] | None = None
    evaluation: str | None = None
    conclusion: str

def hypothesis_test(
    demo_name: str = "hypothesis_test",
) -> HypothesisTest:
    """Streamlit UI + ChatCompletion で仮説検証型 CoT を実演"""
    st.subheader("Hypothesis-Test（仮説検証型）デモ")

    # --- ユーザー入力 ---
    problem = st.text_area(
        "Problem（バグ・実験目的）を入力してください",
        value="モバイル版 Web アプリの初回表示が遅い",
    )
    hypothesis = st.text_input(
        "Hypothesis（原因/改善案）を入力してください",
        value="画像サイズが大き過ぎて帯域を圧迫している",
    )
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)

    # --- 実行 ---
    if not st.button("実行"):
        st.stop()

    # --- プロンプト設計 ---
    system_prompt = (
        "You are a senior QA engineer following a hypothesis-test loop. "
        "Return ONLY a JSON object with the keys: "
        "`evidence` (array of strings), `evaluation` (string), `conclusion` (string). "
        "• Evidence: list at least 3 concrete measurements or experiments you would run. "
        "• Evaluation: explain whether the evidence supports or refutes the hypothesis. "
        "• Conclusion: a short statement saying the hypothesis is accepted or rejected."
        "あなたは仮説―検証ループを実践する上級 QA エンジニアです。"
        "次のキーを持つ JSON オブジェクト のみ を返してください:"
        "evidence（文字列の配列）、evaluation（文字列）、conclusion（文字列）。"
        "・Evidence: 実施する具体的な測定や実験を 3 つ以上列挙してください。"
        "・Evaluation: その証拠が仮説を支持するか反証するかを説明してください。"
        "・Conclusion: 仮説を採択するか棄却するかを簡潔に述べてください。"
    )

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Problem: {problem}\nHypothesis: {hypothesis}",
        ),
    ]

    model = st.session_state["selected_model"]  # main() で選択済み
    # 正しい型オブジェクトを生成して渡す
    resp = client.chat.completions.create(
        model=model,  # 例: "gpt-4o" / "gpt-3.5-turbo-1106"
        messages=messages,
        temperature=temperature,
        # response_format=ResponseFormatJSONObject(),  # ← dict ではなく型インスタンス
    )

    # --- JSON をパースして Pydantic に流し込む ---
    data = json.loads(resp.choices[0].message.content)
    result = HypothesisTest(
        problem=problem,
        hypothesis=hypothesis,
        evidence=data.get("evidence"),
        evaluation=data.get("evaluation"),
        conclusion=data.get("conclusion"),
    )

    st.json(result.model_dump(), expanded=False)
    return result

# ------------------------------------------
# 3. Tree-of-Thought（分岐探索型）
#    用途: パズル・最適化・プランニング・ゲーム AI
# ------------------------------------------
class Branch(BaseModel):
    state: str
    action: str
    score: float | None = None

class TreeOfThought(BaseModel):
    goal: str
    branches: List[Branch]
    best_path: List[int] | None = None
    result: str

def tree_of_thought(demo_name: str = "tree_of_thought") -> TreeOfThought:
    st.subheader("Tree‑of‑Thought（分岐探索型）デモ")

    goal = st.text_input("Goal（達成したいタスク）を入力してください",
                         value="4, 9, 10, 13 の数で Game of 24 を解いてください。")
    temperature = st.slider("temperature", 0.0, 1.0, 0.5, 0.05)
    num_branches = st.number_input("各ステップの分岐数", min_value=2, max_value=6, value=3, step=1)
    num_steps = st.number_input("探索ステップ数", min_value=1, max_value=5, value=2, step=1)

    if not st.button("探索開始"):
        st.stop()

    system_prompt = (
        "You are an assistant that performs Tree-of-Thoughts search. "
        "We want to solve a problem through branching reasoning steps. "
        "At each step, generate up to `num_branches` candidate thoughts, each with state and action. "
        "Then evaluate them with scores. "
        "After `num_steps` steps, return a JSON with keys:\n"
        "- branches: array of objects {state,action,score}\n"
        "- best_path: array of selected branch indices (0-based)\n"
        "- result: the final answer.\n"
        f"Use exactly num_branches={num_branches} and num_steps={num_steps}.\n"
        "Return only valid JSON without extra text."
        "あなたはTree-of-Thoughts探索を実行するアシスタントです。"
        "問題を分岐的な推論ステップで解決します。"
        "各ステップでは、状態とアクションを含む最大 num_branches 個の候補思考を生成してください。"
        "次にそれらをスコアで評価します。"
        "num_steps ステップが完了したら、以下のキーを持つ JSON を返してください:"
        "- branches: オブジェクト {state,action,score} の配列"
        "- best_path: 選択したブランチのインデックス (0 始まり) の配列"
        "- result: 最終的な答え"
        "必ず num_branches={num_branches} と num_steps={num_steps} を使用してください。"
        "余計なテキストを含めず、有効な JSON のみを返してください。"
    )

    messages: List[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=f"Goal: {goal}")
    ]

    resp = client.chat.completions.create(
        model=st.session_state["selected_model"],
        messages=messages,
        temperature=temperature,
    )
    text = resp.choices[0].message.content.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        st.error("JSONパースエラー。レスポンス内容を確認してください。")
        st.code(text, language="json")
        st.stop()

    # Pydantic にパース
    result = TreeOfThought(
        goal=goal,
        branches=[Branch(**b) for b in data.get("branches",[])],
        best_path=data.get("best_path"),
        result=data.get("result",""),
    )
    st.json(result.model_dump(), expanded=True)
    return result

# ------------------------------------------
# 4. Pros-Cons-Decision（賛否比較型）
#    用途: 技術選定・意思決定ドキュメント・企画提案
#    賛否比較 → 意思決定 → 根拠提示
# ------------------------------------------
class ProsConsDecision(BaseModel):
    topic: str
    pros: List[str] | None = None
    cons: List[str] | None = None
    decision: str
    rationale: str | None = None

def pros_cons_decision(demo_name: str = "pros_cons_decision") -> ProsConsDecision:
    st.subheader("Pros‑Cons‑Decision（賛否比較型）デモ")

    topic = st.text_input(
        "意思決定したいトピックを入力してください",
        value="リモートワークとオフィス出社、どちらが良い？"
    )
    temperature = st.slider("temperature", 0.0, 1.0, 0.4, 0.05)

    if not st.button("実行"):
        st.stop()

    system_prompt = (
        "You are a decision-making assistant. "
        "Given a topic, list its Pros and Cons, then make a decision with rationale. "
        "Return only valid JSON with keys: pros (string array), cons (string array), "
        "decision (string), rationale (string)."
        "Have at least 3 items in each of pros and cons."
        "あなたは意思決定支援アシスタントです。"
        "トピックが与えられたら、その長所と短所を列挙し、根拠を添えて決定を下してください。"
        "次のキーを持つ有効な JSON のみを返してください: pros（文字列配列）、cons（文字列配列）、"
        "decision（文字列）、rationale（文字列）。"
        "pros と cons にはそれぞれ少なくとも 3 項目を含めてください。"
    )

    messages: List[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=f"Topic: {topic}")
    ]

    resp = client.chat.completions.create(
        model=st.session_state["selected_model"],
        messages=messages,
        temperature=temperature,
    )
    text = resp.choices[0].message.content.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        st.error("JSONパースエラー。AI 出力内容を確認してください。")
        st.code(text, language="json")
        st.stop()

    result = ProsConsDecision(
        topic=topic,
        pros=data.get("pros"),
        cons=data.get("cons"),
        decision=data.get("decision", ""),
        rationale=data.get("rationale")
    )

    st.json(result.model_dump(), expanded=True)
    return result

# ------------------------------------------
# 5. Plan-Execute-Reflect（反復改良型）
#    用途: 自律エージェント・長期プロジェクト管理
#    システムプロンプトで
#    ・Plan
#    ・Execute
#    ・Reflect
#    ・Next_plan
#    役割を明確に振ることで、Chain of Thoughtの流れが整います
# ------------------------------------------
class PlanExecuteReflect(BaseModel):
    objective: str
    plan: List[str]
    execution_log: List[str] | None = None
    reflect: str
    next_plan: List[str] | None = None

def plan_execute_reflect(demo_name: str = "plan_execute_reflect") -> PlanExecuteReflect:
    st.subheader("Plan‑Execute‑Reflect（反復改良型）デモ")

    objective = st.text_input(
        "Objective（達成したい目標）を入力してください",
        value="3日以内にブログ記事を仕上げる"
    )
    temperature = st.slider("temperature", 0.0, 1.0, 0.3, 0.05)

    if not st.button("開始"):
        st.stop()

    system_prompt = (
        "You are an AI assistant implementing the Plan-Execute-Reflect loop:\n"
        "• Plan: propose 3–5 concrete sequential steps to achieve the objective.\n"
        "• Execute: simulate execution and summarize as execution_log.\n"
        "• Reflect: evaluate what worked and what didn’t.\n"
        "• Next_plan: suggest 3 improved steps based on reflection.\n"
        "Return only valid JSON containing keys: plan (array of strings), "
        "execution_log (array of strings), reflect (string), next_plan (array of strings)."
        "あなたはPlan-Execute-Reflectループを実装するAIアシスタントです。"
        "・Plan: 目標を達成するために3〜5個の具体的で順序立った手順を提案してください。"
        "・Execute: 実行をシミュレートし、execution_logとして要約してください。"
        "・Reflect: 成功した点と失敗した点を評価してください。"
        "・Next_plan: Reflectionに基づき、改善された3つの手順を提案してください。"
        "以下のキーを含む有効なJSONのみを返してください: plan（文字列配列）、execution_log（文字列配列）、reflect（文字列）、next_plan（文字列配列）。"
    )
    messages: List[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=f"Objective: {objective}")
    ]

    resp = client.chat.completions.create(
        model=st.session_state["selected_model"],
        messages=messages,
        temperature=temperature,
    )
    text = resp.choices[0].message.content.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        st.error("JSONパースエラー。AI 出力内容を確認してください。")
        st.code(text, language="json")
        st.stop()

    result = PlanExecuteReflect(
        objective=objective,
        plan=data.get("plan", []),
        execution_log=data.get("execution_log"),
        reflect=data.get("reflect", ""),
        next_plan=data.get("next_plan"),
    )

    st.json(result.model_dump(), expanded=True)
    return result

# ------------------------------------------
# OpenAI helper & 共通呼び出し関数
# ------------------------------------------
client = OpenAI()  # OPENAI_API_KEY を環境変数に設定

def main():
    sample_funcs = {
        "step:Step-by-Step（逐次展開型）": step_by_step,
        "hypo:Hypothesis-Test（仮説検証型）": hypothesis_test,
        "treeTree-of-Thought（分岐探索型）": tree_of_thought,
        "pros:Pros-Cons-Decision（賛否比較型）": pros_cons_decision,
        "plan:Plan-Execute-Reflect（反復改良型）": plan_execute_reflect,
    }

    demo_name = st.sidebar.radio("デモを選択", list(sample_funcs.keys()))
    st.session_state.current_demo = demo_name

    # モデル選択はここだけ
    st.write("jsonモード対応モデルは：gpt-4o, gpt-4o-mini, gpt-4-1106-preview, gpt-3.5-turbo-1106")
    # if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = select_model(demo_name)
    model = st.session_state["selected_model"]
    st.write("選択したモデル:", model)

    # 下位関数へモデルを渡す
    sample_funcs[demo_name](demo_name)

# ------------------------------------------
# CLI テスト（逐次展開型の例）
# ------------------------------------------
if __name__ == "__main__":
    main()

# streamlit run a20_05_chain_of_thought.py --server.port=8505
