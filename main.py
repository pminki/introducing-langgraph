import sqlite3

# LangGraph에서 상태 그래프를 만들기 위한 클래스와
# 그래프의 시작/끝을 나타내는 상수를 가져옵니다.
from langgraph.graph import StateGraph, START, END

# 대화 메시지(history)를 상태로 다룰 수 있게 해주는 기본 상태 타입입니다.
from langgraph.graph import MessagesState

# ToolNode, tools_condition:
# - ToolNode: LLM 이 \"도구를 써야 한다\"고 판단했을 때 실제로 도구를 실행해 주는 노드
# - tools_condition: LLM 응답 안에 도구 호출이 있는지 보고,
#                    다음에 ToolNode 로 갈지/그냥 넘어갈지 결정하는 조건 함수
from langgraph.prebuilt import ToolNode, tools_condition

# @tool 데코레이터로 파이썬 함수를 LLM 이 쓸 수 있는 \"도구\"로 등록할 때 사용합니다.
from langchain_core.tools import tool

# (필요시) OpenAI 채팅 모델을 초기화할 때 쓰는 함수입니다.
# 이 파일에서는 Google Vertex AI 를 사용하므로 직접 사용되지는 않습니다.
from langchain.chat_models import init_chat_model

# Google Cloud Vertex AI 의 Gemini 모델을 사용하기 위한 LangChain 래퍼입니다.
from langchain_google_vertexai import ChatVertexAI

# SqliteSaver: LangGraph의 상태(체크포인트)를
# SQLite 데이터베이스에 저장/복원해 주는 도구입니다.
from langgraph.checkpoint.sqlite import SqliteSaver

# interrupt: 그래프 실행을 잠시 멈추고, 사람(휴먼)의 입력을 기다리게 하는 기능입니다.
from langgraph.types import interrupt


@tool
def get_human_feedback(poem: str):
    """
    시(poem)에 대해 사람이 피드백을 줄 수 있게 하는 도구입니다.

    이 함수를 도구로 호출하면:
    - 현재까지 작성된 시를 사용자(휴먼)에게 보여주고
    - 사용자가 직접 피드백을 입력할 수 있도록 그래프 실행을 잠시 멈춥니다.
    """
    # interrupt(...) 를 호출하면 그래프 실행이 중단되고,
    # 외부(사람)에서 추가 정보를 입력해 줄 때까지 기다립니다.
    # 여기서는 시 내용을 함께 전달해서, 사람이 그 내용을 보고 피드백을 줄 수 있게 합니다.
    response = interrupt({"poem": poem})

    # 사람(휴먼)이 입력한 피드백은 response 안에 담겨서 돌아옵니다.
    # 이 예제에서는 response["feedback"] 부분만 꺼내서 반환합니다.
    return response["feedback"]


# LLM 이 사용할 수 있는 도구 목록입니다.
# 지금은 get_human_feedback 하나만 등록했습니다.
tools = [
    get_human_feedback,
]


# Vertex AI 의 Gemini 모델을 LLM 으로 사용합니다.
# - model_name: 사용할 Gemini 모델 이름
# - project: GCP 프로젝트 ID
# - location: 리전 (예: us-central1)
# - max_tokens: 한 번에 생성할 최대 토큰 수
llm = ChatVertexAI(
  model_name="gemini-2.5-flash-lite",
  project="ai-prompt-evaluator-489612",
  location="us-central1",
  max_tokens=500
)

# LLM 에게 \"너는 이런 도구들을 쓸 수 있어\" 라고 알려주는 단계입니다.
# 이렇게 bind_tools 를 해 두면, LLM 이 응답을 생성할 때
# 필요하다고 판단하면 get_human_feedback 도구를 호출하라는 요청을 만들 수 있습니다.
llm_with_tools = llm.bind_tools(tools)


class State(MessagesState):
    """
    이 그래프에서 사용할 상태(state)의 모양입니다.

    MessagesState 를 그대로 상속만 하면:
    - 기본적으로 \"messages\" 라는 키에 대화 내역이 들어갑니다.
    - 여기서는 추가 필드는 사용하지 않습니다.
    """

    pass


def chatbot(state: State) -> State:
    """
    시를 만들어 주는 챗봇 노드입니다.

    동작 흐름:
    1. 현재까지의 대화 내역(state[\"messages\"])을 기반으로 LLM 을 한 번 호출합니다.
    2. 프롬프트 안에서:
       - 시를 잘 쓰는 전문가 역할을 하라고 지시하고,
       - 반드시 get_human_feedback 도구를 먼저 사용해
         사람이 시를 확인하고 \"괜찮다\"고 할 때까지 최종 시를 내놓지 말라고 안내합니다.
    3. LLM 응답(메시지)을 다시 state 의 messages 에 추가해서 반환합니다.
    """
    response = llm_with_tools.invoke(
        f"""
        You are an expoert at making poems.
        You are given a topic and need to write a poem about it.
        Use the `get_human_feedback` tool to get feedback on your poem.
        Only after the user says the poem is ready, you should return the poem.

        Here is the conversation history:
        {state['messages']}
        """
    )

    # LangGraph 가 기존 messages 와 합칠 수 있도록,
    # 새로 받은 응답 메시지를 리스트에 담아 반환합니다.
    return {
        "messages": [response],
    }


# ToolNode 는 LLM 이 요청한 도구 호출을 실제로 실행해 주는 노드입니다.
tool_node = ToolNode(
    tools=tools,
)

# State 타입을 사용하는 상태 그래프 빌더를 생성합니다.
graph_builder = StateGraph(State)

# 그래프에 노드들을 등록합니다.
# - "chatbot": LLM 을 호출해서 시를 작성/수정하게 하는 노드
# - "tools"  : get_human_feedback 도구를 실제로 실행하는 노드
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# 실행 흐름을 정의합니다.
# START -> chatbot -> (필요하면 tools 로 분기) -> 다시 chatbot ... -> END
# 1) 먼저 chatbot 이 실행되어 시를 만들고, 도구 호출을 요청할 수 있습니다.
# 2) tools_condition 은 LLM 응답 안에 도구 호출 정보가 있는지 확인합니다.
#    - 있으면 "tools" 노드로 이동해서 실제로 도구를 실행합니다.
#    - 없으면 바로 다음 간선(여기서는 END) 로 진행합니다.
# 3) tools 노드에서 사람 피드백을 받은 뒤, 다시 chatbot 으로 돌아와
#    피드백을 반영한 최종 시를 만들 수 있습니다.
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)


# SQLite 데이터베이스 파일(memory-sync.db)에 연결합니다.
# - 이 파일 안에 그래프의 상태(체크포인트)가 저장됩니다.
conn = sqlite3.connect(
    "memory-sync.db",
    check_same_thread=False
)
memory = SqliteSaver(conn)


# graph_builder 를 실제 실행 가능한 그래프로 컴파일합니다.
# - name="mr_poet": 이 그래프에 붙이는 이름(디버깅/관리용)입니다.
graph = graph_builder.compile(name="mr_poet")


# 가상 환경 실행 명령어
# uv run lanngraph dev