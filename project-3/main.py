import sys
from typing import TypedDict
from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

if len(sys.argv) != 2:
    print("Expected argument <topic>")
    sys.exit(-1)

userInput: str = sys.argv[1]

INPUT_WRITER_PROMPT = """You are an award winning writer.
Your job is to get the topic from a user, and write a short 2 paragraph essay on it.
"""

WRITER_PROMPT = """You are an award winning writer.
Your job is to rewrite the essay, correcting the critic, for a given topic. Return only the corrected essay. Do not return any conversational text.
"""

CRITIC_PROMPT = """You are a professional critic.
Your job is to read the topic, and a short 2 paragraph essay written on the topic by a professional writer, and write critics on it.
You have to be very harsh. The writer you are critiquing is world famous, they need to have a standard.
If there are critics, return them as a list of strings. If there are no critics and you are satisfied, leave the critic as empty and return done as True
"""


class InputSchema(TypedDict):
    user_input: str


class OutputSchema(TypedDict):
    essay: str


class State(TypedDict):
    user_input: str
    essay: str
    critic: list[str]
    rounds: int
    done: bool


class CriticOutput(BaseModel):
    critic: list[str] = Field(
        description="A list of critques. Leave empty if statisfied"
    )
    done: bool = Field(description="True if satisfied with the essay. False otherwise")


model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
critic_model = model.with_structured_output(CriticOutput)


def InputWriter(state: InputSchema) -> State:
    response = model.invoke(
        (
            [SystemMessage(content=INPUT_WRITER_PROMPT)]
            + [f"The topic is:  {state.get('user_input')}"]
        )
    )

    return {"user_input": state.get("user_input"), "essay": response.text}


def Writer(state: State) -> State:
    essay = state.get("essay", "")
    critic = state.get("critic", [""])
    topic = state.get("user_input")
    response = model.invoke(
        (
            [SystemMessage(content=WRITER_PROMPT)]
            + [
                f"""The topic is: {topic}
        The essay is:  {essay}
        The critics are: {critic}"""
            ]
        )
    )

    return {
        "essay": response.text,
    }


def Critic(state: State) -> State:
    essay = state.get("essay")
    topic = state.get("user_input")

    response = critic_model.invoke(
        [SystemMessage(content=CRITIC_PROMPT)]
        + [
            f"""The topic is:  {topic}
    The essay is: {essay}"""
        ]
    )

    return {
        "critic": response.critic,
        "rounds": state.get("rounds", 0) + 1,
        "done": response.done,
    }


def should_continue(state: State):
    if state.get("rounds", 0) >= 3 or state.get("done", False):
        return END
    return "WRITER"


graph = StateGraph(State, input_schema=InputSchema, output_schema=OutputSchema)
graph.add_node("WRITER", Writer)
graph.add_node("CRITIC", Critic)
graph.add_node("INPUT", InputWriter)

graph.add_edge(START, "INPUT")
graph.add_edge("INPUT", "CRITIC")
graph.add_edge("WRITER", "CRITIC")
graph.add_conditional_edges("CRITIC", should_continue, [END, "WRITER"])

GRAPH = graph.compile()

print(GRAPH.invoke({"user_input": userInput})["essay"])
