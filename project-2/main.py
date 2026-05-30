import sys
from typing import TypedDict
from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate

if len(sys.argv) != 2:
    print("Expected arguments <inquiry>")
    sys.exit(-1)

userInput: str = sys.argv[1]

Model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

ROUTER_PROMPT = """You are a customer support expert.
Your task is to read the message from the user, understand their intent and classify their issue/statement into one of 3 categories: Billing, Technical Support or General Inquiry.
Return the classified category.
"""

BILLING_PROMPT = """You are a customer support agent, expert in Billing for the company.
Your task is to read and understand the user's inquiry, and respond to them accordingly.If given any feedback, correct the response according to the feedback.
Query: {query}
Feedback: {feedback}
When responding, be polite and formal.
DO NOT GIVE OUT ANY CONFIDENTIAL INFORMATION ABOUT THE COMPANY
"""

TECHNICAL_PROMPT = """You are a customer support agent, expert in Technical side of things for the company.
Your task is to read and understand the user's inquiry, and respond to them accordingly. If given any feedback, correct the response according to the feedback.
Query: {query}
Feedback: {feedback}
When responding, be polite and formal.
DO NOT GIVE OUT ANY CONFIDENTIAL INFORMATION ABOUT THE COMPANY
"""

GENERAL_PROMPT = """You are a customer support agent, who understands the overall working of the company.
Your task is to read and understand the user's inquiry, and respond to them accordingly.If given any feedback, correct the response according to the feedback.
Query: {query}
Feedback: {feedback}
When responding, be polite and formal.
DO NOT GIVE OUT ANY CONFIDENTIAL INFORMATION ABOUT THE COMPANY
"""

AUDITOR_PROMPT = """You are an expert auditor.
Your job is to check whether the response given by an agent in reponse to a user's query is valid or not.
When checking the response, give attention to whether the response is relevant, polite, and does not give out any unnecessary or confidential information.
If the response is valid, return PASS. Otherwise return feedback.
Response: {response}
"""


class InputState(TypedDict):
    user_input: str


class OutputState(TypedDict):
    response: str


class State(TypedDict):
    user_input: str
    node: str
    response: str | None
    feedback: str | None


def Router(state: InputState) -> State:
    response = Model.invoke(
        ([SystemMessage(content=ROUTER_PROMPT)] + state.get("user_input"))
    )
    return {"user_input": state.get("user_input"), "node": response.text.upper()}


def Billing(state: State) -> State:
    prompt = PromptTemplate.from_template(BILLING_PROMPT).format(
        query=state.get("user_input"), feedback=state.get("feedback")
    )
    response = Model.invoke(([SystemMessage(content=prompt)] + state.get("user_input")))
    return {
        "response": response.text,
        "feedback": "",
    }


def Technical(state: State) -> State:
    prompt = PromptTemplate.from_template(TECHNICAL_PROMPT).format(
        query=state.get("user_input"), feedback=state.get("feedback")
    )
    response = Model.invoke(([SystemMessage(content=prompt)] + state.get("user_input")))
    return {
        "response": response.text,
        "feedback": "",
    }


def General(state: State) -> State:
    prompt = PromptTemplate.from_template(GENERAL_PROMPT).format(
        query=state.get("user_input"), feedback=state.get("feedback")
    )
    response = Model.invoke(([SystemMessage(content=prompt)] + state.get("user_input")))
    return {
        "response": response.text,
        "feedback": "",
    }


def Auditor(state: State) -> State | OutputState:
    expert_response = state.get("response")
    response = Model.invoke(
        ([SystemMessage(content=AUDITOR_PROMPT)] + [expert_response])
    )
    return {
        "feedback": response.text,
    }


def RoutingNode(state: State):
    node = state.get("node")
    if "billing" in node.lower():
        return "BILLING"
    elif "technical" in node.lower():
        return "TECHNICAL"
    return "GENERAL"


def ShouldContinue(state: State):
    feedback = state.get("feedback")
    if feedback == "PASS":
        return END
    return state.get("node")


graph = StateGraph(State, input_schema=InputState, output_schema=OutputState)

graph.add_node("ROUTER", Router)
graph.add_node("BILLING", Billing)
graph.add_node("TECHNICAL", Technical)
graph.add_node("GENERAL", General)
graph.add_node("AUDITOR", Auditor)

graph.add_edge(START, "ROUTER")
graph.add_conditional_edges("ROUTER", RoutingNode, ["TECHNICAL", "BILLING", "GENERAL"])
graph.add_edge("BILLING", "AUDITOR")
graph.add_edge("TECHNICAL", "AUDITOR")
graph.add_edge("GENERAL", "AUDITOR")
graph.add_conditional_edges(
    "AUDITOR", ShouldContinue, ["TECHNICAL", "BILLING", "GENERAL", END]
)

GRAPH = graph.compile()
message = [HumanMessage(content=userInput)]
print(GRAPH.invoke({"user_input": message})["response"])
