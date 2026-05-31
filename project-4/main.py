import sys
from operator import add
from typing import Annotated, TypedDict
import uuid
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

if len(sys.argv) != 2:
    print("Expected argument <destination>")
    sys.exit(-1)

userInput: str = sys.argv[1]

CREATOR_PROMPT = """You are an expert itinerary creator.
Your task is to create a 3 day vacation plan for the user specified destination.
Choose places that are fun, and exciting.
Do not recommend anything dangerous.
Return only the itinerary, nothing else.
Return in plaintext only.
"""

EDITOR_PROMPT = """You are an expert itinerary creator.
Your task is to look at the user's destination, the itinerary created for it, and edit the itinerary according to the user's feedback.
The itinerary should only be 3 days long.
While editing the itinerary pay close to attention to the user's past feedback if any, to make sure you don't make anything worse.
Return only the itinerary, nothing else,
Return in plaintext only.
"""

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")


class InputSchema(TypedDict):
    destination: str


class OutputSchema(TypedDict):
    itinerary: str


class State(TypedDict):
    destination: str
    itinerary: str
    feedback: Annotated[list[str], add]


def creator(state: State) -> State:
    context = f"The destination is :  {state.get('destination')}"
    response = model.invoke(([SystemMessage(content=CREATOR_PROMPT)] + [context]))

    return {"itinerary": response.text}


def editor(state: State) -> State:
    context = f"""The destination is : {state.get("destination")}
    The itinerary is {state.get("itinerary")}
    The feedback is {state.get("feedback")}"""

    response = model.invoke(([SystemMessage(content=EDITOR_PROMPT)] + [context]))

    return {"itinerary": response.text}


def get_feedback(state: State):
    feedback = interrupt(
        "Would you like to modify the itinerary? Type DONE to confirm."
    )
    return {"feedback": [feedback]}


def should_continue(state: State):
    last_feedback = state.get("feedback", [""])[-1]
    if "DONE" in last_feedback.upper():
        return END
    return "Editor"


builder = StateGraph(State, input_schema=InputSchema, output_schema=OutputSchema)

builder.add_node("Creator", creator)
builder.add_node("Editor", editor)
builder.add_node("Feedback", get_feedback)

builder.add_edge(START, "Creator")
builder.add_edge("Creator", "Feedback")
builder.add_conditional_edges("Feedback", should_continue, [END, "Editor"])
builder.add_edge("Editor", "Feedback")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": uuid.uuid4()}}

graph.invoke({"destination": userInput}, config)
while True:
    state = graph.get_state(config)
    if state.tasks:
        print(f"The current itinerary is \n\n{state.values.get('itinerary')}\n\n")
        print(state.tasks[0].interrupts[0].value)
        feedback = input("> ")
        graph.invoke(Command(resume=feedback), config)
    else:
        break
