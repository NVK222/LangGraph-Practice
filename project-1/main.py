import sys
from langchain.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import MessagesState
from langgraph.graph import START, END, StateGraph
from langchain.messages import HumanMessage

if len(sys.argv) != 2:
    print("Expected argument <input>")
    sys.exit(-1)

userInput: str = sys.argv[1]


@tool
def add(x: int, y: int) -> int:
    """Add 'x' and 'y'

    Args:
        x: First input
        y: Second input
    """

    return x + y


@tool
def multiply(x: int, y: int) -> int:
    """Multiply 'x' and 'y'

    Args:
        x: First input
        y: Second input
    """

    return x * y


@tool
def percentage(x: int, y: float) -> float:
    """Calculates 'y' percentage of 'x'

    Args:
        x: Number to calculate percentage of
        y: Percentage to be calculated in range 0 - 1 inclusive
    """

    if y > 1:
        y *= 100
    return x * y / 100


tools = [add, multiply, percentage]
tools_by_name = {tool.name: tool for tool in tools}

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
model_with_tools = model.bind_tools(tools)


class State(MessagesState):
    llm_calls: int


def router(state: State):
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state.get("messages")
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: State):
    """Performs a tool call"""

    result = []
    messages = state.get("messages")[-1]
    if not isinstance(messages, AIMessage):
        return
    for tool_call in messages.tool_calls:
        tool = tools_by_name.get(tool_call["name"], add)
        ret = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=ret, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: State):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    message = state.get("messages")[-1]
    if not isinstance(message, AIMessage):
        return
    if message.tool_calls:
        return "tool_node"
    return END


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("router", router)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "router")
agent_builder.add_conditional_edges("router", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "router")

agent = agent_builder.compile()

messages = [HumanMessage(content=userInput)]
messages = agent.invoke({"messages": messages})

for m in messages["messages"]:
    m.pretty_print()
