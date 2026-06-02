from dataclasses import dataclass
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from prompts import coach_prompt, saver_prompt, summarizer_prompt

load_dotenv()

DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}?sslmode=disable"
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")


@dataclass
class Context:
    user_id: str


class State(MessagesState):
    long_term_ctx: list[str]
    summary: str


def initializer(state: State, runtime: Runtime[Context]):
    facts = runtime.store.get(namespace, user_id)
    facts = facts.value.get("facts", "") if facts else ""
    return {"long_term_ctx": facts}


def coach(state: State):
    long_term_ctx = state.get("long_term_ctx", [""])
    summary = state.get("summary", "")
    short_term_ctx = state.get("messages")[-10:]

    messages = []

    for message in short_term_ctx:
        if isinstance(message, HumanMessage):
            messages.append(message)

    prompt = PromptTemplate.from_template(coach_prompt).format(facts=long_term_ctx)
    ctx = f"""The summary of the messages so far:  {summary}. 
    The 5 most recent messages so far:  {messages}
    """

    return {"messages": model.invoke(([SystemMessage(content=prompt)] + [ctx]))}


def summarizer(state: State):
    messages = state.get("messages")[:10]
    ctx = f"The summary is:  {messages}"

    response = model.invoke(([SystemMessage(content=summarizer_prompt)] + [ctx]))
    delete = [RemoveMessage(msg.id) for msg in messages]

    return {"summary": response.text, "messages": delete}


def save(state: dict[str], store: BaseStore):
    long_term_ctx = state.get("long_term_ctx", "")
    messages = state.get("messages", [""])[:10]
    summary = state.get("summary", "")

    ctx = f"""Existing facts:  {long_term_ctx}
    Summary of messages:  {summary}
    Rest of the messages:  {messages}
    """

    response = model.invoke(([SystemMessage(content=saver_prompt)] + [ctx]))

    store.put(namespace, user_id, {"facts": response.text})

    return {}


def should_continue(state: State):
    messages = state.get("messages", [""])

    if len(messages) >= 10:
        return "summarizer"

    return END


builder = StateGraph(State, context_schema=Context)
builder.add_node("initializer", initializer)
builder.add_node("coach", coach)
builder.add_node("summarizer", summarizer)
builder.add_edge(START, "initializer")
builder.add_edge("initializer", "coach")
builder.add_conditional_edges("coach", should_continue)
builder.add_edge("summarizer", END)

user_id = "1"
config = {"configurable": {"thread_id": user_id}}
namespace = (user_id, "facts")
checkpointer = InMemorySaver()

with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()
    graph = builder.compile(checkpointer=checkpointer, store=store)

    flag = True
    while flag:
        user_input = input("> ")
        if "DONE" in user_input.strip().upper():
            state = graph.get_state(config).values
            _store = graph.store
            save(state, _store)
            break

        events = graph.stream(
            {"messages": HumanMessage(content=user_input)},
            config,
            durability="exit",
            stream_mode="updates",
        )
        for event in events:
            if "coach" in event:
                print(
                    f"\nCoach:  {event['coach']['messages'].content[0].get('text', '')}\n"
                )
