from langchain_core.messages import HumanMessage, SystemMessage
from prompts import splitter_prompt, summarizer_prompt, synthesizer_prompt
from states import InputSchema, OutputSchema, SplitterSchema, State, SummarizerState
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from google import genai
import sys
import time

if len(sys.argv) != 2:
    print("Expected argument <path/to/pdf>")
    sys.exit(-1)

file_path: str = sys.argv[1]


client = genai.Client()
file = client.files.upload(file=file_path)
print("Processing file ...")
while file.state.name == "PROCESSING":
    time.sleep(2)
    file = client.files.get(name=file.name)


summarizer_model = ChatGoogleGenerativeAI(model="gemma-4-31b-it")
synthesizer_model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
splitter_model = synthesizer_model.with_structured_output(SplitterSchema)


def splitter(state: State):
    response = splitter_model.invoke(
        (
            [SystemMessage(content=splitter_prompt)]
            + [
                HumanMessage(
                    content=[
                        {
                            "type": "file",
                            "file_id": file.uri,
                            "mime_type": "application/pdf",
                        }
                    ]
                )
            ]
        )
    )

    return {"sections": response.sections}


def send_sections(state: State):
    return [
        Send("summarizer", {"section": section}) for section in state.get("sections")
    ]


def summarizer(state: SummarizerState):
    section = state.get("section")
    ctx = f"The section is {section}"
    response = summarizer_model.invoke(
        ([SystemMessage(content=summarizer_prompt)] + [HumanMessage(content=ctx)])
    )
    return {"summarized_sections": [response.text]}


def synthesizer(state: State):
    ctx = f"The summarized sections are :  {state.get('summarized_sections')}"
    response = synthesizer_model.invoke(
        ([SystemMessage(content=synthesizer_prompt)] + [HumanMessage(content=ctx)])
    )
    return {"merged_sections": response.text}


builder = StateGraph(State, input_schema=InputSchema, output_schema=OutputSchema)
builder.add_node("splitter", splitter)
builder.add_node("summarizer", summarizer)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "splitter")
builder.add_conditional_edges("splitter", send_sections)
builder.add_edge("summarizer", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()
print(graph.invoke({"file_path": file_path})["merged_sections"])
