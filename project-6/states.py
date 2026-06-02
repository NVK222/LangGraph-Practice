from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from operator import add


class InputSchema(TypedDict):
    file_path: str


class OutputSchema(TypedDict):
    merged_sections: str


class SplitterSchema(BaseModel):
    sections: list[str] = Field(
        description="A list of organized sections from the PDF. Maximum of 8 sections"
    )


class SummarizerState(TypedDict):
    section: str


class State(TypedDict):
    merged_sections: str
    sections: list[str]
    summarized_sections: Annotated[list[str], add]
