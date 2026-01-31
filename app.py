# app.py

import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict

# LangChain / LangGraph imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import StateGraph

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("hf_token")

# ---------------------------------------------------------------------------
# Workflow state definition
# ---------------------------------------------------------------------------
class Workflow(TypedDict):
    code: str

# ---------------------------------------------------------------------------
# Workflow nodes
# ---------------------------------------------------------------------------
def parse_node(state: Workflow) -> Workflow:
    """Extract Python code block from LLM output."""
    code = state["code"]
    if "```python" in code:
        result = code.split("```python")[1].split("```")[0].strip()
    else:
        result = code.strip()
    return {"code": result}

def validate_node(state: Workflow) -> Workflow:
    """Ensure Streamlit import is present."""
    code = state["code"]
    if "import streamlit as st" not in code:
        code = "import streamlit as st\n" + code
    return {"code": code}

def run_workflow(raw_output: str) -> str:
    """Run the LangGraph workflow: parse → validate."""
    graph = StateGraph(Workflow)
    graph.add_node("parse", parse_node)
    graph.add_node("validate", validate_node)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "validate")
    app = graph.compile()
    result = app.invoke({"code": raw_output})
    return result["code"]

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("Text to Streamlit Chatbot")

user_input = st.text_area("Enter your text to generate Streamlit code...")

if st.button("Generate Code"):
    # Prompt template
    template = PromptTemplate(
        input_variables=["text"],
        template="Convert the following text into streamlit code:\n{text}"
    )

    # Hugging Face LLM
    llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b"))
    parser = StrOutputParser()
    chain = template | llm | parser

    # Generate code
    raw_output = chain.invoke({"text": user_input})
    code = run_workflow(raw_output)

    # Show code
    st.code(code, language="python")

    # Try running generated code
    try:
        exec(code, globals())
    except Exception as e:
        st.error(f"Error running generated code: {e}")
