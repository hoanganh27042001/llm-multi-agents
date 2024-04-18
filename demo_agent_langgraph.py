import getpass
import os
import uuid


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# Optional, add tracing in LangSmith.
# This will help you visualize and debug the control flow
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Demo Agent"

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import JSONLoader
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import json
from typing import Callable, Dict, List, Optional, Union
from pathlib import Path
from pprint import pprint

import operator
from typing import Annotated, Sequence, TypedDict

from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolInvocation
from langchain_core.output_parsers import StrOutputParser


file_path = "../docs/intent_trade_v2.json"
# data = json.loads(Path(file_path).read_text())

"""Loader that loads data from JSON."""
class JSONLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        content_key: Optional[str] = None,
    ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""
        docs=[]
        # Load JSON file
        with open(self.file_path) as file:
            data = json.load(file)
        
        for page in data:
            docs.append(Document(page_content=page['pageContent'], metadata=page['metadata']))
        return docs
    
loader = JSONLoader(file_path=file_path)
data = loader.load()
vectorstore = Chroma.from_documents(
    documents=data,
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

# question = "what is intend trade"
# docs = retriever.get_relevant_documents(question)
# docs

from langchain.tools.retriever import create_retriever_tool

tool = create_retriever_tool(
    retriever, 
    "retrieve_docs",
    "Search and return information from the internal source about some blockchain knowledge."
)
tools = [tool]
from langgraph.prebuilt import ToolExecutor
tool_executor = ToolExecutor(tools)

import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
# Edges
def should_retrieve(state):
    """
    Decide whether the agent should retrieve more information or end the process.
    
    This function checks the last message in the state for a function call. If a function call is
    present, the process continues to retrieve information. Otherwise, it ends the process.

    Args:
        state (messages): The current state
        
    Returns:
        str: A decision to either "continue" the retrieval process or "end" it
    """
    print("---DECIDE TO RETRIEVE---")
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        print("---DECISION: DO NOT RETRIEVE / DONE---")
        return "end"
    # Otherwise there is a function call, so we continue
    else:
        print("---DECISION: RETRIEVE---")
        return "continue"
    
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state
        
    Returns:
        str: A decision for whether the documents are relevant or not
    """
    print("---CHECK RELEVANCE---")
    
    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
        
    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", streaming=True)
    # Tool
    grade_tool_oai = convert_to_openai_tool(grade)
    
    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[convert_to_openai_tool(grade_tool_oai)],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )
    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])
    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader accessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    # Chain
    chain = prompt | llm_with_tool | parser_tool
    
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    
    score = chain.invoke(
        {"question": question,
         "context": docs}
    )
    
    if grade == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "yes"
    
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(grade)
        return "no"
    
### Nodes
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.
    
    Args:
        state (messages): The current state
        
    Returns:
        dict: The updated state with the agent response apended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo")
    functions = [format_tool_to_openai_function(t) for t in tools]
    model = model.bind_functions(functions)
    response = model.invoke(messages)
    
    return {"messages": [response]}

def retrieve(state):
    """
    Use tool to execute retrieval.
    
    Args:
        state (messages): the current state
        
    Returns:
        dict: The updated state with retrieved docs
    """
    print("---EXECUTE RETRIEVAL---")
    messages = state["messages"]
    
    last_message = messages[-1]
    
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    
    return {"messages": [function_message]}

def rewrite(state):
    """
    Transform the query to produce a better question.
    
    Args:
        state (messages): The current state
        
    Returns:
        dict: The updated state with re-phased question
    """
    
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    
    msg = [HumanMessage(
        content=f""" \n
        Look at the input and try to reason about the underlying semantic intent / meaning. \n
        Here is the initial question:
        \n ------- \n
        {question}
        \n ------- \n
        Formulate an improved question: """,
    )]
    
    # Grader 
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", streaming=True)
    response = model.invoke(msg)
    return {"message": [response]}

def generate(state):
    """
    Generate answer
    
    Args:
        state (messages): The current state
        
    Returns:
        dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["message"]
    question = messages[0].content
    last_message = messages[-1]
    
    question = messages[0].content
    docs = last_message.content
    
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    
    # Run 
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

# Graph

from langgraph.graph import END, StateGraph

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_retrieve,
    {
        "continue": "retrieve",
        "end": END,
    }
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "yes": "generate",
        "no": "rewrite",
    },
)

workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

app = workflow.compile()

inputs = {
    "messages": [
        HumanMessage(
            content="What is intend trade?"
        )
    ]
}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Output from node '{key}':")
        pprint("---")
        pprint(value, indent=2, width=80, depth=None)
    pprint("\n---\n")