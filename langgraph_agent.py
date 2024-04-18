import getpass
import os

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated, List, Sequence, TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

import operator
import functools
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")
        
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Test Multi-agent"

# Create tools
repl = PythonREPL()
@tool
def python_repl(
    code: Annotated[str, "The python code to execute."]
):
    """You this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

@tool
def get_balance(id: int) -> str:
    """Get balance of a user id."""
    print(f"balance of user {id} is: ")
    return str(100)

@tool 
def get_info(id: int) -> str:
    """Get info of a user id."""
    print(f"Info of user {id} is: ")
    return "User infor name is John"
@tool 
def get_info_history(id: int) -> str:
    """Get info history of a user id."""
    print(f"Info history of user {id} is: ")
    return "User info was born in 1999 and is a software engineer."

@tool 
def get_balance_history(id: int) -> str:
    """Get balance history of a user id."""
    print("history balance of user {id} is: ")
    return str(1000000)

# Embedding
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=20
)
loader = PyPDFLoader("whales_pdf.pdf")
doc_splits = loader.load_and_split(text_splitter)
vectorstore =  Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag_chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

### Generate
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = prompt | llm | StrOutputParser()

# Create worker agent and supervisor
def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> str:
    """Create a function-calling agent and add it to the graph."""    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

# Graph
class RagState(TypedDict):
    """
    Represents the state of the graph.
    
    Attributes:
        question: question
        generation: LLM generation
        web_search: where to add search
        documents: list of documents"""
    question: str
    generation: str
    documents: List[str]
    
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str 
       
# Helper function
def retrieve(state):
    """Retrieve documents

    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer
    Args:
        state (dict): The current graph state
        
    Returns:
    state (dict): New key added to stated, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
    
def get_last_message(state: AgentState) -> str:
    # print("messages: ", state["messages"][-1])
    return state["messages"][-1].content
def join_graph(response: dict):
    # print("response: ", response)
    return {"messages": [HumanMessage(content=response["generation"])]}

# Define node
llm = ChatOpenAI(model="gpt-3.5-turbo")
assistant_agent = create_agent(llm, [get_info, get_balance], "You can retrieve information and balance from an user id account.")
assistant_node = functools.partial(agent_node, agent=assistant_agent, name="Assistant")

history_aid_agent = create_agent(llm, [get_info_history, get_balance_history], "You may help to retrieve history of infor or balance for an user id.")
history_aid_node = functools.partial(agent_node, agent=history_aid_agent, name="History_Assistant")

history_agent = create_agent(llm, [], "You are a Vietnamese historian. You can only answer question about Vietnamese history.")
history_node = functools.partial(agent_node, agent=history_agent, name="Historian")

code_agent = create_agent(llm, [python_repl], "You may generate safe python code to solve problems.")
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

# Build rag subgraph
ragflow = StateGraph(RagState)

ragflow.add_node("retrieve", retrieve)
ragflow.add_node("generate", generate)

ragflow.set_entry_point("retrieve")
ragflow.add_edge("retrieve", "generate")
ragflow.add_edge("generate", END)

chain = ragflow.compile()

def enter_chain(message: str):
    results = {
        "question": message,
    }
    return results
RAG_chain = enter_chain | chain
from pprint import pprint

# inputs = "What is the idea of Whale Markets?"
# for output in RAG_chain.stream(inputs):
#     for key, value in output.items():
#         pprint(f"Node '{key}':")
        
#     pprint("\n---\n")
# pprint(value["documents"])
# pprint(value["generation"])

members = ["Assistant", "Coder", "History_Assistant", "Historian", "Rag_node"]
supervisor_agent = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. "
    " If you need information about WHALE MARKETS, Rag_node can help you"
    " When the response is complete, respond with FINISH.",
    members,
)
workflow = StateGraph(AgentState)
workflow.add_node("Assistant", assistant_node)
workflow.add_node("Coder", code_node)
workflow.add_node("History_Assistant", history_aid_node)
workflow.add_node("Historian", history_node)
workflow.add_node("Rag_node", get_last_message | RAG_chain | join_graph)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_edge("Rag_node", END)

for member in members:
    workflow.add_edge(member, "supervisor")
    
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")
graph = workflow.compile()

# Invoke the team
for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="Write a function to get price of Solana token")
        ]
    },
    {"recursion_limit": 100}
):
    if "__end__" not in s:
        pprint(s)
        print("----")
        
        