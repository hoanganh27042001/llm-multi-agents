import getpass
import os
import uuid
import json
from typing import Callable, Dict, List, Optional, Union, Annotated, Sequence, TypedDict
from pathlib import Path
from pprint import pprint
import operator
import functools
import operator

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
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
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph

from langgraph.prebuilt import ToolExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser



def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# Optional, add tracing in LangSmith.
# This will help you visualize and debug the control flow
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Demo Agent"


file_path = "../docs/intent_trade_v2.json"
file_path2 = "../docs/whales_market_v2.json"
file_price = "price.pdf"
# data = json.loads(Path(file_path).read_text())
"""Load csv file"""
loader = PyPDFLoader(file_path = file_price)
data_price = loader.load_and_split()

vectorstore_price = Chroma.from_documents(
    documents=data_price,
    collection_name="price",
    embedding=OpenAIEmbeddings(),
)
retrieve_price = vectorstore_price.as_retriever(search_kwargs={"k": 1})
tool_price = create_retriever_tool(
    retrieve_price,
    "retrieve_price_history",
    "Search and return information about price history of some blockchain token."
)

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
    collection_name="data_offline",
    embedding=OpenAIEmbeddings(),
)
retrieve_offline_info = vectorstore.as_retriever()

loader1 = JSONLoader(file_path=file_path2)
data1 = loader1.load()
vectorstore1 = Chroma.from_documents(
    documents=data1,
    collection_name="data_online",
    embedding=OpenAIEmbeddings(),
)
retrieve_online_info = vectorstore1.as_retriever()


tool_offline = create_retriever_tool(
    retrieve_offline_info, 
    "retrieve_offline_docs",
    "Search and return information from the internal source about some blockchain knowledge."
)
tool_online = create_retriever_tool(
    retrieve_online_info, 
    "retrieve_online_docs",
    "Search and return information from the online source about some blockchain knowledge."
)
tools = [tool_offline, tool_online, tool_price]
tool_executor = ToolExecutor(tools)

tavily_tool = TavilySearchResults(max_results=2)
tool_search_executor = ToolExecutor([tavily_tool])


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    
# Edges
def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    functions = [format_tool_to_openai_function(t) for t in tools]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt | llm.bind_functions(functions)
    
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

def agent_node(state, agent, name):
    response = agent.invoke(state)
    return {
        "messages": [response],
        "next": name,
    }
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

def tool_search(state):
    print("---EXECUTE SEARCHING---")
    messages = state["messages"]
    last_message = messages[-1]
    
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    response = tool_search_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}
def tool_rag(state):
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
    
    print("---GENERATE---")
    question = messages[0].content
    
    docs = function_message.content
    
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    
    # Run 
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    if "function_call" in last_message.additional_kwargs:
        if last_message.additional_kwargs["function_call"]["name"] == "tavily_search_results_json":
            return "search_tool"
        else:
            return "call_tool"
    return "end"
# Graph
llm = ChatOpenAI(model="gpt-3.5-turbo")

docs_agent = create_agent(
    llm,
    [tool_offline, tool_online],
    system_message="You should provide information about some blockchain knowledge."
)
docs_node = functools.partial(agent_node, agent=docs_agent, name="DocAgent")
price_agent = create_agent(
    llm,
    [tavily_tool, tool_price],
    system_message="Use tavily_tool if got `Search` keyword in the question. Otherwise, you should retrieve or search for price of any blockchain token."
)
price_node = functools.partial(agent_node, agent=price_agent, name="PriceAgent")

supervisor_agent = create_team_supervisor(
    llm,
    """You are a supervisor who distribute task between the 
     following members: DocAgent, PriceAgent. Given the following user request,
     respond with the member to act next. Each member will perform a
     task and respond with their results and status. When finished,
     respond with FINISH.""",
     ["DocAgent", "PriceAgent"],
)
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_agent)
# DocAgent
workflow.add_node("DocAgent", docs_node)
workflow.add_node("call_tool", tool_rag)

# PriceAgent
workflow.add_node("PriceAgent", price_node)
workflow.add_node("search_tool", tool_search)

workflow.set_entry_point("supervisor")
members = ["DocAgent", "PriceAgent"]
workflow.add_edge("DocAgent", "supervisor")
workflow.add_edge("PriceAgent", "supervisor")
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.add_conditional_edges(
    "PriceAgent",
    router,
    {"call_tool": "call_tool", "search_tool": "search_tool", "end": END}
)
workflow.add_conditional_edges(
    "DocAgent",
    router,
    {"call_tool": "call_tool", "end": END}
)
workflow.add_edge("search_tool", END)
workflow.add_edge("call_tool", END)

app = workflow.compile()

# mess = "What is the price of Whale Markets?"
# mess = "Search for the current Whale Markets price?"
# mess = "What is intend trade (offline)?"
mess = "What is the referal program of Whale Markets?"
inputs = {
    "messages": [
        HumanMessage(
            content=mess
        )
    ]
}
for output in app.stream(inputs):
    # print("output: ",output)
    for key, value in output.items():
        pprint(f"Output from node '{key}':")
        pprint("---")
        pprint(value, indent=2, width=80, depth=None)
    pprint("\n---\n")