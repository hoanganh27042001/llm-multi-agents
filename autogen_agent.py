import logging
import os 
import autogen.runtime_logging
import chromadb
from typing_extensions import Annotated
import pandas as pd

import autogen
from autogen import UserProxyAgent, AssistantAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.retrieve_utils import TEXT_FORMATS
from autogen import register_function

import json
from dotenv import load_dotenv
load_dotenv()
config_list = [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]
llm_config = {"config_list": config_list, "cache_seed": None}

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

def get_log(dbname="logs.db", table="chat_completions"): #table: [('chat_completions',), ('agents',), ('oai_wrappers',), ('oai_clients',), ('version',)]
    import sqlite3
    
    con = sqlite3.connect(dbname)
    query = f"SELECT * from {table}"
    cursor = con.execute(query)
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    data = [dict(zip(column_names,row)) for row in rows] 
    con.close()
    os.remove(dbname)
    return data

def str_to_dict(s):
    return json.loads(s)

# Start logging
logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
print("Logging session ID: " + str(logging_session_id))

# define function tool
def get_balance(id: int) -> str:
    print("balance of user is: ", id)
    return str(100)

def get_info(id: int) -> str:
    print("Info of user: ", id)
    return "User info name is John"

def get_info_history(id: int) -> str:
    print("info history of user: ", id)
    return "User info was born in 1999 and is a software engineer."
 
def get_balance_history(id: int) -> str:
    print("history balance of user is: ", id)
    return str(1000000)

# Create agents workflow
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    human_input_mode="NEVER",
    default_auto_reply="TERMINATE",
    is_termination_msg=termination_msg,
    code_execution_config={
        # "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    }
)

assistant = AssistantAgent(
    name="Assistant",
    is_termination_msg=termination_msg,
    system_message="""You are a helpful AI assitant.
    You can do some simple task like get information or balance of user id, but not the history. Return 'TERMINATE' when the task is done.
    """,
    llm_config=llm_config,
)
history_assistant = AssistantAgent(
    name="History_Assistant",
    is_termination_msg=termination_msg,
    system_message="You are a helpful AI assistant to retrieve history of balance or information from an user id. Return 'TERMINATE' when the task is done.",
    llm_config=llm_config,
)
coder = autogen.AssistantAgent(
    name="Coder",
    system_message="""You are a Senior Python Coder. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
    llm_config=llm_config,
    is_termination_msg=termination_msg,
    description="A coder"
)
executor = autogen.UserProxyAgent(
    name="Code_Executor",
    system_message="Executor. Execute the code written by the Coder and report the result.",  
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "groupchat",
        "use_docker": False,
    },
    description="An executor"
)

math = autogen.AssistantAgent(
    name="Mathematian",
    system_message="You are an professional mathemetian. You will solve difficult math problem by breaking it down and handle sub-problem step by step. Reply `TERMINATE` if the task is done.",
    llm_config=llm_config,
    is_termination_msg=termination_msg,
    description="A mathematian",
)

history = autogen.AssistantAgent(
    name="Historian",
    system_message="You are a helpful AI assistant to retrieve history of balance or information from an user id. You are also a professor in Vietnamese history. Reply `TERMINATE` if the task is done.",
    llm_config=llm_config,
    is_termination_msg=termination_msg,
    description="A historian"
)

rag_aid = RetrieveUserProxyAgent(
    name="RAG_Assistant",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": r"C:\Users\nguye\OneDrive - Hanoi University of Science and Technology\Documents\LLMs\AutoGen\whales_pdf.pdf",
        "chunk_token_size": 1000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,
    description="Assistant who has extra content retrieval power for solving difficult problems.",
)

whaler = RetrieveAssistantAgent(
    name = "Whales_Professor",
    system_message = "You are a Whales professor who can answer anything related to WHALEs support by the documents.",
    llm_config=llm_config,
    is_termination_msg=termination_msg,
    description="A whale professor"
)

# Register the tool function with the user proxy agent.
register_function(
    get_info,
    caller=assistant,
    executor=user_proxy,
    name="get_info",
    description="get info of an user id",
)
register_function(
    get_balance,
    caller=assistant,
    executor=user_proxy,
    name="get_balance",
    description="get balance of user id",
)
register_function(
    get_info_history,
    caller=history_assistant,
    executor=user_proxy,
    name="get_info_history",
    description="get history info of an user id",
)
register_function(
    get_balance_history,
    caller=history_assistant,
    executor=user_proxy,
    name="get_info_history",
    description="get history balance of an user id",
)
# task = "Who is Quang Trung?"
# task = "Find all prime number under 100."
# task = "Introduce the detail of Whales Market’s native utility token."
# task = "What is the derivative of x**3?"
# task = "get balance of user id 1."
task = "What is the history balance of user id 1."

def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    
    if last_speaker is coder:
        return executor 
    elif last_speaker is executor:
        if messages[-1]["content"] == "exitcode: 1":
            return coder
        else:
            return None 
    elif last_speaker is math:
        return None 
    elif last_speaker is history:
        return None 
    elif last_speaker is whaler:
        return history
    else:
        return "auto"

def _reset_agents():
    user_proxy.reset() 
    rag_aid.reset() 
    coder.reset() 
    executor.reset() 
    history.reset()     
def call_rag_chat():
    _reset_agents()
    def retrieve_content(
        message: Annotated[str,"Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering."],
        n_results: Annotated[int, "number of results"] = 2,
    ) -> str:
        rag_aid.n_results = n_results # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = rag_aid._check_update_context(message)
        if (update_context_case1 or update_context_case2) and rag_aid.update_context:
            rag_aid.problem = message if not hasattr(rag_aid, "problem") else rag_aid.problem
            _, ret_msg = rag_aid._generate_retrieve_user_reply(message)
        else:
            _context = {"problem": message, "n_results": n_results}
            ret_msg = rag_aid.message_generator(rag_aid, None, _context)
        return ret_msg if ret_msg else message

    rag_aid.human_input_mode = "NEVER" # Disable human input for rag_aid since it only retrieves content.
    for caller in [whaler]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content for code generation and question answering.", api_style="function"
        )(retrieve_content)
    for exe in [user_proxy]:
        exe.register_for_execution()(d_retrieve_content)
        
    groupchat = autogen.GroupChat(
        agents=[user_proxy, assistant,history_assistant, coder, math, history, whaler],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )
    
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list, "timeout": 60, "temperature": 0})
    
    # Start chatting with the boss as this is the user proxy agent.
    user_proxy.initiate_chat(manager, message=task)
call_rag_chat()
        
# groupchat = autogen.GroupChat(agents=[user_proxy, coder, executor, math, history], messages=[], max_round=10, speaker_selection_method=state_transition,)
# manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# user_proxy.initiate_chat(manager, message=task)

autogen.runtime_logging.stop()

log_data = get_log() 
log_data_df = pd.DataFrame(log_data)

log_data_df["total_tokens"] = log_data_df.apply(lambda row: str_to_dict(row["response"])["usage"]["total_tokens"], axis=1)

log_data_df["request"] = log_data_df.apply(lambda row: str_to_dict(row["request"])["messages"][0]["content"], axis=1)

log_data_df["response"] = log_data_df.apply(lambda row: str_to_dict(row["response"])["choices"][0]["message"]["content"], axis=1)
# print(log_data_df)
log_data_df.to_csv("log.csv", index=False)

# Computing cost
# Sum total tokens for all sessions
total_tokens = log_data_df["total_tokens"].sum()
# Sum total cost for all sessions
total_cost = log_data_df["cost"].sum()
# Total tokens for specific session
session_tokens = log_data_df[log_data_df["session_id"] == logging_session_id]["total_tokens"].sum()
session_cost = log_data_df[log_data_df["session_id"] == logging_session_id]["cost"].sum()

print("Total tokens for all sessions: " + str(total_tokens) + ", total cost: " + str(round(total_cost, 4)))
print(
    "Total tokens for session "
    + str(logging_session_id) 
    + ": " 
    + str(session_tokens) 
    + ", cost: " 
    + str(round(session_cost, 4)) 
)
