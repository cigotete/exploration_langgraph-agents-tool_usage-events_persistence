from dotenv import load_dotenv

_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

from uuid import uuid4
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

"""
In previous examples we've annotated the `messages` state key
with the default `operator.add` or `+` reducer, which always
appends new messages to the end of the existing messages array.

Now, to support replacing existing messages, we annotate the
`messages` key with a customer reducer function, which replaces
messages with the same `id`, and appends them otherwise.
"""
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


# Tool definition
tool = TavilySearchResults(max_results=2)


# Agent definition with manual human approval.
class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"] # Manual approval. This interruption happens before we call the action node where all tools are called.
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        print(state)
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-3.5-turbo")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)


"""
messages = [HumanMessage(content="Whats the weather in SF?")]
thread = {"configurable": {"thread_id": "1"}}
# Streaming back responses 
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)


abot.graph.get_state(thread)
abot.graph.get_state(thread).next

### Continue after interrupt.
print("*"*50)
print("Continue after interrupt.")
print("*"*50)

for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)

abot.graph.get_state(thread)
abot.graph.get_state(thread).next

# Testing in a loop (new thread).
print("*"*50)
print("Testing in a loop (new thread).")
print("*"*50)
messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
while abot.graph.get_state(thread).next:
    print("\n", abot.graph.get_state(thread),"\n")
    _input = input("proceed?")
    if _input != "y":
        print("aborting")
        break
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)
"""

# Modify State.
print("*"*50)
print("Modifying State.")
print("*"*50)
messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "3"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

print("abot.graph.get_state(thread)" + "-"*50)
print(abot.graph.get_state(thread))

print("abot.graph.get_state_history(thread)" + "-"*50)
states = []
for state in abot.graph.get_state_history(thread):
    print(state)
    print('--')
    states.append(state)

print("current_values.values['messages'][-1]" + "-"*50)
current_values = abot.graph.get_state(thread)
print(current_values.values['messages'][-1])

print("current_values.values['messages'][-1].tool_calls" + "-"*50)
print(current_values.values['messages'][-1].tool_calls)

# Updating tool query
print("Updating tool query:" + "-"*50)
_id = current_values.values['messages'][-1].tool_calls[0]['id']
current_values.values['messages'][-1].tool_calls = [
    {'name': 'tavily_search_results_json',
  'args': {'query': 'current weather in Louisiana'},
  'id': _id}
]

# Updating tool at state.
print("Updating tool on state." + "-"*50)
abot.graph.update_state(thread, current_values.values)

print("abot.graph.get_state(thread)" + "-"*50)
print(abot.graph.get_state(thread))

for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)

print("abot.graph.get_state_history(thread)" + "-"*50)
states = []
for state in abot.graph.get_state_history(thread):
    print(state)
    print('--')
    states.append(state)

print("State to return -3:" + "-"*50)
to_replay = states[-3]

print("Printing events:" + "-"*50)
for event in abot.graph.stream(None, to_replay.config):
    for k, v in event.items():
        print(v)


### Go back in time and edit.
print("*"*50)
print("Modifying State.")
print("*"*50)

print("Printing last state modified:" + "-"*50)
print(to_replay)

print("Updating -1 state:" + "-"*50)
_id = to_replay.values['messages'][-1].tool_calls[0]['id']
to_replay.values['messages'][-1].tool_calls = [{'name': 'tavily_search_results_json',
  'args': {'query': 'current weather in LA, accuweather'},
  'id': _id}]

branch_state = abot.graph.update_state(to_replay.config, to_replay.values)

print("Printing events:" + "-"*50)
for event in abot.graph.stream(None, branch_state):
    for k, v in event.items():
        if k != "__end__":
            print(v)


### Adding message (a mockup) as a model response for a state
"""Adding message (a mockup) as a model response for a state \
at a given time.
Here is implemented "as_node" parameter, to declare that is actually \
a response (in this case, a mockup response). We are updating \
the action node as a real action response. \
Before is added this message, the current state of this graph is \
about to go into the action node.
"""
print("PRinting last state modified:" + "-"*50)
print(to_replay)

_id = to_replay.values['messages'][-1].tool_calls[0]['id']

state_update = {"messages": [ToolMessage(
    tool_call_id=_id,
    name="tavily_search_results_json",
    content="54 degree celcius",
)]}

branch_and_add = abot.graph.update_state(
    to_replay.config, 
    state_update, 
    as_node="action")

print("Printing events:" + "-"*50)
for event in abot.graph.stream(None, branch_and_add):
    for k, v in event.items():
        print(v)