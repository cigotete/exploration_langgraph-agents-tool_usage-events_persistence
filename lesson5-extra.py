from dotenv import load_dotenv

_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langgraph.checkpoint.sqlite import SqliteSaver

class AgentState(TypedDict):
    lnode: str
    scratch: str
    count: Annotated[int, operator.add]

def node1(state: AgentState):
    print(f"node1, count:{state['count']}")
    return {"lnode": "node_1",
            "count": 1,
           }

def node2(state: AgentState):
    print(f"node2, count:{state['count']}")
    return {"lnode": "node_2",
            "count": 1,
           }

def should_continue(state):
    return state["count"] < 3

builder = StateGraph(AgentState)
builder.add_node("Node1", node1)
builder.add_node("Node2", node2)

builder.add_edge("Node1", "Node2")
builder.add_conditional_edges("Node2", 
                              should_continue, 
                              {True: "Node1", False: END})
builder.set_entry_point("Node1")

memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": str(1)}}
graph.invoke({"count":0, "scratch":"hi"}, thread)


# Look at current state
print("*"*50)
print("Look at current state")
print("*"*50)

print(graph.get_state(thread))


# Look at state history
print("*"*50)
print("Look at state history")
print("*"*50)

print("history" + "-"*50)
for state in graph.get_state_history(thread):
    print(state, "\n")

print("history detail" + "-"*50)
states = []
for state in graph.get_state_history(thread):
    states.append(state.config)
    print(state.config, state.values['count'])

print("state -3" + "-"*50)
print(graph.get_state(states[-3]))


# Go Back in Time
print("*"*50)
print("Go Back in Time")
print("*"*50)
# Use that state in invoke to go back in time. Notice it uses states[-3] as current_state and continues to node2,
graph.invoke(None, states[-3])

print("history" + "-"*50)
thread = {"configurable": {"thread_id": str(1)}}
for state in graph.get_state_history(thread):
    print(state.config, state.values['count'])


# Modify State
print("*"*50)
print("Modify State - New thread and running to clean out history.")
print("*"*50)

thread2 = {"configurable": {"thread_id": str(2)}} # new thread and running to clean out history.
graph.invoke({"count":0, "scratch":"hi"},thread2)

print("Generating events first." + "-"*50)
states2 = []
for state in graph.get_state_history(thread2):
    states2.append(state.config)
    print(state.config, state.values['count'])

print("Created variable save_state." + "-"*50)
save_state = graph.get_state(states2[-3])
print(save_state)

print("Modifying values 'count' and 'scratch' of save_state." + "-"*50)
save_state.values["count"] = -3
save_state.values["scratch"] = "hello"
print(save_state)

print("Updating the state with graph.update_state." + "-"*50)
graph.update_state(thread2,save_state.values)

print("Printing whole state." + "-"*50)
for state in graph.get_state_history(thread2):
    print(state.config, state.values['count'])


# Try again with as_node
print("*"*50)
print("Try again with as_node")
print("*"*50)

print("Updating with 'as_node'. Printing state history." + "-"*50)
graph.update_state(thread2,save_state.values, as_node="Node1")
for state in graph.get_state_history(thread2):
    print(state, '\n')

print("Using invoke." + "-"*50)
graph.invoke(None,thread2)

print("Printing state history'." + "-"*50)
for state in graph.get_state_history(thread2):
    print(state,"\n")
print("Printing state history'." + "-"*50)
for state in graph.get_state_history(thread2):
    print(state.config, state.values['count'])