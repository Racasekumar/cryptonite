import json
import os
import operator
from typing import TypedDict, Annotated, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

class InterviewState(TypedDict):
    """
    Represents the state of the interview.
    - messages: A list of all messages (chat history) in the conversation.
    - questions: The list of interview questions loaded from a JSON file.
    - current_question_index: The index of the question currently being asked.
    - interview_complete: A boolean flag to signal the end of the interview.
    - waiting_for_user: A flag to indicate when we need user input.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    questions: list
    current_question_index: int
    interview_complete: bool
    waiting_for_user: bool


with open("questions.json", "r") as f:
    questions = json.load(f)["questions"]


def ask_question_node(state: InterviewState):
    """Asks the next main interview question."""
    questions = state["questions"]
    index = state["current_question_index"]
    
    if index < len(questions):
        question = questions[index]
        print(f"\nAI Agent: {question}")
        return {
            "messages": [AIMessage(content=question)],
            "waiting_for_user": True
        }
    else:
        # Signal that the interview is complete
        return {"interview_complete": True}


def decide_followup_node(state: InterviewState):
    """
    Analyzes the user's response to decide if a follow-up is needed.
    """
    # Use the conversation history to inform the decision
    prompt = PromptTemplate(
        input_variables=["chat_history"],
        template="""You are an interviewer. Based on the following conversation history, 
        decide if the candidate's last answer is sufficiently detailed to move on, or if you need to ask a follow-up question.
        Respond with either "next_question" or "follow_up". Do not add any other text.
        
        Chat History:
        {chat_history}
        """
    )
    chain = prompt | llm
    
    # Pass only the last few messages to keep the context focused
    response = chain.invoke({"chat_history": state["messages"][-4:]})
    
    # Return the decision and update state accordingly
    decision = response.content.strip().lower()
    if "follow_up" in decision:
        return {"waiting_for_user": False}  # Continue with follow-up
    else:
        return {
            "current_question_index": state["current_question_index"] + 1,
            "waiting_for_user": False
        }


def generate_followup_node(state: InterviewState):
    """Generates a contextual follow-up question based on the user's last response."""
    prompt = PromptTemplate(
        input_variables=["chat_history", "main_question"],
        template="""You are an interviewer. Based on the conversation history,
        generate a concise and specific follow-up question.
        
        Main Question: {main_question}
        Chat History: {chat_history}
        
        Follow-up Question:
        """
    )
    chain = prompt | llm
    
    # Get the most recent main question
    main_question = state["questions"][state["current_question_index"]]
    
    response = chain.invoke({
        "chat_history": state["messages"],
        "main_question": main_question
    })

    print(f"\nAI Agent (Follow-up): {response.content.strip()}")
    return {
        "messages": [AIMessage(content=response.content.strip())],
        "waiting_for_user": True
    }


def end_interview_node(state: InterviewState):
    """Generates a final, conclusive message when the interview is over."""
    prompt = PromptTemplate(
        input_variables=["chat_history"],
        template="""You have just finished an interview. Write a brief, polite, and conclusive message to the candidate.
        
        Chat History:
        {chat_history}
        """
    )
    chain = prompt | llm
    response = chain.invoke({"chat_history": state["messages"]})
    print(f"\nAI Agent: {response.content.strip()}")
    return {"messages": [AIMessage(content=response.content.strip())]}


def route_after_decision(state: InterviewState):
    """Routes based on the decision made."""
    # Check if interview is complete
    if state.get("interview_complete", False):
        return "end_interview"
    
    # Check if we moved to next question (index was updated)
    if not state.get("waiting_for_user", False):
        # If current_question_index was updated, we go to next question
        # If not, we generate follow-up
        return "ask_question"
    
    return "generate_followup"


# This defines the flow and logic of the entire application.
# def create_workflow ():
#         workflow = StateGraph(InterviewState)
#         # Add the nodes
#         workflow.add_node("ask_question", ask_question_node)
#         workflow.add_node("decide_followup", decide_followup_node)
#         workflow.add_node("generate_followup", generate_followup_node)
#         workflow.add_node("end_interview", end_interview_node)

#         # Set the entry point of the graph
#         workflow.set_entry_point("ask_question")

#         # Define the edges and conditional routing
#         workflow.add_conditional_edges(
#             "ask_question",
#             lambda state: "end_interview" if state.get("interview_complete") else "decide_followup"
#         )

#         workflow.add_conditional_edges(
#             "decide_followup",
#             lambda state: "ask_question" if state["current_question_index"] != state.get("original_index", state["current_question_index"]) else "generate_followup"
#         )

#         workflow.add_edge("generate_followup", "decide_followup")
#         workflow.add_edge("end_interview", END)

#         # Compile the graph
#         # app = workflow.compile()
#         return workflow.compile()


workflow = StateGraph(InterviewState)
    
workflow.add_node("ask_question", ask_question_node)
workflow.add_node("decide_followup", decide_followup_node)
workflow.add_node("generate_followup", generate_followup_node)
workflow.add_node("end_interview", end_interview_node)

# Set the entry point of the graph
workflow.set_entry_point("ask_question")

# Define the edges and conditional routing
workflow.add_conditional_edges(
    "ask_question",
    lambda state: "end_interview" if state.get("interview_complete") else "decide_followup"
)

workflow.add_conditional_edges(
    "decide_followup",
        lambda state: "ask_question" if state["current_question_index"] != state.get("original_index", state["current_question_index"]) else "generate_followup"
)

workflow.add_edge("generate_followup", "decide_followup")
workflow.add_edge("end_interview", END)

app = workflow.compile()


# from IPython.display import Image, Markdown
# print(Image(app.get_graph().draw_mermaid_png()))



# This part simulates the user interaction and runs the agent.

# Simplified approach - run the interview step by step
def run_interview():
    # Initialize the state properly with all required fields
    state = {
        "messages": [],
        "questions": questions,
        "current_question_index": 0,
        "interview_complete": False,
        "waiting_for_user": False
    }
    
    print("--- AI Interview Agent Started ---")
    print("Type 'exit' to end the interview at any time.\n")
    
    while not state.get("interview_complete", False):
        # If we're not waiting for user input, process the next step
        if not state.get("waiting_for_user", False):
            # Ask question or end interview
            if state["current_question_index"] < len(state["questions"]):
                result = ask_question_node(state)
                # Update state
                if "messages" in result:
                    state["messages"].extend(result["messages"])
                state["waiting_for_user"] = result.get("waiting_for_user", False)
                state["interview_complete"] = result.get("interview_complete", False)
            else:
                state["interview_complete"] = True
                end_interview_node(state)
                break
        
        # Get user input
        if state.get("waiting_for_user", False):
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print("--- Interview Ended ---")
                break
            
            # Add user message
            state["messages"].append(HumanMessage(content=user_input))
            state["waiting_for_user"] = False
            
            # Decide if follow-up is needed
            decision_result = decide_followup_node(state)
            
            # Check if we should move to next question
            if "current_question_index" in decision_result:
                state["current_question_index"] = decision_result["current_question_index"]
            else:
                # Generate follow-up
                followup_result = generate_followup_node(state)
                state["messages"].extend(followup_result["messages"])
                state["waiting_for_user"] = followup_result.get("waiting_for_user", False)
    
    print("\n--- Interview Completed ---")

# Run the interview
if __name__ == "__main__":
    run_interview()