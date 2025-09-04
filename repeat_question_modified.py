import json
import asyncio
import os
import operator
from typing import TypedDict, Annotated, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint import MemorySaver  # Added for persistent state and resumption
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.memory import MemorySaver

from IPython.display import Image, Markdown
from langsmith import traceable

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import deepgram, cartesia, openai, silero, elevenlabs

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

class InterviewState(TypedDict):
    """
    Represents the state of the interview.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    questions: list
    current_question_index: int
    followup_count: int
    max_followups: int   # Maximum allowed follow-ups per question
    interview_complete: bool
    waiting_for_user: bool
    turn_complete: bool
    interview_started: bool
    last_question: str  # Store the last asked question
    needs_repeat: bool  # Flag for repeat request
    needs_explain: bool  # Flag for explain request

# Load questions from JSON file
with open("/Users/rakeshrout/demointerview/questions.json", "r") as f:
    questions = json.load(f)["questions"]

# Global state to track interview progress
interview_state = {
    "messages": [],
    "questions": questions,
    "current_question_index": 0,
    "followup_count": 0,
    "max_followups": 2,
    "interview_complete": False,
    "waiting_for_user": False,
    "turn_complete": False,
    "interview_started": False,
    "last_question": "",
    "needs_repeat": False,
    "needs_explain": False
}

@traceable(name="ask_main_question_node")
def ask_main_question_node(state: InterviewState):
    """Asks the next main interview question and resets follow-up counter."""
    questions = state["questions"]
    index = state["current_question_index"]
    
    if index < len(questions):
        question = questions[index]
        return {
            "messages": [AIMessage(content=question)],
            "last_question": question,
            "followup_count": 0,  # Reset follow-up counter for new question
            "waiting_for_user": True,
            "turn_complete": True,
        }
    else:
        return {"interview_complete": True}



@traceable(name="should_ask_followup")
def should_ask_followup_repeat_explain(state: InterviewState) -> str:
    """
    Determines the next step based on follow-up limits and response quality.
    Returns: 'next_question', 'followup' or 'end_interview'
    """
    # Check if interview is complete
    if state["current_question_index"] >= len(state["questions"]):
        return "end_interview"
    
    # Check if we've reached the follow-up limit
    if state["followup_count"] >= state["max_followups"]:
        return "next_question"
    
    # Analyze response quality to decide if follow-up is needed
    prompt = PromptTemplate(
        input_variables=["chat_history", "followup_count", "max_followups", "main_question"],
        template="""You are an interviewer analyzing a candidate's response. 
        
        Current follow-up count: {followup_count}/{max_followups}
        
        Based on the conversation history, decide if the candidate's last answer needs clarification or elaboration.
        
        Guidelines:
        - If the answer is comprehensive and clear, respond with "next_question"
        - If the answer is vague, incomplete, or needs clarification, respond with "followup" 
        - Consider that you can only ask {max_followups} follow-ups total for this question
        
        Respond with ONLY "next_question" or "followup" - no other text.
        
        Chat History:
        {chat_history}
        """
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "chat_history": state["messages"][-4:],
        "followup_count": state["followup_count"],
        "max_followups": state["max_followups"]
    })
    
    decision = response.content.strip().lower()
    if "followup" in decision:
        return "followup"
    else:
        return "next_question"

@traceable(name="generate_followup_node")
def generate_followup_node(state: InterviewState):
    """Generates a contextual follow-up question and increments the counter."""
    prompt = PromptTemplate(
        input_variables=["chat_history", "main_question", "followup_count"],
        template="""You are an interviewer conducting a follow-up question.
        
        Main Question: {main_question}
        This is follow-up #{followup_count}
        
        Based on the conversation history, generate a specific, targeted follow-up question 
        that will help clarify or expand on the candidate's previous response.
        
        Keep the follow-up concise and focused. Do not repeat the main question or any answers.
        
        Chat History: {chat_history}
        
        Follow-up Question:
        """
    )
    
    chain = prompt | llm
    main_question = state["questions"][state["current_question_index"]]
    
    response = chain.invoke({
        "chat_history": state["messages"][-6:],  # More context for follow-ups
        "main_question": main_question,
        "followup_count": state["followup_count"] + 1
    })

    followup_question = response.content.strip()

    return {
        "messages": [AIMessage(content=followup_question)],
        "last_question": followup_question,
        "followup_count": state["followup_count"] + 1,
        "waiting_for_user": True,
        "turn_complete": True,
    }

@traceable(name="move_to_next_question_node")
def move_to_next_question_node(state: InterviewState):
    """Moves to the next question by incrementing the index."""
    return {
        "current_question_index": state["current_question_index"] + 1,
        "followup_count": 0,  # Reset follow-up counter
    }

@traceable(name="end_interview_node")
def end_interview_node(state: InterviewState):
    """Generates a final message when the interview is over."""
    prompt = PromptTemplate(
        input_variables=[],
        template="""You have just finished conducting an interview. Write a brief, professional, and concise closing message to the candidate. It should be about two sentences. Thank them for their time and let them know we'll be in touch about the next steps."""
    )
    
    chain = prompt | llm
    response = chain.invoke({})
    
    return {
        "messages": [AIMessage(content=response.content.strip())],
        "turn_complete": True,
        "interview_complete": True
    }

@traceable(name="user_input_node")
# Dummy node for human input (external handling)
def user_input_node(state: InterviewState):
    return {}

@traceable(name="repeat_question_node")
def repeat_question_node(state: InterviewState):
    """Repeats the last question."""
    return {
        "messages": [AIMessage(content=f"Let me repeat that question: {state['last_question']}")],
        "waiting_for_user": True,
        "turn_complete": True,
        "needs_repeat": False
    }

@traceable(name="explain_question_node")
def explain_question_node(state: InterviewState):
    """Provides an explanation for the last question."""
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""Briefly clarify the following interview question without giving any hints or the answer. Just rephrase it in a simpler way. Keep it to one or two sentences.

        Question: {question}
        """
    )
    
    chain = prompt | llm
    response = chain.invoke({"question": state["last_question"]})
    
    return {
        "messages": [AIMessage(content=response.content)],
        "waiting_for_user": True,
        "turn_complete": True,
        "needs_explain": False
    }

@traceable(name="determine_next_step")
def determine_next_step(state: InterviewState) -> str:
    """
    Determines the next step in the interview process.
    Returns: 'next_question', 'followup', 'repeat', 'explain', or 'end_interview'
    """
    # Check if interview is complete
    if state["current_question_index"] >= len(state["questions"]):
        return "end_interview"
    
    # Check for repeat or explain requests
    if state["needs_repeat"]:
        return "repeat"
    if state["needs_explain"]:
        return "explain"
    
    # Check follow-up logic
    if state["followup_count"] >= state["max_followups"]:
        return "next_question"
        
    # Analyze response quality
    last_message = state["messages"][-1].content.lower() if state["messages"] else ""
    
    # Check for repeat/explain keywords in user's response
    if "repeat" in last_message or "say that again" in last_message:
        return "repeat"
    if "explain" in last_message or "what do you mean" in last_message:
        return "explain"
    
    # Use existing logic for follow-up decisions
    prompt = PromptTemplate(
        input_variables=["chat_history", "followup_count", "max_followups"],
        template="""You are an interviewer analyzing a candidate's response.
        Current follow-up count: {followup_count}/{max_followups}
        Based on the conversation history, should we ask a follow-up?
        Respond with ONLY "next_question" or "followup"
        
        Chat History:
        {chat_history}
        """
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "chat_history": state["messages"][-4:],
        "followup_count": state["followup_count"],
        "max_followups": state["max_followups"]
    })
    
    return "followup" if "followup" in response.content.lower() else "next_question"

# Create the enhanced workflow
workflow = StateGraph(InterviewState)

# Add all nodes
workflow.add_node("ask_main_question", ask_main_question_node)
workflow.add_node("generate_followup", generate_followup_node)
workflow.add_node("move_to_next_question", move_to_next_question_node)
workflow.add_node("end_interview", end_interview_node)
workflow.add_node("user_input", user_input_node)  # Added for interruption

# Add new nodes
workflow.add_node("repeat_question", repeat_question_node)
workflow.add_node("explain_question", explain_question_node)

# Set entry point
workflow.set_entry_point("ask_main_question")

# Rewired edges to support interruption
# workflow.add_edge("ask_main_question", "user_input")

# Add this instead:
workflow.add_conditional_edges(
    "ask_main_question",
    lambda state: "end_interview" if state.get("interview_complete", False) else "user_input",
    {
        "end_interview": "end_interview",
        "user_input": "user_input"
    }
)
workflow.add_edge("generate_followup", "user_input")
workflow.add_edge("move_to_next_question", "ask_main_question")

# Update conditional edges
workflow.add_conditional_edges(
    "user_input",
    determine_next_step,
    {
        "followup": "generate_followup",
        "next_question": "move_to_next_question",
        "end_interview": "end_interview",
        "repeat": "repeat_question",
        "explain": "explain_question"
    }
)

# Add edges back to user input
workflow.add_edge("repeat_question", "user_input")
workflow.add_edge("explain_question", "user_input")

workflow.add_edge("end_interview", END)

# Compile with checkpointer and interruption
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer, interrupt_before=["user_input"])
