from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory

from ..core.rag_agent import RAGAgent
from ..core.config import MAX_MESSAGES
from ..models import ChatRequest, UpdateDocumentsRequest, ChatResponse, UpdateDocumentsResponse

router = APIRouter()

class AgentState(MessagesState, total=False):
    """State definition for the agent."""

# Initialize RAG agent and memory
rag_agent = RAGAgent()
chat_history_memory = InMemoryChatMessageHistory()
memory = MemorySaver()

def add_message_with_limit(chat_history_memory, message):
    chat_history_memory.add_message(message)
    if len(chat_history_memory.messages) > MAX_MESSAGES:
        chat_history_memory.messages = chat_history_memory.messages[-MAX_MESSAGES:]

async def arag_node(state: AgentState, config: RunnableConfig) -> AgentState:
    if not state.get("messages"):
        return {"messages": []}
    
    messages = state["messages"]
    
    for msg in messages:
        add_message_with_limit(chat_history_memory, msg)

    chat_history = chat_history_memory.messages
    formatted_chat_history = "\n".join([
        f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" 
        for msg in chat_history
    ])
    
    query = messages[-1].content
    
    try:
        if not rag_agent.vectorstore:
            try:
                rag_agent.load_existing_vectorstore()
            except ValueError:
                return {"messages": messages + [AIMessage(content="No documents have been indexed yet. Please update documents first.")]}
        
        qa_chain = rag_agent.get_qa_chain(formatted_chat_history)
        result = qa_chain.invoke({"input": query, "chat_history": formatted_chat_history})
        answer = result['answer']
        
        chat_history_memory.add_message(AIMessage(content=answer))
        
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        
        return {"messages": messages + [AIMessage(content=answer)]}
    
    except Exception as e:
        return {"messages": messages + [AIMessage(content=str(e))]}

# Create workflow
workflow = StateGraph(AgentState)
workflow.add_node("rag", arag_node)
workflow.set_entry_point("rag")
workflow.add_edge("rag", END)
graph = workflow.compile(checkpointer=memory)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        thread_id = request.thread_id
        config = {"configurable": {"thread_id": thread_id}}
        
        current_state = None
        try:
            state = graph.get_state(config)
            if state is not None and "messages" in state.values:
                current_state = {"messages": state.values["messages"]}
                print(f"Retrieved {len(current_state['messages'])} messages from graph memory for thread {thread_id}")
        except Exception as e:
            print(f"Error retrieving state from graph memory: {str(e)}")
        
        if current_state is None:
            current_state = {"messages": []}
        
        current_state["messages"].append(HumanMessage(content=request.question))
        
        result = await graph.ainvoke(current_state, config=config)
        
        if result["messages"] and len(result["messages"]) > 0:
            ai_message = result["messages"][-1]
            answer = ai_message.content
        else:
            answer = "No response generated"
            
        print(f"Thread {thread_id} now has {len(result['messages'])} messages")
        
        return ChatResponse(answer=answer, thread_id=thread_id)
            
    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/update-documents", response_model=UpdateDocumentsResponse)
async def update_documents(request: UpdateDocumentsRequest):
    try:
        result = rag_agent.update_documents(request.file_paths)
        return UpdateDocumentsResponse(message=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}") 