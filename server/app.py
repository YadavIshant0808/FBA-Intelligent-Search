from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI #Use Gemini models
import os
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Depends, HTTPException, Header, status
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

load_dotenv()

# App version - update on new version
APP_VERSION = "1.0.0"
DEPLOYMENT_TIME = datetime.utcnow().isoformat()

# Adding Authorisation Check Point
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

async def verify_admin_api_key(authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header."
        )

    token = authorization.split(" ")[1]
    if token != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key."
        )

# Initialize memory saver for checkpointing
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results=4)
tools = [search_tool]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools=tools)

async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [result]}

async def tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else: 
        return END

async def tool_node(state):
    """Custom tool node that handles tool calls from the LLM."""
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name == "tavily_search_results_json":
            search_results = await search_tool.ainvoke(tool_args)
            tool_message = ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)

    return {"messages": tool_messages}

graph_builder = StateGraph(State)
graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)

app = FastAPI(
    title="FBA Intelligent Search",
    version=APP_VERSION,
    description="AI-powered search engine with conversational capabilities"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    expose_headers=["Content-Type"], 
)

def serialise_ai_message_chunk(chunk): 
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )

async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None

    if is_new_conversation:
        new_checkpoint_id = str(uuid4())
        config = {"configurable": {"thread_id": new_checkpoint_id}}

        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {"configurable": {"thread_id": checkpoint_id}}
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

    async for event in events:
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
            if isinstance(chunk_content, list):
                chunk_content = " ".join(str(item) for item in chunk_content)
            safe_content = str(chunk_content).replace("'", "\\'").replace("\n", "\\n")
            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"

        elif event_type == "on_chat_model_end":
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "tavily_search_results_json"]

            if search_calls:
                search_query = search_calls[0]["args"].get("query", "")
                safe_query = search_query.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
                yield f"data: {{\"type\": \"search_start\", \"query\": \"{safe_query}\"}}\n\n"

        elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
            output = event["data"]["output"]

            if isinstance(output, list):
                urls = []
                for item in output:
                    if isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])

                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\", \"urls\": {urls_json}}}\n\n"

    yield f"data: {{\"type\": \"end\"}}\n\n"

#Public Endpoint
@app.get("/health")
async def health_check():
    """
    Public health check endpoint for Cloud Run.
    """
    return JSONResponse({
        "status": "healthy",
        "service": "FBA Dev AI Search Engine",
        "version": APP_VERSION,
        "deployed_at": DEPLOYMENT_TIME,
        "timestamp": datetime.utcnow().isoformat()
    })
    
#Authenticated Endpoints
@app.get("/")
async def root(_auth=Depends(verify_admin_api_key)):
    return JSONResponse({
        "message": "Welcome to FBA Dev AI Search Engine. YOU Successfully authenticated",
        "version": APP_VERSION,
        "endpoints": {
            "health": "/health",
            "chat": "/chat_stream/{message}",
            "docs": "/docs"
        }
    })

@app.get("/chat_stream/{message}")
async def chat_stream(
    message: str, 
    checkpoint_id: Optional[str] = Query(None), 
    _auth=Depends(verify_admin_api_key)
):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), 
        media_type="text/event-stream"
        )
