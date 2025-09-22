# User Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   FRONTEND                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────────────┐
│   User Input    │───▶│   sendMessage()  │───▶│     POST /api/query             │
│   script.js:45  │    │   script.js:45   │    │     { query, session_id }      │
└─────────────────┘    └──────────────────┘    └─────────────────────────────────┘
                                                                │
                                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   BACKEND                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐    ┌─────────────────────────────────────────┐
│     FastAPI Endpoint            │───▶│           RAG System                    │
│     /api/query (app.py:56)      │    │      rag_system.query()                 │
│   • Create session if needed    │    │      (rag_system.py:102)                │
│   • Call rag_system.query()     │    └─────────────────────────────────────────┘
└─────────────────────────────────┘                           │
                                                               ▼
                                    ┌─────────────────────────────────────────────┐
                                    │         Session Manager                     │
                                    │    session_manager.py:42                   │
                                    │   • Get conversation history               │
                                    │   • Format previous messages               │
                                    └─────────────────────────────────────────────┘
                                                               │
                                                               ▼
                                    ┌─────────────────────────────────────────────┐
                                    │          AI Generator                       │
                                    │     ai_generator.py:43                     │
                                    │   • Build system prompt + history          │
                                    │   • Call Claude API with tools             │
                                    └─────────────────────────────────────────────┘
                                                               │
                                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLAUDE API DECISION                               │
└─────────────────────────────────────────────────────────────────────────────────┘

                          ┌─────────────────┐    ┌─────────────────┐
                          │  General Query  │    │  Course Query   │
                          │  (no tool use)  │    │  (uses tools)   │
                          └─────────────────┘    └─────────────────┘
                                    │                       │
                                    ▼                       ▼
                          ┌─────────────────┐    ┌─────────────────────────────────┐
                          │ Direct Response │    │      Tool Execution            │
                          │                 │    │   search_tools.py:52           │
                          └─────────────────┘    │ ┌─────────────────────────────┐ │
                                    │            │ │   CourseSearchTool          │ │
                                    │            │ │ • vector_store.search()     │ │
                                    │            │ │ • ChromaDB semantic search  │ │
                                    │            │ │ • Format results + sources  │ │
                                    │            │ └─────────────────────────────┘ │
                                    │            └─────────────────────────────────┘
                                    │                       │
                                    │                       ▼
                                    │            ┌─────────────────────────────────┐
                                    │            │    Second Claude API Call       │
                                    │            │   ai_generator.py:89            │
                                    │            │ • Add tool results to messages  │
                                    │            │ • Generate final response       │
                                    │            └─────────────────────────────────┘
                                    │                       │
                                    └───────────┬───────────┘
                                                ▼
                                    ┌─────────────────────────────────────────────┐
                                    │         Response Assembly                   │
                                    │      rag_system.py:129                     │
                                    │   • Get sources from tool_manager          │
                                    │   • Update conversation history            │
                                    │   • Return (response, sources)             │
                                    └─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────────────┐
│     FastAPI Response            │◀───│         JSON Response                   │
│     app.py:66                   │    │   {                                     │
│   • Wrap in QueryResponse       │    │     "answer": "...",                   │
│   • Return to frontend          │    │     "sources": [...],                  │
└─────────────────────────────────┘    │     "session_id": "..."                │
                │                      │   }                                     │
                ▼                      └─────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   FRONTEND                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐    ┌─────────────────────────────────────────┐
│      Display Response           │◀───│         Process Response               │
│     script.js:113               │    │        script.js:76                    │
│   • Convert markdown to HTML    │    │   • Remove loading animation           │
│   • Add collapsible sources     │    │   • Update session ID                  │
│   • Scroll to bottom            │    │   • Re-enable input                    │
└─────────────────────────────────┘    └─────────────────────────────────────────┘

KEY COMPONENTS:
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  ChromaDB       │  │ Session Manager │  │  Tool Manager   │  │  Vector Store   │
│  (vector_store) │  │ (conversation)  │  │  (search tools) │  │  (embeddings)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘

DECISION POINTS:
• Claude decides whether to search based on query content
• One search maximum per query to prevent tool loops
• Sources tracked and reset after each response
• Session management enables conversational context
```