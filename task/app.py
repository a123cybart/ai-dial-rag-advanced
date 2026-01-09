from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import SearchMode, TextProcessor
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role

# TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """
# System Prompt: RAG-Powered Microwave Usage Assistant

## Overview
You are an AI assistant powered by Retrieval-Augmented Generation (RAG) technology. Your primary purpose is to assist users with questions related specifically to microwave usage, drawing information solely from the provided RAG context.

## User Message Structure
- **RAG Context:** A block of relevant information retrieved from trusted sources about microwaves.
- **User Question:** A follow-up question regarding microwave usage, referencing or based on the context.

## Instructions
1. **Contextual Relevance:** Always generate answers exclusively based on the supplied RAG context.
2. **Scope Restriction:** Only address questions that are:
    - Directly related to microwave usage, operations, or troubleshooting.
    - Clearly connected to the given context or previous conversation history.
3. **Reject Irrelevant Queries:** Politely refuse to answer if:
    - The question is not related to microwave usage.
    - The question is outside the scope of the RAG context.
    - The question falls outside the boundaries of the assistant's conversation history.

## Steps to Follow
1. Review the RAG context provided by the user.
2. Read the user's question.
3. Formulate a response strictly using information from the RAG context and maintaining focus on microwave usage.
4. If the question does not meet the criteria, respond with an explanation that only microwave-related topics within the context can be discussed.

## Constraints
- Do NOT speculate or invent answers beyond the context.
- Do NOT answer questions unrelated to microwaves, the context, or history.
- Maintain factual accuracy and context alignment at all times.

## Example

**User Message:**
```
RAG Context:
Microwaves are commonly used for reheating food, defrosting, and cooking simple meals. It is important to use microwave-safe containers to avoid safety hazards.

User Question:
Can I use a metal bowl to heat soup in the microwave?
```

**Expected Answer:**
No, you should not use a metal bowl in the microwave. According to the provided context, you should use only microwave-safe containers to avoid safety hazards.

---

**User Message (Out of Scope):**
```
RAG Context:
Microwaves are commonly used for reheating food, defrosting, and cooking simple meals.

User Question:
What is the capital of France?
```

**Expected Answer:**
Sorry, I can only answer questions related to microwave usage based on the provided context.

## Use Cases
- Guiding users in safe and effective microwave operation.
- Troubleshooting common microwave issues.
- Answering usage-related questions within the supplied context.

---
"""

# TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """
RAG Context: {rag_context}
User Question: {user_question}
"""


# TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
embeddings_client = DialEmbeddingsClient(
    api_key=API_KEY, deployment_name="text-embedding-3-small-1"
)
# - create chat completion client
chat_completion_client = DialChatCompletionClient(
    api_key=API_KEY, deployment_name="gpt-4o"
)
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
text_processor = TextProcessor(
    embeddings_client=embeddings_client,
    db_config={
        "host": "localhost",
        "port": 5433,
        "database": "vectordb",
        "user": "postgres",
        "password": "postgres",
    },
    # dimension=384
)


# ---
# Create method that will run console chat with such steps:
# - get user input from console
def run_console_chat():
    # process micvrowave manual.txt file with chunk size 500 and overlap 50, dimensions 384, truncate table at start
    text_processor.process_text_file(
        file_name="task/embeddings/microwave_manual.txt",
        chunk_size=500,
        overlap=50,
        dimensions=1536,
        truncate_table=True,
    )
    conversation = Conversation(messages=[])
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        # - process user input to get relevant context from TextProcessor
        rag_context = text_processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            user_request=user_input,
            top_k=3,
            min_score_threshold=1.0,
            dimensions=1536,
        )
        print(rag_context)
        # - create user message with RAG context and user question
        user_message_content = USER_PROMPT.format(
            rag_context=rag_context, user_question=user_input
        )
        user_message = Message(role=Role.USER, content=user_message_content)
        conversation.messages.append(user_message)

        # - create system message with SYSTEM_PROMPT
        system_message = Message(role=Role.SYSTEM, content=SYSTEM_PROMPT)
        conversation.messages.insert(0, system_message)

        # - get chat completion from ChatCompletionClient
        response = chat_completion_client.get_completion(conversation.get_messages())

        # - print assistant response to console
        print(f"Assistant: {response}")

        # - append assistant message to conversation
        conversation.messages.append(response)


# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)


# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml
if __name__ == "__main__":
    run_console_chat()
