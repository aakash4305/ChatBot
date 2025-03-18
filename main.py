import gradio as gr
import json
import argparse
import time
from openai import OpenAI
from rag import rag_search, init_rag

mc, encoder, dict_list = init_rag()

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8000/v1',
                    help='Model URL')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=True,
                    help='Model name for the chatbot')
parser.add_argument('--temp',
                    type=float,
                    default=0.8,
                    help='Temperature for text generation')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def predict(message, history):
    start_time = time.time()
    results = rag_search(mc, encoder, dict_list, message)
    retrieved_lines_with_distances = []
    for res in results[0]:
        if "chunk" in res["entity"]:
            retrieved_lines_with_distances.append((res["entity"]["chunk"], res["distance"]))
    print(json.dumps(retrieved_lines_with_distances, indent=4))
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    SYSTEM_PROMPT = """
    You are a helpful, cautious, and knowledgeable AI assistant specialized in answering questions strictly based on the provided contextual passages. 
    Follow these guidelines:
    1. Use only the information provided within the <context> tags to construct your answer.
    2. If the context does not clearly address the question, explicitly state that the provided context is insufficient to answer the query.
    3. For general greetings or simple questions (e.g., 'hello'), respond in a friendly and natural manner without relying on the context.
    4. Do not assume or hallucinate additional information beyond the context.
    5. If the context appears irrelevant to the question, ask for clarification rather than generating potentially incorrect information.
    """
    USER_PROMPT = f"""
    Below is the contextual information extracted for your query:
    <context>
    {context}
    </context>

    Please answer the following question based strictly on the provided context. If the context does not contain sufficient information, indicate that you cannot determine a complete answer rather than guessing.

    <question>
    {message}
    </question>
    """

    # Convert chat history to OpenAI format
    history_openai_format = [{
        "role": "user",
        "content": USER_PROMPT
    }]
    history.insert(0, {
        "role": "system",
        "content": SYSTEM_PROMPT
    })
    history.extend(history_openai_format)
    print("Chat history:")
    print(json.dumps(history, indent=4))
    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=history,  # Chat history
        temperature=args.temp,  # Temperature for text generation
        stream=True,  # Stream response
        extra_body={
            'repetition_penalty':
            1,
            'stop_token_ids': [
                int(id.strip()) for id in args.stop_token_ids.split(',')
                if id.strip()
            ] if args.stop_token_ids else []
        })

    end_time = time.time()
    response_time = f"Response time: {end_time - start_time:.4f} s"
    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message, response_time

def ui():
    with gr.Blocks() as demo:
        with gr.Column():
            response_time = gr.Label(label="Response Time")
            gr.ChatInterface(
                fn=predict, 
                type="messages",
                additional_outputs=[response_time],
            )
    return demo


if __name__ == "__main__":
    demo = ui()
    demo.queue().launch(server_name=None,
                        server_port=8001,
                        share=True)





