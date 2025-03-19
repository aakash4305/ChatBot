import gradio as gr
import json
import argparse
import time
from openai import OpenAI
from rag import rag_search, init_rag
from difflib import SequenceMatcher
import os
from langchain.document_loaders import PyPDFLoader

# Argument parser setup
parser = argparse.ArgumentParser(description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url', type=str, default='http://localhost:8000/v1', help='Model URL')
parser.add_argument('-m', '--model', type=str, required=True, help='Model name for the chatbot')
parser.add_argument('--temp', type=float, default=0.8, help='Temperature for text generation')
parser.add_argument('--stop-token-ids', type=str, default='', help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--pdf-file", type=str, help="Path to the PDF file (default: relative path to SlamonetalSCIENCE1987.pdf)")

# Parse arguments
args = parser.parse_args()

# Set default PDF path if not provided
if not args.pdf_file:
    # Use relative path based on the location of main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.pdf_file = os.path.join(script_dir, "SlamonetalSCIENCE1987.pdf")

# Initialize RAG components with the correct PDF path
mc, encoder, dict_list = init_rag(args.pdf_file)

# Set OpenAI API configurations
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create OpenAI client
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

# Function to load a single PDF file
def load_pdf(file_path):
    """Loads the PDF file using PyPDFLoader."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {file_path}")
    return docs

# Function to compute similarity
def is_similar(answer1: str, answer2: str, threshold=0.8) -> bool:
    """Checks if two responses are similar using fuzzy matching."""
    return SequenceMatcher(None, answer1.lower().strip(), answer2.lower().strip()).ratio() > threshold

# Function to collect chatbot responses
def collect_chatbot_responses(filename: str = "chatbot_responses.json"):
    """Collects chatbot responses with response times and saves them."""
    responses = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            responses = json.load(f)

    print("Enter questions (type 'done' to finish):")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "done":
            break

        start_time = time.time()
        response = get_chatbot_response(question)
        response_time = round((time.time() - start_time) * 1000, 2)

        responses.append({"question": question, "response": response, "response_time_ms": response_time})
        print(f"Chatbot Response: {response} (Response Time: {response_time} ms)\n")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4)

    print(f"Chatbot responses saved to {filename}")
    return responses

# Function to establish ground truths
def establish_ground_truths(chatbot_responses, filename="ground_truths.json"):
    """Manually inputs ground truths for chatbot responses."""
    ground_truths = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            ground_truths = json.load(f)

    existing_questions = {gt["question"] for gt in ground_truths}

    print("\nEnter ground truths for each question:")
    for cr in chatbot_responses:
        question = cr["question"]
        if question in existing_questions:
            continue
        print(f"Question: {question}")
        print(f"Chatbot Response: {cr['response']}")
        ground_truth = input("Ground Truth Answer: ").strip()
        ground_truths.append({"question": question, "ground_truth": ground_truth})

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ground_truths, f, indent=4)

    print(f"Ground truths saved to {filename}")
    return ground_truths

# Function to compute accuracy
def compute_accuracy(chatbot_file="chatbot_responses.json", ground_truth_file="ground_truths.json"):
    """Computes chatbot accuracy by comparing responses to ground truths."""
    with open(chatbot_file, "r", encoding="utf-8") as f:
        chatbot_responses = json.load(f)

    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truths = json.load(f)

    chatbot_dict = {cr["question"]: (cr["response"], cr["response_time_ms"]) for cr in chatbot_responses}
    ground_truth_dict = {gt["question"]: gt["ground_truth"] for gt in ground_truths}

    common_questions = set(chatbot_dict.keys()) & set(ground_truth_dict.keys())
    if not common_questions:
        print("No matching questions found.")
        return

    total_questions = len(common_questions)
    correct_answers = 0
    total_response_time = 0

    print("\nComparison Results:")
    print("-" * 100)
    for question in common_questions:
        chatbot_answer, response_time = chatbot_dict[question]
        ground_truth_answer = ground_truth_dict[question]

        is_correct = is_similar(ground_truth_answer, chatbot_answer)
        correct_answers += 1 if is_correct else 0
        total_response_time += response_time

        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth_answer}")
        print(f"Chatbot Response: {chatbot_answer}")
        print(f"Correct: {is_correct}")
        print(f"Response Time: {response_time} ms")
        print("-" * 100)

    accuracy = (correct_answers / total_questions) * 100
    avg_response_time = total_response_time / total_questions

    print(f"\nTotal Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Response Time: {avg_response_time:.2f} ms")

# Chatbot response function
def predict(message, history):
    start_time = time.time()
    results = rag_search(mc, encoder, dict_list, message)

    retrieved_lines = [(res["entity"]["chunk"], res["distance"]) for res in results[0] if "chunk" in res["entity"]]
    context = "\n".join([line[0] for line in retrieved_lines])

    SYSTEM_PROMPT = """
    You are an AI assistant answering based only on the provided context. If the context lacks information, state that clearly.
    """
    USER_PROMPT = f"""
    <context>
    {context}
    </context>

    Answer based strictly on this context. If the information is insufficient, say so.

    <question>
    {message}
    </question>
    """

    history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    history.append({"role": "user", "content": USER_PROMPT})

    stream = client.chat.completions.create(
        model=args.model,
        messages=history,
        temperature=args.temp,
        stream=True,
        extra_body={
            'repetition_penalty': 1,
            'stop_token_ids': [int(id.strip()) for id in args.stop_token_ids.split(',') if id.strip()] if args.stop_token_ids else []
        })

    end_time = time.time()
    response_time = f"Response time: {end_time - start_time:.4f} s"

    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message, response_time

# Gradio UI
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

# Launch UI
if __name__ == "__main__":
    demo = ui()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=True)
