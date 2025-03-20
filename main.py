import gradio as gr
import json
import argparse
import time
from openai import OpenAI
from rag import rag_search, init_rag
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
parser.add_argument("--use-llm-eval", action="store_true", help="Use LLM for evaluation")

# Parse arguments
args = parser.parse_args()

# Set default PDF path if not provided
if not args.pdf_file:
    # Use relative path based on the location of main.py
    script_dir = os.path.dirname(os.path.abspath(_file_))
    args.pdf_file = os.path.join(script_dir, "SlamonetalSCIENCE1987.pdf")

# Initialize RAG components with the correct PDF path
mc, encoder, dict_list = init_rag(args.pdf_file)

# Set OpenAI API configurations
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create OpenAI client
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

# Definition of generic greetings and responses
GENERIC_GREETINGS = [
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
    "good evening", "howdy", "what's up", "how are you", "how's it going",
    "how do you do", "how have you been", "how's your day"
]

GREETING_RESPONSES = {
    "hello": "Hello! I'm your PDF assistant. I can help you find information from the document. What would you like to know?",
    "hi": "Hi there! I'm ready to help you with information from the PDF. What are you looking for?",
    "hey": "Hey! I'm your PDF assistant. How can I help you today?",
    "how are you": "I'm functioning well and ready to assist you with information from the PDF. What would you like to know?",
    "default": "Welcome! I'm your PDF assistant. I can answer questions based on the document content. What would you like to know?"
}

# Helper function to make objects JSON serializable
def make_json_serializable(obj):
    """
    Recursively convert objects to JSON serializable types.
    Handles NumPy types and nested structures.
    """
    if hasattr(obj, 'item'):  # For numpy types
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# Function to load a single PDF file
def load_pdf(file_path):
    """Loads the PDF file using PyPDFLoader."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {file_path}")
    return docs

# Function to evaluate response using LLM
def evaluate_with_llm(question, ground_truth, response, eval_client, model_name):
    """
    Uses an LLM to evaluate if the response correctly answers the question
    compared to the ground truth.
    
    Args:
        question (str): The original question
        ground_truth (str): The ground truth answer
        response (str): The chatbot's response
        eval_client: The LLM client
        model_name (str): The model to use for evaluation
        
    Returns:
        bool: True if the response is judged correct, False otherwise
        float: Confidence score (0-1)
    """
    prompt = f"""
    Question: {question}
    
    Ground Truth Answer: {ground_truth}
    
    Model Response: {response}
    
    Evaluate if the Model Response correctly answers the question compared to the Ground Truth.
    Consider:
    1. Factual accuracy
    2. Completeness
    3. Relevance to the question
    
    First, provide a brief explanation of your evaluation.
    Then, rate the response on a scale of 0-10 where 10 means perfect match in meaning and 0 means completely incorrect.
    Finally, provide a yes/no judgment on whether the response is correct.
    
    Format your response as:
    Explanation: [your explanation]
    Score: [0-10]
    Correct: [yes/no]
    """
    
    try:
        result = eval_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        
        response_text = result.choices[0].message.content
        
        # Extract score and correct/incorrect judgment
        score_line = [line for line in response_text.split('\n') if line.startswith('Score:')]
        correct_line = [line for line in response_text.split('\n') if line.startswith('Correct:')]
        
        score = 0
        correct = False
        
        if score_line:
            try:
                score = float(score_line[0].split(':')[1].strip()) / 10.0  # Normalize to 0-1
            except:
                score = 0
                
        if correct_line:
            correct = 'yes' in correct_line[0].lower()
        
        return correct, score
    except Exception as e:
        print(f"Error in LLM evaluation: {str(e)}")
        return False, 0.0

# Function to check if a message is a generic greeting
def is_generic_greeting(message):
    """Check if the message is a generic greeting."""
    message = message.lower().strip()
    return any(greeting in message for greeting in GENERIC_GREETINGS)

# Function to get response for generic greetings
def get_greeting_response(message):
    """Return an appropriate response for a generic greeting."""
    message = message.lower().strip()
    
    for greeting in GENERIC_GREETINGS:
        if greeting in message:
            return GREETING_RESPONSES.get(greeting, GREETING_RESPONSES["default"])
    
    return GREETING_RESPONSES["default"]

# Function to collect chatbot responses
def collect_chatbot_responses(filename: str = "chatbot_responses.json"):
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
        # Use predict function instead of get_chatbot_response
        response, _ = next(predict(question, []))  # Simplified call to get first response
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

# Function to compute accuracy metrics (LLM evaluation only)
def compute_accuracy(chatbot_file="chatbot_responses.json", ground_truth_file="ground_truths.json"):
    """Computes chatbot accuracy using LLM evaluation."""
    print("\nComputing accuracy with LLM evaluation...")
    
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
    
    # Initialize metrics
    metrics = {
        "llm_evaluation": {"correct": 0, "scores": []}
    }
    
    total_response_time = 0

    print("\nEvaluation Results:")
    print("-" * 100)
    
    for question in common_questions:
        chatbot_answer, response_time = chatbot_dict[question]
        ground_truth_answer = ground_truth_dict[question]
        total_response_time += response_time
        
        # Use LLM evaluation
        if args.use_llm_eval:
            try:
                is_correct_llm, llm_score = evaluate_with_llm(
                    question, ground_truth_answer, chatbot_answer, client, args.model
                )
                # Ensure it's a native Python float
                llm_score = float(llm_score)
                metrics["llm_evaluation"]["scores"].append(llm_score)
                metrics["llm_evaluation"]["correct"] += int(is_correct_llm)
                
                # Print results for this question
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth_answer}")
                print(f"Chatbot Response: {chatbot_answer}")
                print(f"LLM Evaluation: {'Correct' if is_correct_llm else 'Incorrect'} (Score: {llm_score:.2f})")
                print(f"Response Time: {response_time} ms")
                print("-" * 100)
            except Exception as e:
                print(f"LLM evaluation error for question '{question}': {str(e)}")
        else:
            print("LLM evaluation is disabled. Use --use-llm-eval flag to enable it.")
            break

    # Calculate overall metrics
    avg_response_time = total_response_time / total_questions
    
    # Print overall results
    print(f"\nSUMMARY METRICS:")
    print(f"Total Questions: {total_questions}")
    
    if args.use_llm_eval and metrics["llm_evaluation"]["scores"]:
        avg_llm_score = sum(metrics["llm_evaluation"]["scores"]) / len(metrics["llm_evaluation"]["scores"])
        print(f"LLM Evaluation Accuracy: {metrics['llm_evaluation']['correct'] / total_questions * 100:.2f}%")
        print(f"Average LLM Score: {avg_llm_score:.2f}")
    
    print(f"Average Response Time: {avg_response_time:.2f} ms")
    
    # Create metrics_results dictionary
    metrics_results = {
        "total_questions": total_questions,
        "response_time": {
            "average_ms": avg_response_time,
        }
    }
    
    if args.use_llm_eval and metrics["llm_evaluation"]["scores"]:
        metrics_results["llm_evaluation"] = {
            "average": avg_llm_score,
            "accuracy": metrics['llm_evaluation']['correct'] / total_questions,
            "scores": metrics["llm_evaluation"]["scores"]
        }
    
    # Make sure all values are JSON serializable
    metrics_results = make_json_serializable(metrics_results)
    
    with open("evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_results, f, indent=4)
    
    print(f"\nDetailed evaluation metrics saved to evaluation_metrics.json")
    
    return metrics_results

# Chatbot response function
def predict(message, history):
    start_time = time.time()
    
    # Check if the message is a generic greeting
    if is_generic_greeting(message):
        greeting_response = get_greeting_response(message)
        end_time = time.time()
        response_time = f"Response time: {end_time - start_time:.4f} s"
        yield greeting_response, response_time
        return
    
    # For non-greeting messages, use RAG
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

# Main function to run everything in sequence
def main():
    print(f"Starting chatbot evaluation for PDF: {args.pdf_file}")
    print(f"JSON files are saved in the current working directory: {os.getcwd()}")
    print(f"Ground truth file: {os.path.join(os.getcwd(), 'ground_truths.json')}")
    print(f"Chatbot responses file: {os.path.join(os.getcwd(), 'chatbot_responses.json')}")
    print(f"Evaluation metrics file: {os.path.join(os.getcwd(), 'evaluation_metrics.json')}")
    
    # Add a try-except block to handle the computation phase with proper error handling
    try:
        # Step 1: Collect chatbot responses and create chatbot_responses.json
        chatbot_responses = collect_chatbot_responses()

        # Step 2: Establish ground truths and create ground_truths.json
        ground_truths = establish_ground_truths(chatbot_responses)

        # Step 3: Compute and display accuracy scores using LLM evaluation only
        metrics = compute_accuracy()
    except Exception as e:
        print(f"Error during evaluation setup: {str(e)}")
        print("Continuing to Gradio UI despite evaluation error.")
    
    # Step 4: Launch the Gradio interface
    try:
        demo = ui()
        demo.queue().launch(server_name=args.host, server_port=args.port, share=True)
    except Exception as e:
        print(f"Error launching Gradio UI: {str(e)}")

if _name_ == "_main_":
    main()
