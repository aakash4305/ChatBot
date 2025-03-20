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
parser.add_argument('--temp', type=float, default=0.2, help='Temperature for text generation')  
parser.add_argument('--stop-token-ids', type=str, default='', help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--pdf-file", type=str, help="Path to the PDF file (default: relative path to SlamonetalSCIENCE1987.pdf)")
parser.add_argument("--use-llm-eval", action="store_true", help="Use LLM for evaluation")
# Add new arguments for JSON file management
parser.add_argument("--clear-responses", action="store_true", help="Clear chatbot responses file before starting")
parser.add_argument("--clear-ground-truths", action="store_true", help="Clear ground truths file before starting")
parser.add_argument("--clear-metrics", action="store_true", help="Clear evaluation metrics file before starting")
parser.add_argument("--export-json", type=str, help="Export all JSON data to a specified directory")

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

# Add the find_available_port function here
def find_available_port(start_port, max_attempts=100):
    """Find an available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                print(f"Found available port: {port}")
                return port
        except OSError:
            print(f"Port {port} is already in use, trying next port...")
            continue
    print(f"Could not find an available port after {max_attempts} attempts")
    return None

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

# Constants for file paths
CHATBOT_RESPONSES_FILE = "chatbot_responses.json"
GROUND_TRUTHS_FILE = "ground_truths.json"
EVALUATION_METRICS_FILE = "evaluation_metrics.json"


def read_json_file(filename):
    """Read a JSON file and return its contents"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File {filename} not found. Returning empty list.")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filename}. Returning empty list.")
        return []

def write_json_file(filename, data):
    """Write data to a JSON file"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error writing to {filename}: {str(e)}")
        return False

def clear_json_file(filename, empty_structure="list"):
    """Clear a JSON file to either an empty list [] or empty object {}"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            if empty_structure == "list":
                json.dump([], f)
            else:
                json.dump({}, f)
        print(f"File {filename} has been cleared.")
        return True
    except Exception as e:
        print(f"Error clearing {filename}: {str(e)}")
        return False

def remove_entries(filename, filter_func):
    """Remove entries from JSON file based on a filter function"""
    data = read_json_file(filename)
    if not data:
        return False
    
    # Filter out entries
    filtered_data = [item for item in data if not filter_func(item)]
    
    # Write back filtered data
    result = write_json_file(filename, filtered_data)
    if result:
        print(f"Removed {len(data) - len(filtered_data)} entries from {filename}")
    return result

def add_entry(filename, new_entry):
    """Add a new entry to a JSON file"""
    data = read_json_file(filename)
    data.append(new_entry)
    result = write_json_file(filename, data)
    if result:
        print(f"Added new entry to {filename}")
    return result

def update_entry(filename, match_key, match_value, update_dict):
    """Update entries where entry[match_key] == match_value"""
    data = read_json_file(filename)
    updates = 0
    
    for item in data:
        if item.get(match_key) == match_value:
            item.update(update_dict)
            updates += 1
    
    result = write_json_file(filename, data)
    if result:
        print(f"Updated {updates} entries in {filename}")
    return updates > 0

def get_entry(filename, match_key, match_value):
    """Get an entry where entry[match_key] == match_value"""
    data = read_json_file(filename)
    for item in data:
        if item.get(match_key) == match_value:
            return item
    return None

def export_all_json(export_dir):
    """Export all JSON files to a specified directory"""
    if not os.path.exists(export_dir):
        try:
            os.makedirs(export_dir)
        except Exception as e:
            print(f"Error creating directory {export_dir}: {str(e)}")
            return False
    
    files_to_export = [
        CHATBOT_RESPONSES_FILE,
        GROUND_TRUTHS_FILE,
        EVALUATION_METRICS_FILE
    ]
    
    success = True
    for file in files_to_export:
        if os.path.exists(file):
            try:
                data = read_json_file(file)
                export_path = os.path.join(export_dir, file)
                write_json_file(export_path, data)
                print(f"Exported {file} to {export_path}")
            except Exception as e:
                print(f"Error exporting {file}: {str(e)}")
                success = False
        else:
            print(f"File {file} not found, skipping export.")
    
    return success

def merge_chatbot_responses(source_files, deduplicate=True):
    """Merge multiple chatbot response files into one"""
    merged_data = []
    seen_questions = set()
    
    for file in source_files:
        data = read_json_file(file)
        for item in data:
            question = item.get("question")
            if not deduplicate or question not in seen_questions:
                merged_data.append(item)
                if deduplicate and question:
                    seen_questions.add(question)
    
    result = write_json_file(CHATBOT_RESPONSES_FILE, merged_data)
    if result:
        print(f"Merged {len(source_files)} files into {CHATBOT_RESPONSES_FILE}")
    return result

def analyze_feedback(filename=CHATBOT_RESPONSES_FILE):
    """Analyze user feedback to identify patterns and areas for improvement"""
    data = read_json_file(filename)
    
    # Filter entries that have user ratings
    rated_responses = [item for item in data if "user_rating" in item]
    
    if not rated_responses:
        print("No user feedback found.")
        return
    
    # Calculate average rating
    avg_rating = sum(float(item["user_rating"]) for item in rated_responses) / len(rated_responses)
    
    # Group by rating
    rating_groups = {}
    for item in rated_responses:
        rating = item["user_rating"]
        if rating not in rating_groups:
            rating_groups[rating] = []
        rating_groups[rating].append(item)
    
    # Print analysis
    print(f"\nFEEDBACK ANALYSIS:")
    print(f"Total rated responses: {len(rated_responses)}")
    print(f"Average rating: {avg_rating:.2f}/5.0")
    
    # Identify low-rated responses
    if 1 in rating_groups or 2 in rating_groups:
        low_rated = rating_groups.get(1, []) + rating_groups.get(2, [])
        print(f"\nLow-rated responses ({len(low_rated)}):")
        for item in low_rated:
            print(f"Q: {item['question']}")
            print(f"A: {item['response'][:100]}..." if len(item['response']) > 100 else f"A: {item['response']}")
            print(f"Rating: {item['user_rating']}/5")
            if "user_comment" in item and item["user_comment"]:
                print(f"Comment: {item['user_comment']}")
            print("---")
    
    return {
        "average_rating": avg_rating,
        "total_rated": len(rated_responses),
        "rating_distribution": {str(k): len(v) for k, v in rating_groups.items()}
    }

def generate_kpi_report():
    """Generate a comprehensive KPI report combining all metrics"""
    # Load data
    responses = read_json_file(CHATBOT_RESPONSES_FILE)
    metrics = read_json_file(EVALUATION_METRICS_FILE)
    
    # Basic metrics
    total_queries = len(responses)
    avg_response_time = sum(r.get("response_time_ms", 0) for r in responses) / total_queries if total_queries else 0
    
    # User satisfaction metrics
    rated_responses = [r for r in responses if "user_rating" in r]
    avg_satisfaction = sum(float(r["user_rating"]) for r in rated_responses) / len(rated_responses) if rated_responses else 0
    
    # Accuracy metrics from LLM evaluation
    accuracy = metrics.get("llm_evaluation", {}).get("accuracy", 0) if metrics else 0
    
    # Compile KPI report
    kpi_report = {
        "report_date": time.strftime("%Y-%m-%d"),
        "total_queries": total_queries,
        "performance": {
            "response_time_ms": avg_response_time,
            "accuracy": accuracy * 100,  
        },
        "user_satisfaction": {
            "average_rating": avg_satisfaction,
            "total_rated": len(rated_responses),
            "rating_percentage": (len(rated_responses) / total_queries * 100) if total_queries else 0
        }
    }
    
    # Save report
    report_filename = f"kpi_report_{time.strftime('%Y%m%d')}.json"
    write_json_file(report_filename, kpi_report)
    print(f"KPI report saved to {report_filename}")
    
   
    print("\nKPI SUMMARY:")
    print(f"Total Queries: {total_queries}")
    print(f"Average Response Time: {avg_response_time:.2f} ms")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"User Satisfaction: {avg_satisfaction:.2f}/5.0 ({len(rated_responses)} ratings)")
    
    return kpi_report

# Add this function to test if the LLM connection is working
def test_llm_connection():
    """Test if the connection to the LLM server is working properly."""
    try:
        print("Testing connection to LLM server...")
        test_response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            temperature=0.2,
            max_tokens=10
        )
        
        response_content = test_response.choices[0].message.content
        print(f"LLM connection test successful. Response: {response_content}")
        return True, response_content
    except Exception as e:
        error_msg = f"Error connecting to LLM server: {str(e)}"
        print(error_msg)
        return False, error_msg

# Function to display JSON management UI
def display_json_manager():
    """Display a simple command-line UI for JSON management"""
    while True:
        print("\n===== JSON FILE MANAGEMENT =====")
        print("1. View chatbot responses")
        print("2. View ground truths")
        print("3. View evaluation metrics")
        print("4. Clear chatbot responses")
        print("5. Clear ground truths")
        print("6. Clear evaluation metrics")
        print("7. Remove specific entries from responses")
        print("8. Remove specific entries from ground truths")
        print("9. Export all JSON data")
        print("10. Import and merge JSON data")
        print("11. Generate KPI report")
        print("12. Analyze user feedback")
        print("0. Exit to main program")
        
        choice = input("\nEnter your choice (0-12): ").strip()
        
        if choice == "1":
            data = read_json_file(CHATBOT_RESPONSES_FILE)
            print(f"\nChatbot Responses ({len(data)} entries):")
            for i, entry in enumerate(data):
                print(f"{i+1}. Q: {entry.get('question')}")
                print(f"   A: {entry.get('response')[:100]}..." if len(entry.get('response', "")) > 100 else f"   A: {entry.get('response')}")
                print(f"   Time: {entry.get('response_time_ms')} ms")
                print("---")
        
        elif choice == "2":
            data = read_json_file(GROUND_TRUTHS_FILE)
            print(f"\nGround Truths ({len(data)} entries):")
            for i, entry in enumerate(data):
                print(f"{i+1}. Q: {entry.get('question')}")
                print(f"   GT: {entry.get('ground_truth')}")
                print("---")
        
        elif choice == "3":
            data = read_json_file(EVALUATION_METRICS_FILE)
            print("\nEvaluation Metrics:")
            print(json.dumps(data, indent=2))
        
        elif choice == "4":
            confirm = input("Are you sure you want to clear all chatbot responses? (y/n): ").lower()
            if confirm == 'y':
                clear_json_file(CHATBOT_RESPONSES_FILE)
        
        elif choice == "5":
            confirm = input("Are you sure you want to clear all ground truths? (y/n): ").lower()
            if confirm == 'y':
                clear_json_file(GROUND_TRUTHS_FILE)
        
        elif choice == "6":
            confirm = input("Are you sure you want to clear evaluation metrics? (y/n): ").lower()
            if confirm == 'y':
                clear_json_file(EVALUATION_METRICS_FILE)
        
        elif choice == "7":
            keyword = input("Enter keyword to filter responses (or 'all' to see all): ").strip()
            data = read_json_file(CHATBOT_RESPONSES_FILE)
            
            if keyword.lower() != 'all':
                filtered_data = [item for item in data if keyword.lower() in item.get("question", "").lower() or keyword.lower() in item.get("response", "").lower()]
            else:
                filtered_data = data
            
            if not filtered_data:
                print("No matching entries found.")
                continue
            
            print("\nMatching Entries:")
            for i, entry in enumerate(filtered_data):
                print(f"{i+1}. Q: {entry.get('question')}")
                print(f"   A: {entry.get('response')[:100]}..." if len(entry.get('response', "")) > 100 else f"   A: {entry.get('response')}")
            
            to_remove = input("\nEnter numbers to remove (comma-separated, or 'all'): ").strip()
            if to_remove.lower() == 'all':
                indices = list(range(len(filtered_data)))
            else:
                indices = [int(x.strip()) - 1 for x in to_remove.split(',') if x.strip().isdigit()]
            
            if indices:
                questions_to_remove = [filtered_data[i]["question"] for i in indices if 0 <= i < len(filtered_data)]
                remove_entries(CHATBOT_RESPONSES_FILE, lambda x: x.get("question") in questions_to_remove)
        
        elif choice == "8":
            keyword = input("Enter keyword to filter ground truths (or 'all' to see all): ").strip()
            data = read_json_file(GROUND_TRUTHS_FILE)
            
            if keyword.lower() != 'all':
                filtered_data = [item for item in data if keyword.lower() in item.get("question", "").lower() or keyword.lower() in item.get("ground_truth", "").lower()]
            else:
                filtered_data = data
            
            if not filtered_data:
                print("No matching entries found.")
                continue
            
            print("\nMatching Entries:")
            for i, entry in enumerate(filtered_data):
                print(f"{i+1}. Q: {entry.get('question')}")
                print(f"   GT: {entry.get('ground_truth')}")
            
            to_remove = input("\nEnter numbers to remove (comma-separated, or 'all'): ").strip()
            if to_remove.lower() == 'all':
                indices = list(range(len(filtered_data)))
            else:
                indices = [int(x.strip()) - 1 for x in to_remove.split(',') if x.strip().isdigit()]
            
            if indices:
                questions_to_remove = [filtered_data[i]["question"] for i in indices if 0 <= i < len(filtered_data)]
                remove_entries(GROUND_TRUTHS_FILE, lambda x: x.get("question") in questions_to_remove)
        
        elif choice == "9":
            export_dir = input("Enter export directory path: ").strip()
            export_all_json(export_dir)
        
        elif choice == "10":
            files = input("Enter paths to response files to merge (comma-separated): ").strip()
            file_list = [f.strip() for f in files.split(",") if f.strip()]
            if file_list:
                deduplicate = input("Deduplicate based on questions? (y/n): ").lower() == 'y'
                merge_chatbot_responses(file_list, deduplicate)
                
        elif choice == "11":
            generate_kpi_report()
            
        elif choice == "12":
            analyze_feedback()
        
        elif choice == "0":
            break
        
        else:
            print("Invalid choice, please try again.")

# Helper function to make objects JSON serializable
def make_json_serializable(obj):
    """
    Recursively convert objects to JSON serializable types.
    Handles NumPy types and nested structures.
    """
    if hasattr(obj, 'item'):  
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
                score = float(score_line[0].split(':')[1].strip()) / 10.0  
            except:
                score = 0
                
        if correct_line:
            correct = 'yes' in correct_line[0].lower()
        
        return correct, score
    except Exception as e:
        print(f"Error in LLM evaluation: {str(e)}")
        return False, 0.0

# Improved function to check if a message is a generic greeting
def is_generic_greeting(message):
    """Check if the message is a generic greeting."""
    message = message.lower().strip()
    
    # Check for exact matches first
    if message in GENERIC_GREETINGS:
        return True
    
    # Check if the message is just a greeting with punctuation
    message_no_punct = ''.join(c for c in message if c.isalnum() or c.isspace())
    message_words = message_no_punct.strip().split()
    
 
    if len(message_words) <= 2:
        for greeting in GENERIC_GREETINGS:
            if greeting in message_words:
                return True
    
    return False

# Function to get response for generic greetings
def get_greeting_response(message):
    """Return an appropriate response for a generic greeting."""
    message = message.lower().strip()
    
    for greeting in GENERIC_GREETINGS:
        if greeting in message:
            return GREETING_RESPONSES.get(greeting, GREETING_RESPONSES["default"])
    
    return GREETING_RESPONSES["default"]

# Function to collect chatbot responses
def collect_chatbot_responses(filename: str = CHATBOT_RESPONSES_FILE):
    # Load existing responses if available
    responses = read_json_file(filename)
    
    print("Enter questions (type 'done' to finish):")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "done":
            break

        start_time = time.time()
        # Use predict function instead of get_chatbot_response
        try:
            response, _ = next(predict(question, []))  
            response_time = round((time.time() - start_time) * 1000, 2)
            responses.append({"question": question, "response": response, "response_time_ms": response_time})
            print(f"Chatbot Response: {response} (Response Time: {response_time} ms)\n")
        except Exception as e:
            print(f"Error getting response: {str(e)}")

    # Write the responses back to file
    write_json_file(filename, responses)
    print(f"Chatbot responses saved to {filename}")
    return responses

# Function to establish ground truths
def establish_ground_truths(chatbot_responses, filename=GROUND_TRUTHS_FILE):
    """Manually inputs ground truths for chatbot responses."""
    # Load existing ground truths
    ground_truths = read_json_file(filename)
    
    # Create a set of questions that already have ground truths
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

    # Write ground truths back to file
    write_json_file(filename, ground_truths)
    print(f"Ground truths saved to {filename}")
    return ground_truths

# Function to compute accuracy metrics (LLM evaluation only)
def compute_accuracy(chatbot_file=CHATBOT_RESPONSES_FILE, ground_truth_file=GROUND_TRUTHS_FILE):
    """Computes chatbot accuracy using LLM evaluation."""
    print("\nComputing accuracy with LLM evaluation...")
    
    chatbot_responses = read_json_file(chatbot_file)
    ground_truths = read_json_file(ground_truth_file)

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
    
    # Write metrics to file
    write_json_file(EVALUATION_METRICS_FILE, metrics_results)
    print(f"\nDetailed evaluation metrics saved to {EVALUATION_METRICS_FILE}")
    
    return metrics_results


def predict(message, history):
    start_time = time.time()
    
    try: 
        
        if is_generic_greeting(message):
            greeting_response = get_greeting_response(message)
            end_time = time.time()
            response_time = f"Response time: {end_time - start_time:.4f} s"
            yield greeting_response, response_time
            return
        
        try:
            results = rag_search(mc, encoder, dict_list, message, top_k=8)  
            
            # Check if results are valid
            if not results or len(results) == 0 or len(results[0]) == 0:
                yield "I couldn't find relevant information in the document to answer your question. Please try rephrasing or asking something else.", f"Response time: {time.time() - start_time:.4f} s"
                return
                
            retrieved_lines = [(res["entity"]["chunk"], res["distance"]) for res in results[0] if "chunk" in res["entity"]]

            if not retrieved_lines:
                yield "The document doesn't contain information to answer this question. Please try a different query.", f"Response time: {time.time() - start_time:.4f} s"
                return

            # Only keep chunks with distance score below threshold (lower is better)
            retrieved_lines = [line for line in retrieved_lines if line[1] < 0.9]
            # Sort by relevance
            retrieved_lines = sorted(retrieved_lines, key=lambda x: x[1])
            # Take top results
            retrieved_lines = retrieved_lines[:6]  # Use top 6 most relevant chunks

            if not retrieved_lines:
                yield "I couldn't find relevant information in the document to answer your question. Please try rephrasing or asking something else.", f"Response time: {time.time() - start_time:.4f} s"
                return

            context = "\n".join([f"Document chunk (distance {line[1]:.4f}):\n{line[0]}\n" for line in retrieved_lines])
          
            print(f"Retrieved context length: {len(context)} characters")
            print(f"First 100 chars of context: {context[:100]}...")
            
        except Exception as rag_error:
            error_msg = f"Error during document search: {str(rag_error)}"
            print(error_msg)
            yield f"I encountered an error while searching the document: {str(rag_error)}", f"Response time: {time.time() - start_time:.4f} s"
            return

        SYSTEM_PROMPT = """
        You are an AI assistant answering questions about a scientific paper on HER-2/neu oncogene in breast cancer. 
        Use ONLY the context provided to answer questions. Be specific and direct with your answers.
        Even if the information seems incomplete, provide the best answer possible based on what's available in the context.
        Do not mention limitations of the context unless absolutely necessary.
        """

        USER_PROMPT = f"""
        <context>
        {context}
        </context>

        Based on the above context from the research paper, answer the following question concisely and accurately:

        <question>
        {message}
        </question>
        """

        # Create a new history for each query to avoid context confusion
        current_history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]

        try:
            # Use only streaming approach with timeout
            stream = client.chat.completions.create(
                model=args.model,
                messages=current_history,
                temperature=0.2,  
                stream=True,
                timeout=30,  
                extra_body={
                    'repetition_penalty': 1,
                    'stop_token_ids': [int(id.strip()) for id in args.stop_token_ids.split(',') if id.strip()] if args.stop_token_ids else []
                })

            end_time = time.time()
            response_time = f"Response time: {end_time - start_time:.4f} s"

            partial_message = ""
            for chunk in stream:
                try:
                    # Handle potential different formats from different LLM providers
                    if hasattr(chunk, 'choices'):
                        # Standard OpenAI format
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                        else:
                            content = None
                    elif isinstance(chunk, dict) and 'choices' in chunk:
                        # Dictionary format
                        content = chunk['choices'][0].get('delta', {}).get('content')
                    elif isinstance(chunk, tuple) and len(chunk) > 0:
                        # Tuple format (unexpected but handled)
                        print(f"Received chunk as tuple: {chunk}")
                        if len(chunk) > 0 and isinstance(chunk[0], str):
                            content = chunk[0]
                        else:
                            content = str(chunk)
                    else:
                        
                        print(f"Unknown chunk format: {type(chunk)} - {str(chunk)}")
                        try:
                            content = str(chunk)
                        except:
                            content = None
                    
                    if content is not None:
                        partial_message += content
                        yield partial_message, response_time
                except Exception as chunk_error:
                    error_msg = f"Error processing chunk: {str(chunk_error)}, type: {type(chunk)}"
                    print(error_msg)
                    if not partial_message:  
                        yield f"Error while generating response: {str(chunk_error)}", response_time
                    return
            
            if not partial_message:
                print("No content returned from LLM")
                yield "I couldn't generate a response based on the available information. Please try a different question.", response_time
                
        except Exception as llm_error:
            error_msg = f"Error calling language model: {str(llm_error)}"
            print(error_msg)
            yield f"I encountered an error while generating a response: {str(llm_error)}", f"Response time: {time.time() - start_time:.4f} s"
            
    except Exception as e:
       
        error_msg = f"Unexpected error in predict function: {str(e)}"
        print(error_msg)
        yield f"An unexpected error occurred: {str(e)}", f"Response time: {time.time() - start_time:.4f} s"
        

# Gradio UI with error handling
def ui():
    with gr.Blocks(css=".error-message { color: red; font-weight: bold; }") as demo:
        with gr.Column():
            response_time = gr.Label(label="Response Time")
            chatbot = gr.ChatInterface(
                fn=predict,
                type="messages",
                additional_outputs=[response_time],
            )
            
            # Add user satisfaction feedback component
            with gr.Accordion("Provide Feedback", open=False):
                feedback_score = gr.Slider(1, 5, value=3, step=1, label="Rate Response (1-5)")
                feedback_text = gr.Textbox(label="Additional Comments")
                feedback_btn = gr.Button("Submit Feedback")
                feedback_status = gr.Textbox(label="Feedback Status", interactive=False)
                
                def collect_feedback(score, comment):
                    # Get last question and response
                    try:
                        responses = read_json_file(CHATBOT_RESPONSES_FILE)
                        if responses:
                            last_response = responses[-1]
                            last_response["user_rating"] = score
                            last_response["user_comment"] = comment
                            write_json_file(CHATBOT_RESPONSES_FILE, responses)
                            return "Feedback submitted. Thank you!"
                    except Exception as e:
                        return f"Error saving feedback: {str(e)}"
                    return "No recent responses to rate."
                
                feedback_btn.click(collect_feedback, [feedback_score, feedback_text], feedback_status)
    
    return demo

def main():
    print(f"Starting chatbot evaluation for PDF: {args.pdf_file}")
    print(f"JSON files are saved in the current working directory: {os.getcwd()}")
    print(f"Ground truth file: {os.path.join(os.getcwd(), GROUND_TRUTHS_FILE)}")
    print(f"Chatbot responses file: {os.path.join(os.getcwd(), CHATBOT_RESPONSES_FILE)}")
    print(f"Evaluation metrics file: {os.path.join(os.getcwd(), EVALUATION_METRICS_FILE)}")
    
    # Handle command-line flags for clearing files
    if args.clear_responses:
        clear_json_file(CHATBOT_RESPONSES_FILE)
    if args.clear_ground_truths:
        clear_json_file(GROUND_TRUTHS_FILE)
    if args.clear_metrics:
        clear_json_file(EVALUATION_METRICS_FILE)
    
    # Handle JSON export if requested
    if args.export_json:
        export_all_json(args.export_json)
    
    # Test LLM connection before proceeding
    connection_success, connection_message = test_llm_connection()
    if not connection_success:
        print(f"ERROR: Failed to connect to LLM server. {connection_message}")
        print("Please make sure your vLLM server is running properly.")
        print("You can still try to launch the UI, but responses may not work correctly.")
        user_continue = input("Do you want to continue anyway? (y/n): ").lower().strip()
        if user_continue != 'y':
            print("Exiting based on user request.")
            return
    
    # Display JSON management interface before starting main workflow
    choice = input("\nWould you like to manage JSON files before starting? (y/n): ").lower()
    if choice == 'y':
        display_json_manager()
    
    try:
     
        chatbot_responses = collect_chatbot_responses()


        ground_truths = establish_ground_truths(chatbot_responses)

        metrics = compute_accuracy()
    except Exception as e:
        print(f"Error during evaluation setup: {str(e)}")
        print("Continuing to Gradio UI despite evaluation error.")
    
    
    port_to_use = find_available_port(args.port) if args.port else find_available_port(8001)
    if not port_to_use:
        print("Could not find an available port after multiple attempts. Exiting.")
        return
    
    print(f"Using port: {port_to_use}")

    try:
        print("Launching Gradio UI...")
        demo = ui()
        demo.queue().launch(
            server_name=args.host, 
            server_port=port_to_use, 
            share=True, 
            quiet=True
        )
    except Exception as e:
        print(f"Error launching Gradio UI: {str(e)}")
        print("Check if port is already in use or if Gradio is properly installed.")

if __name__ == "__main__":
    main()
