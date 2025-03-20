



import gradio as gr
import json
import argparse
import time
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Tuple, Generator, Optional, Union
import traceback
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_qa.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import RAG components with error handling
try:
    from rag import rag_search, init_rag
    RAG_AVAILABLE = True
except ImportError:
    logger.warning("RAG module not available. Will run without RAG capabilities.")
    RAG_AVAILABLE = False
    
    # Define stub functions if RAG is not available
    def init_rag(pdf_path):
        logger.warning(f"RAG initialization called for {pdf_path} but RAG is not available")
        return None, None, None
        
    def rag_search(query, mc, encoder, dict_list, top_k=5):
        logger.warning(f"RAG search called for query '{query}' but RAG is not available")
        return []

# Argument parser setup with improved descriptions
parser = argparse.ArgumentParser(description='Advanced PDF Q&A System with Evaluation Capabilities')
parser.add_argument('--model-url', type=str, default='http://localhost:8000/v1', 
                    help='Model endpoint URL (default: http://localhost:8000/v1)')
parser.add_argument('-m', '--model', type=str, required=True, 
                    help='LLM model name for chatbot responses and evaluation')
parser.add_argument('--temp', type=float, default=0.7, 
                    help='Temperature for text generation (0.0-1.0, lower = more deterministic)')
parser.add_argument('--stop-token-ids', type=str, default='', 
                    help='Comma-separated stop token IDs for text generation')
parser.add_argument("--host", type=str, default=None,
                    help="Host address for Gradio web interface")
parser.add_argument("--port", type=int, default=8001,
                    help="Port for Gradio web interface")
parser.add_argument("--pdf-file", type=str, 
                    help="Path to the PDF file for Q&A")
parser.add_argument("--use-llm-eval", action="store_true", 
                    help="Use LLM for evaluation of response quality")
parser.add_argument("--extract-questions", action="store_true", 
                    help="Automatically extract questions from the PDF")
parser.add_argument("--chunk-size", type=int, default=1000, 
                    help="Chunk size for PDF processing (higher = more context but slower)")
parser.add_argument("--chunk-overlap", type=int, default=200, 
                    help="Chunk overlap for PDF processing to maintain context between chunks")
parser.add_argument("--export-csv", action="store_true", 
                    help="Export results to CSV for further analysis")
parser.add_argument("--visualization", action="store_true", 
                    help="Generate visualizations of evaluation metrics")
parser.add_argument("--auto-ground-truth", action="store_true",
                    help="Automatically generate ground truth answers with LLM")
parser.add_argument("--debug", action="store_true",
                    help="Enable debug mode with extra logging")

# Parse arguments
args = parser.parse_args()

# Configure logging level based on debug flag
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.debug("Debug mode enabled")

# Set default PDF path if not provided
if not args.pdf_file:
    # Use relative path based on the location of main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.pdf_file = os.path.join(script_dir, "SlamonetalSCIENCE1987.pdf")
    logger.info(f"Using default PDF file: {args.pdf_file}")

# Create output directory for results
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)
logger.info(f"Results will be saved to: {results_dir}")

# Initialize global variables
mc, encoder, dict_list = None, None, None
client = None
chunks = None
documents = None

def initialize_system() -> bool:
    """Initialize all system components and return success status."""
    global mc, encoder, dict_list, client, chunks, documents
    
    try:
        # Check if PDF file exists
        if not os.path.exists(args.pdf_file):
            logger.error(f"PDF file not found: {args.pdf_file}")
            return False
            
        # Initialize RAG components if available
        if RAG_AVAILABLE:
            logger.info(f"Initializing RAG with PDF: {args.pdf_file}")
            mc, encoder, dict_list = init_rag(args.pdf_file)
        
        # Load and process PDF
        logger.info(f"Loading PDF: {args.pdf_file}")
        chunks, documents = load_pdf(args.pdf_file, args.chunk_size, args.chunk_overlap)
        
        # Set OpenAI API configurations
        openai_api_key = "EMPTY"  # Placeholder for local models
        openai_api_base = args.model_url

        # Create OpenAI client
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        logger.info(f"Initialized OpenAI client with base URL: {args.model_url}")
        
        # Test connection with a simple query
        test_response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=args.temp,
        )
        logger.info("Successfully connected to LLM API")
        
        return True
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        return False

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
def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON serializable types.
    Handles NumPy types and nested structures.
    
    Args:
        obj: The object to convert
        
    Returns:
        JSON serializable version of the object
    """
    if hasattr(obj, 'item'):  # For numpy types
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    else:
        return obj

# Function to load a single PDF file with improved chunking
def load_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[List[str], List[Any]]:
    """
    Loads the PDF file using PyPDFLoader with customizable chunking.
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks to maintain context
        
    Returns:
        Tuple of (list of chunks, list of document objects)
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Extract metadata
        total_pages = len(documents)
        logger.info(f"Loaded PDF '{file_path}' with {total_pages} pages")
        
        # Extract raw text from the PDF
        text_content = "\n\n".join([doc.page_content for doc in documents])
        logger.debug(f"Total characters in PDF: {len(text_content)}")
        
        # Create text splitter for improved chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text_content)
        
        logger.info(f"Processed PDF into {len(chunks)} chunks with size {chunk_size} and overlap {chunk_overlap}")
        
        # Log sample chunks for debugging
        if args.debug and chunks:
            logger.debug(f"Sample chunk (1st): {chunks[0][:100]}...")
            logger.debug(f"Sample chunk (last): {chunks[-1][:100]}...")
        
        return chunks, documents
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        logger.error(traceback.format_exc())
        return [], []

# Function to extract potential questions from a PDF
def extract_questions_from_pdf(file_path: str) -> List[str]:
    """
    Extract potential questions from a PDF document using an LLM.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of extracted questions
    """
    global client
    chunks, _ = load_pdf(file_path, args.chunk_size, args.chunk_overlap)
    questions = []
    
    if not chunks:
        logger.error("No chunks found in PDF for question extraction")
        return questions
    
    logger.info(f"Extracting questions from {len(chunks)} chunks...")
    
    # Process each chunk to extract questions
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} for question extraction...")
        
        try:
            # Limit chunk size to avoid token limit issues
            chunk_text = chunk[:4000] if len(chunk) > 4000 else chunk
            
            prompt = f"""
            Given the following text extracted from a PDF document, identify 2-3 important questions that might be asked about this content.
            Focus on extracting questions that:
            1. Target key information in the text
            2. Would require detailed answers
            3. Are relevant to understanding the main points
            
            TEXT:
            {chunk_text}
            
            FORMAT YOUR RESPONSE AS A JSON ARRAY OF QUESTIONS ONLY, like:
            ["Question 1?", "Question 2?", "Question 3?"]
            """
            
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            
            result = response.choices[0].message.content
            
            try:
                # Extract just the JSON array part
                result = result.strip()
                if result.startswith("```json"):
                    result = result.split("```json")[1].split("```")[0].strip()
                elif result.startswith("```"):
                    result = result.split("```")[1].split("```")[0].strip()
                
                # Parse the JSON array
                extracted_questions = json.loads(result)
                questions.extend(extracted_questions)
                logger.info(f"Extracted {len(extracted_questions)} questions from chunk {i+1}")
                
                if args.debug:
                    for q in extracted_questions:
                        logger.debug(f"  - {q}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse questions from LLM response: {result}")
        except Exception as e:
            logger.error(f"Error extracting questions from chunk {i+1}: {str(e)}")
    
    # Remove duplicates and similar questions
    unique_questions = list(set(questions))
    logger.info(f"Extracted {len(unique_questions)} unique questions")
    
    # Save extracted questions to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    questions_file = os.path.join(results_dir, f"extracted_questions_{timestamp}.json")
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(unique_questions, f, indent=4)
    
    logger.info(f"Saved extracted questions to {questions_file}")
    return unique_questions

# Function to evaluate response using LLM
def evaluate_with_llm(question: str, ground_truth: str, response: str, 
                       eval_client: Any, model_name: str) -> Tuple[bool, float, str, Dict[str, float]]:
    """
    Uses an LLM to evaluate if the response correctly answers the question
    compared to the ground truth.
    
    Args:
        question: The original question
        ground_truth: The ground truth answer
        response: The chatbot's response
        eval_client: The LLM client
        model_name: The model to use for evaluation
        
    Returns:
        Tuple of (is_correct, confidence_score, explanation, detailed_scores)
    """
    prompt = f"""
    Question: {question}
    
    Ground Truth Answer: {ground_truth}
    
    Model Response: {response}
    
    Evaluate if the Model Response correctly answers the question compared to the Ground Truth.
    Consider:
    1. Factual accuracy - Does the response contain factually correct information consistent with the ground truth?
    2. Completeness - Does the response cover all important information from the ground truth?
    3. Relevance - Does the response directly address the question asked?
    4. Consistency - Is the response free from contradictions with the ground truth?
    5. Hallucination - Does the response make up information not in the ground truth?
    
    First, provide a brief explanation of your evaluation (2-3 sentences).
    Then, rate the response on each criteria (Accuracy, Completeness, Relevance, Consistency, Hallucination-free) on a scale of 0-10.
    Then, provide an overall score on a scale of 0-10 where 10 means perfect match in meaning and 0 means completely incorrect.
    Finally, provide a yes/no judgment on whether the response is correct overall.
    
    Format your response exactly as follows:
    Explanation: [your explanation]
    Accuracy: [0-10]
    Completeness: [0-10]
    Relevance: [0-10]
    Consistency: [0-10]
    Hallucination-free: [0-10]
    Overall Score: [0-10]
    Correct: [yes/no]
    """
    
    try:
        result = eval_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        
        response_text = result.choices[0].message.content
        logger.debug(f"LLM evaluation response: {response_text}")
        
        # Parse the evaluation response
        scores = {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "consistency": 0,
            "hallucination_free": 0,
            "overall": 0
        }
        
        explanation = ""
        correct = False
        
        # Extract each metric from the response
        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith('Explanation:'):
                explanation = line.split(':', 1)[1].strip()
            elif line.startswith('Accuracy:'):
                try:
                    scores["accuracy"] = float(line.split(':', 1)[1].strip())
                except:
                    logger.warning(f"Failed to parse Accuracy score: {line}")
            elif line.startswith('Completeness:'):
                try:
                    scores["completeness"] = float(line.split(':', 1)[1].strip())
                except:
                    logger.warning(f"Failed to parse Completeness score: {line}")
            elif line.startswith('Relevance:'):
                try:
                    scores["relevance"] = float(line.split(':', 1)[1].strip())
                except:
                    logger.warning(f"Failed to parse Relevance score: {line}")
            elif line.startswith('Consistency:'):
                try:
                    scores["consistency"] = float(line.split(':', 1)[1].strip())
                except:
                    logger.warning(f"Failed to parse Consistency score: {line}")
            elif line.startswith('Hallucination-free:'):
                try:
                    scores["hallucination_free"] = float(line.split(':', 1)[1].strip())
                except:
                    logger.warning(f"Failed to parse Hallucination-free score: {line}")
            elif line.startswith('Overall Score:'):
                try:
                    scores["overall"] = float(line.split(':', 1)[1].strip()) / 10.0  # Normalize to 0-1
                except:
                    logger.warning(f"Failed to parse Overall Score: {line}")
            elif line.startswith('Correct:'):
                correct = 'yes' in line.lower()
        
        return correct, scores["overall"], explanation, scores
    except Exception as e:
        logger.error(f"Error in LLM evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return False, 0.0, f"Evaluation error: {str(e)}", {
            "accuracy": 0, 
            "completeness": 0, 
            "relevance": 0, 
            "consistency": 0, 
            "hallucination_free": 0, 
            "overall": 0
        }

# Function to check if a message is a generic greeting
def is_generic_greeting(message: str) -> bool:
    """Check if the message is a generic greeting."""
    message = message.lower().strip()
    return any(greeting in message for greeting in GENERIC_GREETINGS)

# Function to get response for generic greetings
def get_greeting_response(message: str) -> str:
    """Return an appropriate response for a generic greeting."""
    message = message.lower().strip()
    
    for greeting in GENERIC_GREETINGS:
        if greeting in message:
            return GREETING_RESPONSES.get(greeting, GREETING_RESPONSES["default"])
    
    return GREETING_RESPONSES["default"]

# Core prediction function using RAG if available
def predict(message: str, history: List[List[str]]) -> Generator[Tuple[str, List[List[str]]], None, None]:
    """
    Generate a response to the user's message using RAG if available.
    
    Args:
        message: The user's message
        history: Chat history
        
    Returns:
        Generator yielding tuples of (response, updated_history)
    """
    global mc, encoder, dict_list, client, chunks
    
    # Handle generic greetings
    if is_generic_greeting(message):
        response = get_greeting_response(message)
        history.append([message, response])
        yield response, history
        return

    try:
        # Use RAG to get relevant context if available
        context = ""
        if RAG_AVAILABLE and mc and encoder and dict_list:
            logger.info(f"Performing RAG search for: {message}")
            search_results = rag_search(message, mc, encoder, dict_list, top_k=3)
            if search_results:
                context = "\n\n".join([result["text"] for result in search_results])
                logger.debug(f"RAG context (first 200 chars): {context[:200]}...")
        
        # If RAG is not available or didn't provide results, use simple chunk search
        if not context and chunks:
            logger.info("Using simple chunk search as fallback")
            # Simple keyword matching as fallback
            keywords = message.lower().split()
            relevant_chunks = []
            
            for chunk in chunks:
                chunk_lower = chunk.lower()
                # Count how many keywords appear in this chunk
                keyword_matches = sum(1 for keyword in keywords if keyword in chunk_lower)
                if keyword_matches > 1:  # At least 2 keywords must match
                    relevant_chunks.append((chunk, keyword_matches))
            
            # Sort by number of keyword matches (highest first)
            relevant_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # Take the top 2 most relevant chunks
            if relevant_chunks:
                context = "\n\n".join([chunk for chunk, _ in relevant_chunks[:2]])
                logger.debug(f"Fallback context (first 200 chars): {context[:200]}...")
        
        # Prepare the prompt with context if available
        if context:
            prompt = f"""
            Use the following information from a PDF document to answer the question.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {message}
            
            Provide a comprehensive answer based on the information provided. 
            If the context doesn't contain relevant information to answer the question, 
            say "I don't have enough information to answer that question based on the document."
            """
        else:
            prompt = f"""
            QUESTION:
            {message}
            
            Answer this question based on your knowledge of the PDF document.
            If you don't have enough information to answer the question, 
            say "I don't have enough information to answer that question based on the document."
            """
        
        # Generate response
        logger.info(f"Generating response for: {message}")
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temp,
        )
        
        # Extract and return response
        response_text = response.choices[0].message.content
        history.append([message, response_text])
        yield response_text, history
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        history.append([message, error_msg])
        yield error_msg, history

# Function to collect chatbot responses with extracted questions
def collect_chatbot_responses(extracted_questions: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    Collect chatbot responses for a set of questions.
    
    Args:
        extracted_questions: Optional list of questions to use
        
    Returns:
        Tuple of (list of response objects, path to saved responses file)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"chatbot_responses_{timestamp}.json")
    responses = []
    
    # Check for existing responses file
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            responses = json.load(f)
        logger.info(f"Loaded {len(responses)} existing responses from {filename}")

    if extracted_questions:
        logger.info(f"Using {len(extracted_questions)} extracted questions from PDF...")
        
        for i, question in enumerate(extracted_questions):
            logger.info(f"Processing question {i+1}/{len(extracted_questions)}")
            
            start_time = time.time()
            try:
                # Use predict function to get response
                response_generator = predict(question, [])
                response, _ = next(response_generator)  # Get first response
                response_time = round((time.time() - start_time) * 1000, 2)

                responses.append({
                    "question": question, 
                    "response": response, 
                    "response_time_ms": response_time,
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"Q: {question}")
                logger.info(f"A: {response}")
                logger.info(f"Response Time: {response_time} ms\n")
            except Exception as e:
                logger.error(f"Error generating response for question: {question}")
                logger.error(str(e))
                logger.error(traceback.format_exc())
    else:
        logger.info("Enter questions manually (type 'done' to finish):")
        while True:
            question = input("Question: ").strip()
            if question.lower() == "done":
                break

            start_time = time.time()
            try:
                # Use predict function to get response
                response_generator = predict(question, [])
                response, _ = next(response_generator)  # Get first response
                response_time = round((time.time() - start_time) * 1000, 2)

                responses.append({
                    "question": question, 
                    "response": response, 
                    "response_time_ms": response_time,
                    "timestamp": datetime.now().isoformat()
                })
                print(f"Chatbot Response: {response}")
                print(f"Response Time: {response_time} ms\n")
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                print(f"Error: {str(e)}")

    # Save responses to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4)

    logger.info(f"Saved {len(responses)} chatbot responses to {filename}")
    return responses, filename

# Function to establish ground truths with more options
def establish_ground_truths(chatbot_responses_file: str, manual_entry: bool = True) -> Tuple[List[Dict[str, Any]], str]:
    """
    Establish ground truths for evaluation.
    
    Args:
        chatbot_responses_file: Path to chatbot responses JSON file
        manual_entry: Whether to collect ground truths manually
        
    Returns:
        Tuple of (list of ground truth objects, path to saved ground truths file)
    """
    global client
    
    with open(chatbot_responses_file, "r", encoding="utf-8") as f:
        chatbot_responses = json.load(f)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"ground_truths_{timestamp}.json")
    
    # Check for existing ground truths
    ground_truths = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            ground_truths = json.load(f)
        logger.info(f"Loaded {len(ground_truths)} existing ground truths from {filename}")

    existing_questions = {gt["question"] for gt in ground_truths}
    
    if manual_entry:
        logger.info("\nEnter ground truths for each question:")
        print("\nEnter ground truths for each question:")
        print("(For each question, you'll see both the question and the chatbot's response)")
        print("(Type 'skip' to skip a question, 'auto' to generate ground truth with LLM, or 'quit' to finish)")
        
        for cr in chatbot_responses:
            question = cr["question"]
            if question in existing_questions:
                continue
                
            print(f"\nQuestion: {question}")
            print(f"Chatbot Response: {cr['response']}")
            
            while True:
                choice = input("Enter ground truth, 'skip', 'auto', or 'quit': ").strip()
                
                if choice.lower() == 'quit':
                    break
                elif choice.lower() == 'skip':
                    print("Skipping this question...")
                    break
                elif choice.lower() == 'auto':
                    print("Generating ground truth with LLM...")
                    try:
                        # Generate ground truth using the LLM
                        prompt = f"""
                        Based on the following question about a scientific paper, provide a factual, neutral, and accurate answer.
                        If you don't have enough information to answer, state that clearly.
                        
                        Question: {question}
                        """
                        
                        result = client.chat.completions.create(
                            model=args.model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                        )
                        
                        ground_truth = result.choices[0].message.content
                        print(f"Generated ground truth: {ground_truth}")
                        
                        confirm = input("Accept this ground truth? (y/n): ").strip().lower()
                        if confirm == 'y':
                            ground_truths.append({
                                "question": question, 
                                "ground_truth": ground_truth,
                                "source": "llm-generated",
                                "timestamp": datetime.now().isoformat()
                            })
                            break
                        else:
                            print("Let's try again...")
                    except Exception as e:
                        logger.error(f"Error generating ground truth: {str(e)}")
                        print(f"Error generating ground truth: {str(e)}")
                else:
                    # Manual ground truth entry
                    ground_truths.append({
                        "question": question, 
                        "ground_truth": choice,
                        "source": "manual",
                        "timestamp": datetime.now().isoformat()
                    })
                    break
                    
            if choice.lower() == 'quit':
                break
    else:
        # Auto-generate all ground truths using LLM
        logger.info("\nAuto-generating ground truths for all questions...")
        for cr in chatbot_responses:
            question = cr["question"]
            if question in existing_questions:
                continue
                
            try:
                # Generate ground truth using the LLM
                prompt = f"""
                Based on the following question about a scientific paper, provide a factual, neutral, and accurate answer.
                If you don't have enough information to answer, state that clearly.
                
                Question: {question}
                """
                
                result = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                
                ground_truth = result.choices[0].message.content
                ground_truths.append({
                    "question": question, 
                    "ground_truth": ground_truth,
                    "source": "llm-generated-auto",
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"Generated ground truth for: {question}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error generating ground truth for {question}: {str(e)}")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ground_truths, f, indent=4)

    logger.info(f"Ground truths saved to {filename}")
    return ground_truths, filename

# Function to generate visualizations
def generate_visualizations(metrics_results: Dict[str, Any], output_dir: str = results_dir) -> Dict[str, str]:
    """
    Generate visualizations for evaluation metrics.
    
    Args:
        metrics_results: Dictionary of evaluation metrics
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of paths to the generated visualizations
    """
    logger.info("\nGenerating visualizations...")
    
    # Create a timestamp for the visualization files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame()
    
    # If we have LLM evaluation data
    if "llm_evaluation" in metrics_results and "detailed_scores" in metrics_results["llm_evaluation"]:
        detailed_scores = metrics_results["llm_evaluation"]["detailed_scores"]
        
        # Create DataFrame from detailed scores
        questions = []
        accuracies = []
        completenesses = []
        relevances = []
        consistencies = []
        hallucination_frees = []
        overalls = []
        response_times = []
        
        for item in detailed_scores:
            questions.append(item["question"])
            accuracies.append(item["scores"]["accuracy"])
            completenesses.append(item["scores"]["completeness"])
            relevances.append(item["scores"]["relevance"])
            consistencies.append(item["scores"]["consistency"])
            hallucination_frees.append(item["scores"]["hallucination_free"])
            overalls.append(item["scores"]["overall"] * 10)  # Convert back to 0-10 scale
            response_times.append(item["response_time_ms"])
        
        metrics_df = pd.DataFrame({
            "Question": questions,
            "Accuracy": accuracies,
            "Completeness": completenesses,
            "Relevance": relevances,
            "Consistency": consistencies,
            "Hallucination-free": hallucination_frees,
            "Overall Score": overalls,
            "Response Time (ms)": response_times
        })
        
        # 1. Bar chart for average scores across all criteria
        plt.figure(figsize=(12, 6))
        avg_scores = {
            "Accuracy": metrics_df["Accuracy"].mean(),
            "Completeness": metrics_df["Completeness"].mean(),
            "Relevance": metrics_df["Relevance"].mean(),
            "Consistency": metrics_df["Consistency"].mean(),
            "Hallucination-free": metrics_df["Hallucination-free"].mean(),
            "Overall": metrics_df["Overall Score"].mean()
        }
        
        plt.bar(avg_scores.keys(), avg_scores.values(), color='skyblue')
        plt.axhline(y=7, color='r', linestyle='--', label='Good threshold (7/10)')
        plt.title('Average Scores by Criteria')
        plt.ylabel('Score (0-10)')
        plt.ylim(0, 10)
        plt.legend()
        plt.tight_layout()
        
        bar_chart_path = os.path.join(output_dir, f"avg_scores_bar_chart_{timestamp}.png")
        plt.savefig(bar_chart_path)
        plt.close()
        
        # 2. Heatmap of all scores for all questions
        plt.figure(figsize=(14, max(6, len(questions) * 0.4)))
        score_columns = ["Accuracy", "Completeness", "Relevance", "Consistency", "Hallucination-free", "Overall Score"]
        
        # Create score matrix
        score_matrix = metrics_df[score_columns].values
        
        # Create heatmap
        im = plt.imshow(score_matrix, cmap='YlGnBu', aspect='auto')
        plt.colorbar(im, label='Score (0-10)')
        
        # Set labels
        plt.yticks(range(len(questions)), [q[:50] + '...' if len(q) > 50 else q for q in questions])
        plt.xticks(range(len(score_columns)), score_columns, rotation=45)
        
        plt.title('Detailed Scores by Question and Criteria')
        plt.tight_layout()
        
        heatmap_path = os.path.join(output_dir, f"scores_heatmap_{timestamp}.png")
        plt.savefig(heatmap_path)
        plt.close()
        
        # 3. Scatter plot of response time vs. overall score
        plt.figure(figsize=(10, 6))
        plt.scatter(metrics_df["Response Time (ms)"], metrics_df["Overall Score"], alpha=0.7)
        
        # Add trend line
        z = np.polyfit(metrics_df["Response Time (ms)"], metrics_df["Overall Score"], 1)
        p = np.poly1d(z)
        plt.plot(metrics_df["Response Time (ms)"], p(metrics_df["Response Time (ms)"]), "r--", alpha=0.7)
        
        plt.title('Response Time vs. Overall Score')
        plt.xlabel('Response Time (ms)')
        plt.ylabel('Overall Score (0-10)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        scatter_path = os.path.join(output_dir, f"time_vs_score_scatter_{timestamp}.png")
        plt.savefig(scatter_path)
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
        
        # Return paths to the visualizations
        return {
            "bar_chart": bar_chart_path,
            "heatmap": heatmap_path,
            "scatter_plot": scatter_path
        }
    else:
        logger.warning("Not enough data for visualizations. Make sure LLM evaluation is enabled.")
        return {}

# Function to export results to CSV
def export_to_csv(metrics_results: Dict[str, Any], chatbot_responses_file: str, 
                  ground_truth_file: str, output_dir: str = results_dir) -> Optional[str]:
    """
    Export all evaluation results to CSV for further analysis.
    
    Args:
        metrics_results: Dictionary of evaluation metrics
        chatbot_responses_file: Path to chatbot responses file
        ground_truth_file: Path to ground truths file
        output_dir: Directory to save CSV file
        
    Returns:
        Path to the CSV file or None if export failed
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Load chatbot responses and ground truths
        with open(chatbot_responses_file, "r", encoding="utf-8") as f:
            chatbot_responses = json.load(f)
        
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            ground_truths = json.load(f)
        
        # Create dictionary from ground truths
        ground_truth_dict = {gt["question"]: gt["ground_truth"] for gt in ground_truths}
        
        # Create export data
        export_data = []
        
        if "llm_evaluation" in metrics_results and "detailed_scores" in metrics_results["llm_evaluation"]:
            detailed_scores = metrics_results["llm_evaluation"]["detailed_scores"]
            
            for item in detailed_scores:
                question = item["question"]
                is_correct = item["is_correct"]
                scores = item["scores"]
                response_time = item["response_time_ms"]
                
                # Find the corresponding response
                response = next((cr["response"] for cr in chatbot_responses if cr["question"] == question), "")
                
                # Get ground truth
                ground_truth = ground_truth_dict.get(question, "")
                
                export_data.append({
                    "Question": question,
                    "Ground Truth": ground_truth,
                    "Response": response,
                    "Correct": "Yes" if is_correct else "No",
                    "Accuracy": scores["accuracy"],
                    "Completeness": scores["completeness"],
                    "Relevance": scores["relevance"],
                    "Consistency": scores["consistency"],
                    "Hallucination-free": scores["hallucination_free"],
                    "Overall Score": scores["overall"] * 10,  # Convert back to 0-10 scale
                    "Response Time (ms)": response_time
                })
        
        # Export to CSV
        csv_filename = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
        
        if export_data:
            with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                fieldnames = export_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_data)
            
            logger.info(f"Results exported to CSV: {csv_filename}")
            return csv_filename
        else:
            logger.warning("No data to export to CSV.")
            return None
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Function to compute accuracy metrics with detailed scores
def compute_accuracy(chatbot_responses_file: str, ground_truth_file: str) -> Dict[str, Any]:
    """
    Computes chatbot accuracy with detailed metrics.
    
    Args:
        chatbot_responses_file: Path to chatbot responses file
        ground_truth_file: Path to ground truths file
        
    Returns:
        Dictionary of accuracy metrics
    """
    logger.info("\nComputing accuracy metrics...")
    
    try:
        with open(chatbot_responses_file, "r", encoding="utf-8") as f:
            chatbot_responses = json.load(f)

        with open(ground_truth_file, "r", encoding="utf-8") as f:
            ground_truths = json.load(f)

        chatbot_dict = {cr["question"]: (cr["response"], cr["response_time_ms"]) for cr in chatbot_responses}
        ground_truth_dict = {gt["question"]: gt["ground_truth"] for gt in ground_truths}

        common_questions = set(chatbot_dict.keys()) & set(ground_truth_dict.keys())
        if not common_questions:
            logger.warning("No matching questions found between responses and ground truths.")
            return {}

        total_questions = len(common_questions)
        logger.info(f"Evaluating {total_questions} questions with both responses and ground truths")
        
        # Initialize metrics
        metrics = {
            "llm_evaluation": {
                "correct": 0,
                "scores": [],
                "detailed_scores": []
            }
        }
        
        total_response_time = 0

        logger.info("\nEvaluation Results:")
        logger.info("-" * 100)
        
        for i, question in enumerate(common_questions):
            logger.info(f"Evaluating question {i+1}/{len(common_questions)}: {question[:50]}...")
            
            chatbot_answer, response_time = chatbot_dict[question]
            ground_truth_answer = ground_truth_dict[question]
            total_response_time += response_time
            
            # Use LLM evaluation
            if args.use_llm_eval:
                try:
                    is_correct_llm, llm_score, explanation, detailed_scores = evaluate_with_llm(
                        question, ground_truth_answer, chatbot_answer, client, args.model
                    )
                    # Ensure it's a native Python float
                    llm_score = float(llm_score)
                    metrics["llm_evaluation"]["scores"].append(llm_score)
                    metrics["llm_evaluation"]["correct"] += int(is_correct_llm)
                    
                    # Store detailed evaluation data
                    metrics["llm_evaluation"]["detailed_scores"].append({
                        "question": question,
                        "is_correct": is_correct_llm,
                        "overall_score": llm_score,
                        "explanation": explanation,
                        "scores": detailed_scores,
                        "response_time_ms": response_time
                    })
                    
                    # Log results
                    result_status = "CORRECT" if is_correct_llm else "INCORRECT"
                    logger.info(f"Question {i+1}: {result_status} (Score: {llm_score:.2f})")
                    logger.debug(f"Explanation: {explanation}")
                except Exception as e:
                    logger.error(f"Error evaluating question {i+1}: {str(e)}")
            else:
                logger.warning("LLM evaluation not enabled. Use --use-llm-eval to enable.")
        
        # Calculate summary metrics
        if args.use_llm_eval and metrics["llm_evaluation"]["scores"]:
            avg_score = sum(metrics["llm_evaluation"]["scores"]) / len(metrics["llm_evaluation"]["scores"])
            accuracy = metrics["llm_evaluation"]["correct"] / total_questions if total_questions > 0 else 0
            avg_response_time = total_response_time / total_questions if total_questions > 0 else 0
            
            metrics["llm_evaluation"]["summary"] = {
                "total_questions": total_questions,
                "correct_answers": metrics["llm_evaluation"]["correct"],
                "accuracy": accuracy,
                "average_score": avg_score,
                "average_response_time_ms": avg_response_time
            }
            
            # Log summary
            logger.info("\nEvaluation Summary:")
            logger.info(f"Total Questions: {total_questions}")
            logger.info(f"Correct Answers: {metrics['llm_evaluation']['correct']}")
            logger.info(f"Accuracy: {accuracy:.2%}")
            logger.info(f"Average Score: {avg_score:.2f}")
            logger.info(f"Average Response Time: {avg_response_time:.2f} ms")
        
        # Make sure metrics are JSON serializable
        metrics = make_json_serializable(metrics)
        
        # Save metrics to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {metrics_file}")
        
        return metrics
    except Exception as e:
        logger.error(f"Error computing accuracy: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

# Create Gradio interface for chatbot
def create_gradio_interface():
    """Create and launch the Gradio web interface."""
    logger.info("Setting up Gradio interface...")
    
    with gr.Blocks(title="PDF Q&A System") as demo:
        gr.Markdown(f"# PDF Q&A System\nAnalyzing: {os.path.basename(args.pdf_file)}")
        
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="Ask a question about the PDF", placeholder="Ask a question...", lines=2)
            clear = gr.Button("Clear Chat")
            
            pdf_info = gr.Markdown(f"""
            ## PDF Information
            - **Filename:** {os.path.basename(args.pdf_file)}
            - **Chunking:** Size: {args.chunk_size}, Overlap: {args.chunk_overlap}
            - **RAG Available:** {"Yes" if RAG_AVAILABLE else "No"}
            """)
            
            def user(user_message, history):
                return "", history + [[user_message, None]]
            
            def respond(message, chat_history):
                response_gen = predict(message, chat_history[:-1])
                response, history = next(response_gen)
                chat_history[-1][1] = response
                return chat_history
            
            def clear_chat():
                return None
            
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                respond, [msg, chatbot], [chatbot]
            )
            clear.click(clear_chat, None, chatbot, queue=False)
        
        with gr.Tab("Evaluation"):
            gr.Markdown("## PDF Q&A Evaluation Tools")
            
            with gr.Row():
                with gr.Column():
                    extract_btn = gr.Button("1. Extract Questions from PDF")
                    collect_responses_btn = gr.Button("2. Collect Responses")
                    ground_truth_btn = gr.Button("3. Establish Ground Truths")
                    compute_metrics_btn = gr.Button("4. Compute Metrics")
                
                with gr.Column():
                    status_md = gr.Markdown("Status: System Ready")
                    metrics_display = gr.JSON(label="Evaluation Metrics")
                    visualize_btn = gr.Button("Generate Visualizations")
                    export_csv_btn = gr.Button("Export to CSV")
            
            def extract_questions_handler():
                status_md.value = "Extracting questions from PDF..."
                try:
                    extracted_questions = extract_questions_from_pdf(args.pdf_file)
                    status_md.value = f"Extracted {len(extracted_questions)} questions from PDF"
                    return {"questions": extracted_questions}
                except Exception as e:
                    status_md.value = f"Error extracting questions: {str(e)}"
                    return {}
            
            def collect_responses_handler(state):
                if state and "questions" in state:
                    status_md.value = "Collecting chatbot responses..."
                    try:
                        responses, responses_file = collect_chatbot_responses(state["questions"])
                        state["responses_file"] = responses_file
                        status_md.value = f"Collected {len(responses)} responses, saved to {responses_file}"
                        return state
                    except Exception as e:
                        status_md.value = f"Error collecting responses: {str(e)}"
                        return state
                else:
                    status_md.value = "No questions found. Extract questions first."
                    return state
            
            def establish_ground_truths_handler(state):
                if state and "responses_file" in state:
                    status_md.value = "Establishing ground truths..."
                    try:
                        manual = not args.auto_ground_truth
                        ground_truths, gt_file = establish_ground_truths(state["responses_file"], manual)
                        state["ground_truth_file"] = gt_file
                        status_md.value = f"Established {len(ground_truths)} ground truths, saved to {gt_file}"
                        return state
                    except Exception as e:
                        status_md.value = f"Error establishing ground truths: {str(e)}"
                        return state
                else:
                    status_md.value = "No responses found. Collect responses first."
                    return state
            
            def compute_metrics_handler(state):
                if state and "responses_file" in state and "ground_truth_file" in state:
                    status_md.value = "Computing evaluation metrics..."
                    try:
                        metrics = compute_accuracy(state["responses_file"], state["ground_truth_file"])
                        state["metrics"] = metrics
                        
                        # Format summary for display
                        if "llm_evaluation" in metrics and "summary" in metrics["llm_evaluation"]:
                            summary = metrics["llm_evaluation"]["summary"]
                            status_md.value = f"""
                            ## Evaluation Complete
                            - **Total Questions:** {summary['total_questions']}
                            - **Correct Answers:** {summary['correct_answers']}
                            - **Accuracy:** {summary['accuracy']*100:.2f}%
                            - **Average Score:** {summary['average_score']:.2f}
                            - **Average Response Time:** {summary['average_response_time_ms']:.2f} ms
                            """
                        else:
                            status_md.value = "Evaluation complete, but no summary metrics available."
                        
                        return state, metrics
                    except Exception as e:
                        status_md.value = f"Error computing metrics: {str(e)}"
                        return state, {}
                else:
                    status_md.value = "Missing responses or ground truths. Complete previous steps first."
                    return state, {}
            
            def generate_visualizations_handler(state):
                if state and "metrics" in state:
                    status_md.value = "Generating visualizations..."
                    try:
                        viz_paths = generate_visualizations(state["metrics"])
                        if viz_paths:
                            viz_html = "<h3>Visualizations Generated</h3><ul>"
                            for viz_type, path in viz_paths.items():
                                viz_html += f"<li>{viz_type}: {os.path.basename(path)}</li>"
                            viz_html += "</ul>"
                            status_md.value = viz_html
                        else:
                            status_md.value = "No visualizations could be generated."
                        return state
                    except Exception as e:
                        status_md.value = f"Error generating visualizations: {str(e)}"
                        return state
                else:
                    status_md.value = "No metrics found. Compute metrics first."
                    return state
            
            def export_csv_handler(state):
                if state and "metrics" in state and "responses_file" in state and "ground_truth_file" in state:
                    status_md.value = "Exporting results to CSV..."
                    try:
                        csv_file = export_to_csv(state["metrics"], state["responses_file"], state["ground_truth_file"])
                        if csv_file:
                            status_md.value = f"Results exported to {csv_file}"
                        else:
                            status_md.value = "Failed to export results to CSV."
                        return state
                    except Exception as e:
                        status_md.value = f"Error exporting to CSV: {str(e)}"
                        return state
                else:
                    status_md.value = "Missing data for CSV export. Complete previous steps first."
                    return state
            
            # Set up state
            state = gr.State({})
            
            # Connect buttons to handlers
            extract_btn.click(extract_questions_handler, [], [state])
            collect_responses_btn.click(collect_responses_handler, [state], [state])
            ground_truth_btn.click(establish_ground_truths_handler, [state], [state])
            compute_metrics_btn.click(compute_metrics_handler, [state], [state, metrics_display])
            visualize_btn.click(generate_visualizations_handler, [state], [state])
            export_csv_btn.click(export_csv_handler, [state], [state])
        
        with gr.Tab("Help & Info"):
            gr.Markdown("""
            # PDF Q&A System Help
            
            This system allows you to:
            
            1. Ask questions about the loaded PDF document
            2. Evaluate the quality of answers using LLM-based evaluation
            3. Generate visualizations and reports of the evaluation
            
            ## Chat Tab
            - Type your question in the text box and press Enter
            - The system will use RAG (if available) to find relevant information and answer your question
            - Use "Clear Chat" to start a new conversation
            
            ## Evaluation Tab
            Follow these steps for evaluation:
            
            1. **Extract Questions**: Generate questions from the PDF automatically
            2. **Collect Responses**: Get chatbot responses for all extracted questions
            3. **Establish Ground Truths**: Define the correct answers for evaluation
            4. **Compute Metrics**: Calculate accuracy and other quality metrics
            5. **Generate Visualizations**: Create charts showing performance
            6. **Export to CSV**: Save results for external analysis
            
            ## Command-line Options
            Run with `--help` to see all available options.
            """)
    
    # Launch Gradio interface
    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=False
        )
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {str(e)}")
        logger.error(traceback.format_exc())

# Main function
def main():
    """Main entry point for the application."""
    logger.info("Starting PDF Q&A System...")
    
    # Initialize system components
    if not initialize_system():
        logger.error("Initialization failed. Exiting.")
        return
    
    # Extract questions if requested
    extracted_questions = None
    if args.extract_questions:
        logger.info("Extracting questions from PDF...")
        extracted_questions = extract_questions_from_pdf(args.pdf_file)
    
    # Launch Gradio UI
    create_gradio_interface()

if __name__ == "__main__":
    main()
