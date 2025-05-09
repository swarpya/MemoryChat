import os
from flask import Flask, render_template, request, jsonify, session
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import numpy as np
import logging # Import logging
from waitress import serve # Import waitress

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Initialize Models (Load these once) ---
# Ensure model path is accessible, default path works in Docker
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("SentenceTransformer model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading SentenceTransformer model: {e}")
    embedding_model = None # Handle potential loading errors

# Initialize the Groq client
# It's recommended to set the API key as an environment variable GROQ_API_KEY
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    logging.error("GROQ_API_KEY environment variable not set.")
    # In deployment, this should ideally stop the app or show an error page
    # For this example, we'll proceed but Groq calls will fail
    client = None
else:
    client = Groq(api_key=groq_api_key)
    logging.info("Groq client initialized.")


# --- Flask App Setup ---
app = Flask(__name__)
# A secret key is required for Flask sessions
# USE A STRONG, RANDOM KEY IN PRODUCTION ENVIRONMENT VARIABLE!
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_please_change') # !!! CHANGE THIS DEFAULT or set ENV VAR in Hugging Face Space Secrets !!!

# --- Memory Management Functions (Adapted for Sessions) ---

# These functions will now operate on a memory list passed to them,
# rather than a global variable. The Flask route will manage the session state.

def add_to_memory(mem_list, role, content):
    """
    Add a message to the provided memory list along with its embedding.
    Returns the updated list.
    """
    if embedding_model is None:
        logging.error("Embedding model not loaded. Cannot add to memory.")
        return mem_list
    try:
        # Check if content is not empty before encoding
        if not content or not content.strip():
             logging.warning(f"Attempted to add empty content to memory for role: {role}")
             return mem_list # Do not add empty messages

        embedding = embedding_model.encode(content, convert_to_numpy=True)
        mem_list.append({"role": role, "content": content, "embedding": embedding.tolist()}) # Store embedding as list for JSON serializability in session
        return mem_list
    except Exception as e:
        logging.error(f"Error adding to memory: {e}")
        return mem_list


def retrieve_relevant_memory(mem_list, user_input, top_k=5):
    """
    Retrieve the top-k most relevant messages from the provided memory list
    based on cosine similarity with user_input.
    Returns a list of relevant messages (dictionaries).
    """
    if not mem_list or embedding_model is None:
        return []

    try:
        # Compute the embedding of the user input
        user_embedding = embedding_model.encode(user_input, convert_to_numpy=True)

        # Calculate similarities. Ensure all memory entries have valid embeddings.
        # We need to convert embedding lists back to numpy arrays for cosine_similarity
        valid_memory_with_embeddings = []
        for m in mem_list:
            if "embedding" in m and m["embedding"] is not None:
                try:
                     # Attempt to convert embedding list back to numpy array
                    np_embedding = np.array(m["embedding"])
                    if np_embedding.shape == (embedding_model.get_sentence_embedding_dimension(),): # Check dimension
                         valid_memory_with_embeddings.append((m, np_embedding))
                except Exception as conv_e:
                    logging.warning(f"Could not convert embedding for memory entry: {m['content'][:50]}... Error: {conv_e}")
                    pass # Skip this memory entry if embedding is invalid

        if not valid_memory_with_embeddings:
             return []

        memory_items, memory_embeddings = zip(*valid_memory_with_embeddings)

        # Calculate similarities
        similarities = cosine_similarity([user_embedding], list(memory_embeddings))[0]

        # Sort memory by similarity and return the top-k messages
        relevant_messages_sorted = sorted(zip(similarities, memory_items), key=lambda x: x[0], reverse=True)

        # Return the message dictionaries
        return [m[1] for m in relevant_messages_sorted[:top_k]]

    except Exception as e:
        logging.error(f"Error retrieving memory: {e}")
        return []


def construct_prompt(mem_list, user_input, max_tokens_in_prompt=1000): # Increased max tokens slightly
    """
    Construct the list of messages suitable for the Groq API's 'messages' parameter
    by combining relevant memory and the current user input.
    Adds relevant memory chronologically from the session history.
    """
    # Retrieve relevant memory *content* based on similarity
    relevant_memory_items = retrieve_relevant_memory(mem_list, user_input)
    # Create a set of content strings from the relevant items for quick lookup
    relevant_content_set = {m["content"] for m in relevant_memory_items}

    messages_for_api = []
    # Add a system message
    messages_for_api.append({"role": "system", "content": "You are a helpful and friendly AI assistant."})

    current_prompt_tokens = len(messages_for_api[0]["content"].split()) # Start count with system message

    # Iterate through chronological session memory and add relevant messages
    context_messages = []
    for msg in mem_list:
        # Only add messages whose content is found in the top-k relevant messages
        # and which have a role suitable for the API messages list
        if msg["content"] in relevant_content_set and msg["role"] in ["user", "assistant", "system"]:
            # Estimate tokens for this message (simple word count)
            msg_text = f'{msg["role"]}: {msg["content"]}\n' # Estimate based on formatted string length
            msg_tokens = len(msg_text.split())
            if current_prompt_tokens + msg_tokens > max_tokens_in_prompt:
                break # Stop if adding this message exceeds the limit

            # Add the message in the format expected by the API
            context_messages.append({"role": msg["role"], "content": msg["content"]})
            current_prompt_tokens += msg_tokens

    # Add the chronological context messages
    messages_for_api.extend(context_messages)

    # Add the current user input as the final message
    # Ensure user input itself doesn't push over the limit significantly (though it should always be included)
    user_input_tokens = len(user_input.split())
    if current_prompt_tokens + user_input_tokens > max_tokens_in_prompt and len(messages_for_api) > 1:
         # If user input pushes over, and there's existing context, log a warning
         logging.warning(f"User input exceeds max_tokens_in_prompt with existing context. Truncating context.")
         # In a real scenario, you might trim context from the beginning here
         pass # User input is always added

    messages_for_api.append({"role": "user", "content": user_input})

    return messages_for_api


def trim_memory(mem_list, max_size=50):
    """
    Trim the memory list to keep it within the specified max size.
    Removes the oldest entries (from the beginning of the list).
    Returns the trimmed list.
    """
    while len(mem_list) > max_size:
        mem_list.pop(0)  # Remove the oldest entry
    return mem_list

# The summarize_memory function is defined but not used in the current web chat loop.
# Keeping it here for completeness.
def summarize_memory(mem_list):
    """
    Summarize the memory buffer to free up space.
    This would typically replace the detailed memory with a summary entry.
    Needs Groq client and memory list as input.
    """
    if not mem_list or client is None:
        logging.warning("Memory is empty or Groq client not initialized. Cannot summarize.")
        return [] # Return empty list or original list? Let's return an empty list + summary

    long_term_memory = " ".join([m["content"] for m in mem_list if "content" in m]) # Add check for content key
    if not long_term_memory.strip(): # Check if memory is empty or just whitespace after joining
         logging.warning("Memory content is empty. Cannot summarize.")
         return []

    try:
        summary_completion = client.chat.completions.create(
            # Use a currently available Groq model for summarization
            model="llama-3.1-8b-instruct-fpt", # Or "llama-3.1-70b-versatile", etc. Check Groq docs.
            messages=[
                {"role": "system", "content": "Summarize the following conversation for key points. Keep it concise."},
                {"role": "user", "content": long_term_memory},
            ],
            max_tokens= 500, # Limit summary length
        )
        # Access the content correctly from the message object
        summary_text = summary_completion.choices[0].message.content
        logging.info("Memory summarized.")
        # Replace detailed memory with summary
        # Embedding for summary isn't strictly needed for retrieval of detailed conversation, but could be added.
        # For simplicity, we'll store it without an embedding here.
        return [{"role": "system", "content": f"Previous conversation summary: {summary_text}"}] # Embedding is less relevant for a summary entry
    except Exception as e:
        logging.error(f"Error summarizing memory: {e}")
        return mem_list # Return original memory on failure


# --- Flask Routes ---

@app.route('/')
def index():
    """
    Serve the main chat interface page.
    """
    # Initialize memory in session if it doesn't exist
    if 'chat_memory' not in session:
        session['chat_memory'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle incoming chat messages, process with the bot logic,
    update session memory, and return the AI response.
    """
    if client is None or embedding_model is None:
         # Check if API key was missing or model failed to load at startup
         status_code = 500
         error_message = "Chatbot backend is not fully initialized (API key or embedding model missing)."
         logging.error(error_message)
         return jsonify({"response": error_message}), status_code


    user_input = request.json.get('message')
    if not user_input or not user_input.strip():
        return jsonify({"response": "Please enter a message."}), 400

    # Get memory from the session
    # Session data needs to be JSON serializable, embeddings are numpy arrays
    # We stored them as lists, retrieve_relevant_memory expects numpy. Handle conversion.
    current_memory_serializable = session.get('chat_memory', [])
    # Create a temporary list that converts embedding lists back to numpy for processing
    current_memory_for_processing = []
    for entry in current_memory_serializable:
        temp_entry = entry.copy() # Copy to avoid modifying session directly before commit
        if "embedding" in temp_entry and isinstance(temp_entry["embedding"], list):
             try:
                  temp_entry["embedding"] = np.array(temp_entry["embedding"])
                  current_memory_for_processing.append(temp_entry)
             except Exception as conv_e:
                  logging.warning(f"Failed to convert session embedding to numpy: {conv_e}")
                  # Skip this entry or handle error
                  pass # Just skip for now

    # Construct prompt using relevant memory from the current session memory
    # The construct_prompt function returns a list of messages for the API
    messages_for_api = construct_prompt(current_memory_for_processing, user_input)

    try:
        # Get response from the model
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instruct-fpt", # Use a suitable, available model
            messages=messages_for_api, # Pass the list of messages
            temperature=0.6,
            max_tokens=1024,  # Limit response length
            top_p=0.95,
            stream=False, # Disable streaming for simpler HTTP response handling
            stop=None,
        )
        ai_response_content = completion.choices[0].message.content # Access content correctly

    except Exception as e:
        logging.error(f"Error calling Groq API: {e}")
        # Provide a user-friendly error message
        ai_response_content = "Sorry, I encountered an error when trying to respond. Please try again later."
        # Optionally clear memory on API error if it might be corrupted
        # session['chat_memory'] = [] # Decide if you want to clear on error


    # --- Update Memory Buffer in Session ---
    # Use the original serializable memory list to add new entries
    # The add_to_memory function now returns the updated list
    current_memory_serializable = add_to_memory(current_memory_serializable, "user", user_input)
    current_memory_serializable = add_to_memory(current_memory_serializable, "assistant", ai_response_content)

    # Optionally trim memory to keep it manageable (e.g., last 20 turns)
    # You might want a larger size for better memory recall
    current_memory_serializable = trim_memory(current_memory_serializable, max_size=20)

    # Store the updated memory back into the session
    # Ensure embeddings are lists when stored
    session['chat_memory'] = current_memory_serializable

    # Return the AI response as JSON
    return jsonify({"response": ai_response_content})


# You can add a route to clear memory if needed (e.g., a "Start New Chat" button)
@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    """
    Clear the chat memory from the session.
    """
    session['chat_memory'] = []
    logging.info("Chat memory cleared.")
    return jsonify({"status": "Memory cleared."})


# --- Running the App ---
if __name__ == '__main__':
    logging.info("Starting Waitress server...")
    # --- IMPORTANT: Use port 7860 for Hugging Face Spaces Docker SDK ---
    # Use the PORT environment variable if set, otherwise default to 7860
    port = int(os.environ.get('PORT', 7860))
    serve(app, host='0.0.0.0', port=port)