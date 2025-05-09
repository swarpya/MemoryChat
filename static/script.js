document.addEventListener('DOMContentLoaded', () => {
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearMemoryBtn = document.getElementById('clearMemoryBtn'); // Get the new button

    // Function to append a message to the chatbox
    function appendMessage(sender, text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);

        const p = document.createElement('p');
        // Use textContent to avoid HTML injection
        p.textContent = text;

        messageElement.appendChild(p);
        chatbox.appendChild(messageElement);

        // Auto-scroll to the bottom
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    // Function to send message to backend
    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (messageText === "") {
            return; // Don't send empty messages
        }

        // Append user message to chatbox
        appendMessage('user', messageText);

        // Clear input and disable buttons
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;
        if (clearMemoryBtn) clearMemoryBtn.disabled = true;


        // Optional: Add a loading indicator
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('message', 'assistant-message', 'loading-indicator');
        loadingElement.textContent = 'AI is thinking...';
        chatbox.appendChild(loadingElement);
        chatbox.scrollTop = chatbox.scrollHeight; // Scroll to show indicator


        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: messageText }),
            });

            // Remove loading indicator
            if (chatbox.contains(loadingElement)) {
                 chatbox.removeChild(loadingElement);
            }


            if (!response.ok) {
                 const errorData = await response.json().catch(() => ({})); // Try parsing JSON, fallback to empty object
                 const errorMessage = errorData.response || `Server returned status ${response.status}: ${response.statusText}`;
                 throw new Error(`Chat request failed: ${errorMessage}`);
            }

            const data = await response.json();
            appendMessage('assistant', data.response);

        } catch (error) {
            console.error('Error sending message:', error);
             // Remove loading indicator if still present
             if (chatbox.contains(loadingElement)) {
                 chatbox.removeChild(loadingElement);
             }
            appendMessage('assistant', `Error: Could not get a response. ${error.message}`);
        } finally {
            // Re-enable input and buttons
            userInput.disabled = false;
            sendBtn.disabled = false;
            if (clearMemoryBtn) clearMemoryBtn.disabled = false;
            userInput.focus(); // Put focus back on input field
        }
    }

     // Function to clear memory
    async function clearMemory() {
        if (confirm("Are you sure you want to clear the chat history? This will start a new conversation.")) {
             // Disable buttons during clear
             userInput.disabled = true;
             sendBtn.disabled = true;
             if (clearMemoryBtn) clearMemoryBtn.disabled = true;

             try {
                const response = await fetch('/clear_memory', {
                    method: 'POST',
                });

                if (!response.ok) {
                     const errorData = await response.json().catch(() => ({}));
                     const errorMessage = errorData.status || `Server returned status ${response.status}: ${response.statusText}`;
                    throw new Error(`Clear memory request failed: ${errorMessage}`);
                }

                // Clear the chatbox UI and add initial message
                chatbox.innerHTML = ''; // Clear all messages
                appendMessage('assistant', 'Chat history cleared. Starting a new conversation!');
                appendMessage('assistant', 'Hello! I\'m an AI chatbot with memory. How can I help you today?');


             } catch (error) {
                console.error('Error clearing memory:', error);
                appendMessage('assistant', `Error: Could not clear chat history. ${error.message}`);
             } finally {
                 // Re-enable buttons
                 userInput.disabled = false;
                 sendBtn.disabled = false;
                 if (clearMemoryBtn) clearMemoryBtn.disabled = false;
                 userInput.focus();
             }
        }
    }


    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent newline in input
            sendMessage();
        }
    });

    // Add event listener for the clear memory button
    if (clearMemoryBtn) {
         clearMemoryBtn.addEventListener('click', clearMemory);
    }

    // Initial welcome message is already in index.html
});