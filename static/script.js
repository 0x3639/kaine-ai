// Kaine AI Web Interface - Frontend JavaScript

// State management
let conversationHistory = [];

// DOM elements
const chatContainer = document.getElementById('chat-container');
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');
const loadingOverlay = document.getElementById('loading-overlay');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    questionInput.focus();
});

function setupEventListeners() {
    // Send button click
    sendButton.addEventListener('click', handleSendQuestion);

    // Enter key to send (Shift+Enter for new line)
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendQuestion();
        }
    });

    // Auto-resize textarea
    questionInput.addEventListener('input', () => {
        autoResizeTextarea();
    });
}

function autoResizeTextarea() {
    questionInput.style.height = 'auto';
    questionInput.style.height = questionInput.scrollHeight + 'px';
}

async function handleSendQuestion() {
    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    if (question.length > 1000) {
        showError('Question is too long (max 1000 characters)');
        return;
    }

    // Disable input while processing
    setInputEnabled(false);

    // Add user message to chat
    addUserMessage(question);

    // Clear input
    questionInput.value = '';
    questionInput.style.height = 'auto';

    // Show loading
    showLoading(true);

    try {
        // Send question to API
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });

        if (!response.ok) {
            const errorData = await response.json();

            if (response.status === 429) {
                // Rate limit exceeded
                const detail = errorData.detail;
                const message = detail.message || 'Rate limit exceeded. Please try again later.';
                throw new Error(message);
            } else {
                throw new Error(errorData.detail || 'Failed to get answer');
            }
        }

        const data = await response.json();

        // Add assistant message to chat
        addAssistantMessage(data.answer, data.sources);

        // Store in conversation history
        conversationHistory.push({
            question: question,
            answer: data.answer,
            sources: data.sources,
            timestamp: new Date()
        });

    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    } finally {
        showLoading(false);
        setInputEnabled(true);
        questionInput.focus();
    }
}

function addUserMessage(text) {
    // Transition from initial state to chat state
    const container = document.querySelector('.container');
    const header = document.getElementById('header');

    if (container.classList.contains('initial-state')) {
        container.classList.remove('initial-state');
        header.classList.add('hidden');
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.id = 'latest-user-message'; // Add ID to track latest user message

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = text;

    contentDiv.appendChild(textDiv);
    messageDiv.appendChild(contentDiv);

    // Remove ID from previous user message if it exists
    const previousUserMessage = document.getElementById('latest-user-message');
    if (previousUserMessage && previousUserMessage !== messageDiv) {
        previousUserMessage.removeAttribute('id');
    }

    chatContainer.appendChild(messageDiv);

    // Scroll to show the user's question at the top
    messageDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function addAssistantMessage(text, sources) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Copy button
    const copyButton = document.createElement('button');
    copyButton.className = 'copy-button';
    copyButton.textContent = 'Copy';
    copyButton.onclick = () => copyToClipboard(text, copyButton);
    contentDiv.appendChild(copyButton);

    // Message text
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';

    // Convert URLs to clickable links
    const linkedText = linkifyUrls(text);
    textDiv.innerHTML = linkedText;

    contentDiv.appendChild(textDiv);

    // Sources removed per user request

    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);

    // Don't scroll - keep the user's question visible at the top
    // User can scroll down to read long responses
}

function createSourcesElement(sources) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources';

    const titleDiv = document.createElement('div');
    titleDiv.className = 'sources-title';
    titleDiv.textContent = 'Sources';
    sourcesDiv.appendChild(titleDiv);

    sources.forEach(source => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';

        // Date
        const dateDiv = document.createElement('div');
        dateDiv.className = 'source-date';
        dateDiv.textContent = source.date;
        sourceItem.appendChild(dateDiv);

        // Author
        const authorDiv = document.createElement('div');
        authorDiv.className = 'source-author';
        authorDiv.textContent = source.author;
        sourceItem.appendChild(authorDiv);

        // Link
        if (source.url) {
            const linkDiv = document.createElement('div');
            linkDiv.className = 'source-link';

            const link = document.createElement('a');
            link.href = source.url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = 'View on Telegram →';

            linkDiv.appendChild(link);

            // Relevance score
            if (source.relevance_score) {
                const scoreSpan = document.createElement('span');
                scoreSpan.className = 'source-score';
                scoreSpan.textContent = `(${(source.relevance_score * 100).toFixed(0)}% relevant)`;
                linkDiv.appendChild(scoreSpan);
            }

            sourceItem.appendChild(linkDiv);
        }

        sourcesDiv.appendChild(sourceItem);
    });

    return sourcesDiv;
}

function convertMarkdownLinks(text) {
    // Convert markdown links [text](url) to HTML links
    const markdownLinkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    return text.replace(markdownLinkRegex, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

function linkifyUrls(text) {
    // First convert markdown links
    text = convertMarkdownLinks(text);

    // Split by existing <a> tags to avoid double-linking
    const parts = text.split(/(<a[^>]*>.*?<\/a>)/g);

    // Only linkify parts that aren't already links
    for (let i = 0; i < parts.length; i++) {
        if (!parts[i].startsWith('<a')) {
            // Convert plain URLs to links
            const urlRegex = /(https?:\/\/[^\s<]+)/g;
            parts[i] = parts[i].replace(urlRegex, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
        }
    }

    return parts.join('');
}

function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        button.textContent = 'Copied!';
        button.classList.add('copied');

        setTimeout(() => {
            button.textContent = 'Copy';
            button.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        button.textContent = 'Failed';
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000);
    });
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;

    chatContainer.appendChild(errorDiv);
    scrollToBottom();

    // Remove error message after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

function showLoading(show) {
    if (show) {
        loadingOverlay.classList.remove('hidden');
    } else {
        loadingOverlay.classList.add('hidden');
    }
}

function setInputEnabled(enabled) {
    questionInput.disabled = !enabled;
    sendButton.disabled = !enabled;
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Export conversation (useful for debugging)
window.exportConversation = function() {
    console.log('Conversation History:', conversationHistory);
    return conversationHistory;
};
