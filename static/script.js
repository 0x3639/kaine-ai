// Kaine AI Web Interface - Frontend JavaScript

// State management
let conversationHistory = [];
let currentSessionId = null;
let isForkedSession = false;

// DOM elements
const chatContainer = document.getElementById('chat-container');
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');
const loadingOverlay = document.getElementById('loading-overlay');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkForSharedSession();
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

        // Handle session forking if this is a loaded shared session
        if (currentSessionId && !isForkedSession) {
            // This is the first new message on a loaded session - fork it
            currentSessionId = null;
            isForkedSession = true;
            showToast('Started new conversation from shared link');
        }

        // Auto-save session after each Q&A
        try {
            await saveSession();

            // Show share button if this is the first message
            if (conversationHistory.length === 1) {
                const shareButton = document.getElementById('share-button');
                if (shareButton) {
                    shareButton.classList.remove('hidden');
                }
            }
        } catch (error) {
            console.error('Failed to save session:', error);
            // Don't show error to user - session save is background operation
        }

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

// Session management functions

async function checkForSharedSession() {
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('session');

    if (!sessionId) {
        return;
    }

    try {
        showLoading(true);

        // Try to load the session
        const response = await fetch(`/api/sessions/${sessionId}`);

        if (!response.ok) {
            throw new Error('Session not found');
        }

        const sessionData = await response.json();

        // Check if password is required
        if (sessionData.has_password && sessionData.messages.length === 0) {
            await promptForPassword(sessionId);
            return;
        }

        // Load the conversation
        await loadSession(sessionData);

    } catch (error) {
        console.error('Error loading session:', error);
        showError('Failed to load shared conversation');
    } finally {
        showLoading(false);
    }
}

async function promptForPassword(sessionId) {
    const password = prompt('This conversation is password protected. Please enter the password:');

    if (!password) {
        // User cancelled
        window.history.replaceState({}, '', window.location.pathname);
        return;
    }

    try {
        showLoading(true);

        const response = await fetch(`/api/sessions/${sessionId}/verify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ password: password })
        });

        if (!response.ok) {
            if (response.status === 403) {
                throw new Error('Incorrect password');
            }
            throw new Error('Failed to verify password');
        }

        const sessionData = await response.json();
        await loadSession(sessionData);

    } catch (error) {
        console.error('Error verifying password:', error);
        alert(error.message);
        // Try again or cancel
        await promptForPassword(sessionId);
    } finally {
        showLoading(false);
    }
}

async function loadSession(sessionData) {
    // Load messages into conversation history
    conversationHistory = sessionData.messages.map(msg => ({
        question: msg.question,
        answer: msg.answer,
        sources: msg.sources,
        timestamp: new Date(msg.timestamp)
    }));

    // Render all messages
    const container = document.querySelector('.container');
    const header = document.getElementById('header');

    if (container.classList.contains('initial-state')) {
        container.classList.remove('initial-state');
        header.classList.add('hidden');
    }

    conversationHistory.forEach(item => {
        addUserMessage(item.question);
        addAssistantMessage(item.answer, item.sources);
    });

    // Set session ID and mark as loaded (will fork on next question)
    currentSessionId = sessionData.session_id;
    isForkedSession = false;

    showToast('Conversation loaded successfully');
}

async function saveSession(password = null) {
    try {
        const messages = conversationHistory.map(item => ({
            question: item.question,
            answer: item.answer,
            sources: item.sources,
            timestamp: item.timestamp.toISOString()
        }));

        const payload = {
            session_id: currentSessionId,
            messages: messages
        };

        if (password) {
            payload.password = password;
        }

        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error('Failed to save session');
        }

        const data = await response.json();
        currentSessionId = data.session_id;

        return data;

    } catch (error) {
        console.error('Error saving session:', error);
        throw error;
    }
}

function showShareModal() {
    const modal = document.getElementById('share-modal');
    modal.classList.remove('hidden');
}

function hideShareModal() {
    const modal = document.getElementById('share-modal');
    modal.classList.add('hidden');
    // Reset form
    document.getElementById('password-checkbox').checked = false;
    document.getElementById('password-input').value = '';
    document.getElementById('password-input').classList.add('hidden');
    document.getElementById('share-link-display').classList.add('hidden');
}

async function generateShareLink() {
    const passwordCheckbox = document.getElementById('password-checkbox');
    const passwordInput = document.getElementById('password-input');
    const password = passwordCheckbox.checked ? passwordInput.value : null;

    if (passwordCheckbox.checked && !password) {
        alert('Please enter a password');
        return;
    }

    try {
        const sessionData = await saveSession(password);
        const shareUrl = `${window.location.origin}${window.location.pathname}?session=${sessionData.session_id}`;

        // Display the link
        const linkDisplay = document.getElementById('share-link-display');
        const linkInput = document.getElementById('share-link-input');
        linkInput.value = shareUrl;
        linkDisplay.classList.remove('hidden');

        // Auto-copy to clipboard
        try {
            await navigator.clipboard.writeText(shareUrl);
            showToast('Link generated and copied to clipboard!');
        } catch (err) {
            // Fallback for older browsers
            linkInput.select();
            document.execCommand('copy');
            showToast('Link generated and copied to clipboard!');
        }

    } catch (error) {
        console.error('Error generating share link:', error);
        alert('Failed to generate share link');
    }
}

async function copyShareLink() {
    const linkInput = document.getElementById('share-link-input');

    try {
        // Modern clipboard API
        await navigator.clipboard.writeText(linkInput.value);
        showToast('Link copied to clipboard!');
    } catch (err) {
        // Fallback for older browsers
        linkInput.select();
        document.execCommand('copy');
        showToast('Link copied to clipboard!');
    }
}

function togglePasswordInput() {
    const checkbox = document.getElementById('password-checkbox');
    const passwordInput = document.getElementById('password-input');

    if (checkbox.checked) {
        passwordInput.classList.remove('hidden');
        passwordInput.focus();
    } else {
        passwordInput.classList.add('hidden');
    }
}

function showToast(message) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.remove('hidden');

    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

// Export conversation (useful for debugging)
window.exportConversation = function() {
    console.log('Conversation History:', conversationHistory);
    return conversationHistory;
};
