document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const predictorForm = document.getElementById("predictor-form");
    const rankInput = document.getElementById("rank-input");
    const categoryInput = document.getElementById("category-input");
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const sendChatBtn = document.getElementById("send-chat-btn");
    const resultsGrid = document.getElementById("results-grid");
    const filterTabs = document.querySelectorAll(".filter-tab");
    
    // Application State
    let chatbotState = {
        rank: null,
        category: null,
        courses: ["CS", "IS", "EC", "AI"]
    };
    let currentPredictions = [];
    let activeFilter = "all";

    // Initialize UI
    function init() {
        // Form submit handler
        predictorForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const checkedBranches = Array.from(document.querySelectorAll('input[name="branches"]:checked')).map(cb => cb.value);
            
            if (checkedBranches.length === 0) {
                alert("Please select at least one branch.");
                return;
            }

            const payload = {
                rank: parseInt(rankInput.value),
                category: categoryInput.value,
                courses: checkedBranches
            };

            // Update local state
            chatbotState.rank = payload.rank;
            chatbotState.category = payload.category;
            chatbotState.courses = payload.courses;

            fetchPredictions(payload);
        });

        // Chat send handlers
        sendChatBtn.addEventListener("click", handleChatSend);
        chatInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") {
                handleChatSend();
            }
        });

        // Filter tab handlers
        filterTabs.forEach(tab => {
            tab.addEventListener("click", () => {
                filterTabs.forEach(t => t.classList.remove("active"));
                tab.classList.add("active");
                activeFilter = tab.getAttribute("data-filter");
                renderCards();
            });
        });
    }

    // Call predictor API
    async function fetchPredictions(payload) {
        showLoadingState();
        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if (data.error) {
                showError(data.error);
                return;
            }

            currentPredictions = data.predictions || [];
            renderCards();

            // Synchronize the forms values to matches state
            rankInput.value = payload.rank;
            categoryInput.value = payload.category;
            
            // Add a bot confirmation message in the chat
            appendMessage("bot", `I've calculated your recommendations for Rank **${payload.rank}** and Category **${payload.category}**!`);
        } catch (error) {
            console.error("Error fetching predictions:", error);
            showError("Could not calculate cutoffs. Server offline.");
        }
    }

    // Chatbot flow
    async function handleChatSend() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Add user bubble
        appendMessage("user", text);
        chatInput.value = "";

        // Send to chat API
        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, state: chatbotState })
            });
            const data = await response.json();

            // Update cached state
            chatbotState = data.state || chatbotState;

            // Render bot message
            appendMessage("bot", data.reply);

            // Update UI prediction cards if recommendations are returned
            if (data.predictions && data.predictions.length > 0) {
                currentPredictions = data.predictions;
                renderCards();

                // Sync the form controls
                if (chatbotState.rank) rankInput.value = chatbotState.rank;
                if (chatbotState.category) categoryInput.value = chatbotState.category;
            }
        } catch (error) {
            console.error("Error sending chat:", error);
            appendMessage("bot", "Oops, I encountered a connection issue. Please check if the backend server is running.");
        }
    }

    // Append Chat Bubble
    function appendMessage(sender, text) {
        const msgDiv = document.createElement("div");
        msgDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");
        
        // Simple markdown parsing for bold text
        let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedText = formattedText.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        msgDiv.innerHTML = `<p>${formattedText}</p>`;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Render Cards in Grid
    function renderCards() {
        resultsGrid.innerHTML = "";
        
        const filtered = currentPredictions.filter(item => {
            if (activeFilter === "all") return true;
            return item.status === activeFilter;
        });

        if (filtered.length === 0) {
            resultsGrid.innerHTML = `
                <div class="empty-state">
                    <p>No colleges found matching the filter: <strong>${activeFilter}</strong></p>
                </div>
            `;
            return;
        }

        filtered.forEach(item => {
            const card = document.createElement("div");
            card.className = "card";
            card.innerHTML = `
                <div class="card-header">
                    <span class="college-code">${item.college_code}</span>
                    <span class="badge ${item.badge_color}">${item.status}</span>
                </div>
                <h4 class="college-name">${item.college_name}</h4>
                <p class="course-name">${item.course_name} (${item.course_code})</p>
                
                <div class="cutoff-section">
                    <div class="cutoff-item">
                        <span class="cutoff-label">Round 1</span>
                        <span class="cutoff-value">${item.round_1_cutoff || "N/A"}</span>
                    </div>
                    <div class="cutoff-item">
                        <span class="cutoff-label">Round 3</span>
                        <span class="cutoff-value">${item.round_3_cutoff || "N/A"}</span>
                    </div>
                    <div class="cutoff-item">
                        <span class="cutoff-label">2026 Adj</span>
                        <span class="cutoff-value adjusted">${item.adjusted_cutoff || "N/A"}</span>
                    </div>
                </div>
                
                <p class="card-explanation">${item.explanation}</p>
            `;
            resultsGrid.appendChild(card);
        });
    }

    function showLoadingState() {
        resultsGrid.innerHTML = `
            <div class="empty-state">
                <svg class="loading-spin" viewBox="0 0 24 24" width="36" height="36" stroke="currentColor" stroke-width="2" fill="none">
                    <circle cx="12" cy="12" r="10" stroke-dasharray="30" stroke-dashoffset="10"></circle>
                </svg>
                <p>Calculating predictions & seat-matrix thresholds...</p>
            </div>
        `;
    }

    function showError(msg) {
        resultsGrid.innerHTML = `
            <div class="empty-state">
                <p style="color: var(--dream-color);">${msg}</p>
            </div>
        `;
    }

    init();
});
