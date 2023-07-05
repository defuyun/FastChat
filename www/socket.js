const chatbotMessages = document.getElementById("chatbot-messages");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const sessionId = (Math.random() + 1).toString(36).substring(7);
const socket = io("https://llm-commercial-dev.eng.nianticlabs.com", {
    path: "/model/falcon40b/ws/socket.io"
}
);

var currentMessageText = null

sendButton.addEventListener("click", () => {
  const message = userInput.value.trim();

  if (message !== "") {
    addMessage("user", message);
    userInput.value = "";
    socket.emit("message", message);
    addMessage("bot", "");
    userInput.disabled = true;
  }
});

socket.on("message", data => {
    if(currentMessageText != null) {
        currentMessageText.innerHTML += data.replace(/\n/g, "<br>");
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    if(data.length == 0) {
        userInput.disabled = false;
        userInput.focus();
    }
});

userInput.addEventListener("keyup", function (e) {
    if (e.keyCode == 13) {
        e.preventDefault()
        sendButton.click()
    }
})

window.addEventListener("beforeunload", function (e) {
    socket.emit("message", "");
});

function addMessage(sender, message) {
    const div = document.createElement("div");
    div.classList.add("message", sender.replace(" ", "-"));
  
    const senderLabel = document.createElement("div");
    senderLabel.classList.add("sender-label");
    senderLabel.innerText = sender;
    div.appendChild(senderLabel);
  
    const messageText = document.createElement("pre");
    messageText.classList.add("message-text");
    messageText.innerText = message;
    div.appendChild(messageText);
    currentMessageText = messageText
  
    chatbotMessages.appendChild(div);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
}