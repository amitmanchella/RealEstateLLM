// Main content script that runs on real estate websites

// Create and inject the chatbot UI
function createChatbotUI() {
  const container = document.createElement('div');
  container.className = 'real-estate-advisor-container';
  container.innerHTML = `
    <div class="real-estate-advisor-header">
      <h3 class="real-estate-advisor-title">RealEstate Advisor</h3>
      <div class="real-estate-advisor-controls">
        <button class="real-estate-advisor-minimize">_</button>
        <button class="real-estate-advisor-close">×</button>
      </div>
    </div>
    <div class="real-estate-advisor-chat">
      <div class="real-estate-advisor-message real-estate-advisor-bot">
        Hi there! I can answer questions about this property or general real estate topics. What would you like to know?
      </div>
    </div>
    <div class="real-estate-advisor-input">
      <input type="text" placeholder="Ask a question...">
      <button>➤</button>
    </div>
  `;
  
  document.body.appendChild(container);
  
  // Make the chatbot draggable
  makeDraggable(container);
  
  // Add event listeners
  setupEventListeners(container);
  
  return container;
}

// Make the chatbot draggable
function makeDraggable(element) {
  const header = element.querySelector('.real-estate-advisor-header');
  let isDragging = false;
  let offsetX, offsetY;
  
  header.addEventListener('mousedown', (e) => {
    isDragging = true;
    offsetX = e.clientX - element.getBoundingClientRect().left;
    offsetY = e.clientY - element.getBoundingClientRect().top;
  });
  
  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    
    const x = e.clientX - offsetX;
    const y = e.clientY - offsetY;
    
    element.style.left = `${x}px`;
    element.style.top = `${y}px`;
    element.style.right = 'auto';
    element.style.bottom = 'auto';
  });
  
  document.addEventListener('mouseup', () => {
    isDragging = false;
  });
}

// Set up event listeners for the chatbot
function setupEventListeners(container) {
  const minimizeBtn = container.querySelector('.real-estate-advisor-minimize');
  const closeBtn = container.querySelector('.real-estate-advisor-close');
  const input = container.querySelector('input');
  const sendBtn = container.querySelector('.real-estate-advisor-input button');
  const chat = container.querySelector('.real-estate-advisor-chat');
  
  // Minimize/maximize the chatbot
  minimizeBtn.addEventListener('click', () => {
    container.classList.toggle('real-estate-advisor-collapsed');
    minimizeBtn.textContent = container.classList.contains('real-estate-advisor-collapsed') ? '□' : '_';
  });
  
  // Close the chatbot
  closeBtn.addEventListener('click', () => {
    container.remove();
  });
  
  // Send a message
  function sendMessage() {
    const message = input.value.trim();
    if (!message) return;
    
    // Add user message to chat
    const userMsg = document.createElement('div');
    userMsg.className = 'real-estate-advisor-message real-estate-advisor-user';
    userMsg.textContent = message;
    chat.appendChild(userMsg);
    
    // Clear input
    input.value = '';
    
    // Get property data from the page
    const propertyData = extractPropertyData();
    
    // Send message to background script
    chrome.runtime.sendMessage({
      action: 'query',
      query: message,
      propertyData: propertyData
    }, response => {
      // Add bot response to chat
      const botMsg = document.createElement('div');
      botMsg.className = 'real-estate-advisor-message real-estate-advisor-bot';
      botMsg.textContent = response.answer;
      chat.appendChild(botMsg);
      
      // Scroll to bottom
      chat.scrollTop = chat.scrollHeight;
    });
  }
  
  // Send message on button click
  sendBtn.addEventListener('click', sendMessage);
  
  // Send message on Enter key
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
  });
}

// Extract property data from the current page
function extractPropertyData() {
  let data = {
    url: window.location.href,
    website: ''
  };
  
  // Determine which website we're on
  if (window.location.hostname.includes('zillow.com')) {
    data.website = 'zillow';
    data = {...data, ...extractZillowData()};
  } else if (window.location.hostname.includes('redfin.com')) {
    data.website = 'redfin';
    data = {...data, ...extractRedfinData()};
  } else if (window.location.hostname.includes('realtor.com')) {
    data.website = 'realtor';
    data = {...data, ...extractRealtorData()};
  }
  
  return data;
}

// Extract data from Zillow
function extractZillowData() {
  const data = {};
  
  try {
    // Price
    const priceElement = document.querySelector('[data-testid="price"]');
    if (priceElement) data.price = priceElement.textContent.trim();
    
    // Address
    const addressElement = document.querySelector('h1');
    if (addressElement) data.address = addressElement.textContent.trim();
    
    // Beds/Baths/SqFt
    const summaryElements = document.querySelectorAll('[data-testid="bed-bath-beyond"]');
    summaryElements.forEach(element => {
      const text = element.textContent.toLowerCase();
      if (text.includes('bed')) data.beds = text.replace(/[^\d.]/g, '');
      if (text.includes('bath')) data.baths = text.replace(/[^\d.]/g, '');
      if (text.includes('sqft')) data.sqft = text.replace(/[^\d.]/g, '');
    });
  } catch (e) {
    console.error('Error extracting Zillow data:', e);
  }
  
  return data;
}

// Extract data from Redfin
function extractRedfinData() {
  const data = {};
  
  try {
    // Price
    const priceElement = document.querySelector('.price');
    if (priceElement) data.price = priceElement.textContent.trim();
    
    // Address
    const addressElement = document.querySelector('.street-address');
    if (addressElement) data.address = addressElement.textContent.trim();
    
    // Beds/Baths/SqFt
    const bedsElement = document.querySelector('[data-rf-test-id="abp-beds"]');
    if (bedsElement) data.beds = bedsElement.textContent.replace(/[^\d.]/g, '');
    
    const bathsElement = document.querySelector('[data-rf-test-id="abp-baths"]');
    if (bathsElement) data.baths = bathsElement.textContent.replace(/[^\d.]/g, '');
    
    const sqftElement = document.querySelector('[data-rf-test-id="abp-sqFt"]');
    if (sqftElement) data.sqft = sqftElement.textContent.replace(/[^\d.]/g, '');
  } catch (e) {
    console.error('Error extracting Redfin data:', e);
  }
  
  return data;
}

// Extract data from Realtor.com
function extractRealtorData() {
  const data = {};
  
  try {
    // Price
    const priceElement = document.querySelector('[data-testid="list-price"]');
    if (priceElement) data.price = priceElement.textContent.trim();
    
    // Address
    const addressElement = document.querySelector('[data-testid="address-container"]');
    if (addressElement) data.address = addressElement.textContent.trim();
    
    // Beds/Baths/SqFt
    const summaryElements = document.querySelectorAll('[data-testid="property-meta-container"]');
    summaryElements.forEach(element => {
      const text = element.textContent.toLowerCase();
      if (text.includes('bed')) data.beds = text.replace(/[^\d.]/g, '');
      if (text.includes('bath')) data.baths = text.replace(/[^\d.]/g, '');
      if (text.includes('sqft')) data.sqft = text.replace(/[^\d.]/g, '');
    });
  } catch (e) {
    console.error('Error extracting Realtor data:', e);
  }
  
  return data;
}

// Initialize on page load
window.addEventListener('load', () => {
  // Check if we should show the chatbot (user might have closed it)
  chrome.storage.local.get(['showChatbot'], (result) => {
    if (result.showChatbot !== false) {
      createChatbotUI();
    }
  });
});

// Listen for messages from popup or background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'toggle') {
    const existingChatbot = document.querySelector('.real-estate-advisor-container');
    
    if (existingChatbot) {
      existingChatbot.remove();
      chrome.storage.local.set({showChatbot: false});
    } else {
      createChatbotUI();
      chrome.storage.local.set({showChatbot: true});
    }
    
    sendResponse({success: true});
  }
});
