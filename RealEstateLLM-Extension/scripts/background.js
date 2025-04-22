// Backend server URL - update this to your server URL if different
const API_URL = 'http://localhost:5000/api/query';

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'query') {
    // Send query to backend server
    fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: message.query,
        property_data: message.propertyData
      })
    })
    .then(response => response.json())
    .then(data => {
      sendResponse({answer: data.answer});
    })
    .catch(error => {
      console.error('Error:', error);
      sendResponse({
        answer: "Sorry, I couldn't connect to the server. Please try again later."
      });
    });
    
    // Return true to indicate we'll respond asynchronously
    return true;
  }
});
