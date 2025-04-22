document.getElementById('toggleBtn').addEventListener('click', () => {
  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, {action: 'toggle'}, (response) => {
      if (chrome.runtime.lastError) {
        console.error(chrome.runtime.lastError.message);
      }
    });
  });
});
