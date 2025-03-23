// Show spinner with message
function showSpinner(message) {
    let spinnerSection = document.getElementById('spinnerSection');
    if (!spinnerSection) {
        spinnerSection = document.createElement('div');
        spinnerSection.id = 'spinnerSection';
        spinnerSection.style.position = 'fixed';
        spinnerSection.style.top = '50%';
        spinnerSection.style.left = '50%';
        spinnerSection.style.transform = 'translate(-50%, -50%)';
        spinnerSection.style.display = 'flex';
        spinnerSection.style.flexDirection = 'column';
        spinnerSection.style.alignItems = 'center';
        spinnerSection.style.justifyContent = 'center';
        spinnerSection.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
        spinnerSection.style.color = '#fff';
        spinnerSection.style.padding = '20px';
        spinnerSection.style.borderRadius = '10px';
        spinnerSection.style.zIndex = '1000';

        const spinner = document.createElement('div');
        spinner.id = 'loadingSpinner';
        spinner.style.width = '40px';
        spinner.style.height = '40px';
        spinner.style.border = '5px solid #f3f3f3';
        spinner.style.borderTop = '5px solid #3498db';
        spinner.style.borderRadius = '50%';
        spinner.style.animation = 'spin 1s linear infinite';
        spinnerSection.appendChild(spinner);

        const messageElement = document.createElement('p');
        messageElement.id = 'spinnerMessage';
        messageElement.style.marginTop = '15px';
        messageElement.style.fontSize = '16px';
        messageElement.style.fontWeight = 'bold';
        messageElement.innerText = message;
        spinnerSection.appendChild(messageElement);

        document.body.appendChild(spinnerSection);
    } else {
        document.getElementById('spinnerMessage').innerText = message;
        spinnerSection.style.display = 'flex';
    }
}

// Hide spinner
function hideSpinner() {
    const spinnerSection = document.getElementById('spinnerSection');
    if (spinnerSection) {
        spinnerSection.style.display = 'none';
    }
}

// Integrate visualization loading into trainModel
async function trainModel(event) {
    if (event) event.preventDefault();
    const output = document.getElementById('output');
    console.log("Train Model button clicked!");

    // Show loading spinner immediately
    showSpinner('Training the model...');

    const healthFile = document.getElementById('healthData').files[0];
    const wearableFiles = document.getElementById('wearableData').files;

    if (!healthFile || wearableFiles.length === 0) {
        showSpinner('Missing files. Upload health and wearable data.');
        setTimeout(hideSpinner, 3000);
        return;
    }

    const formData = new FormData();
    formData.append('healthFile', healthFile);
    for (let i = 0; i < wearableFiles.length; i++) {
        formData.append('wearableFiles', wearableFiles[i]);
    }

    let response;
    try {
        response = await fetch('http://127.0.0.1:5000/train', {
            method: 'POST',
            body: formData,
            mode: 'cors'
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
    } catch (error) {
        console.error('Fetch failed:', error);
        showSpinner('❌ Error: Could not connect to backend. Check if the server is running.');
        setTimeout(hideSpinner, 3000);
        return;
    }

    try {
        const data = await response.json();
        console.log('Train response:', data);
        showSpinner('✅ Model successfully trained!');
        setTimeout(hideSpinner, 2000);

        if (data.visualizations && data.visualizations.length > 0) {
            loadVisualizations(data.visualizations);
        } else {
            showSpinner('⚠️ No visualizations generated. Please check the training data.');
            setTimeout(hideSpinner, 3000);
        }
    } catch (error) {
        console.error('Error during training:', error);
        showSpinner('❌ Error: ' + (error.message || 'Could not connect to backend. Please try again later.'));
        setTimeout(hideSpinner, 3000);
    }
}