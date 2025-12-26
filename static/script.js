let fileContents = { architecture: '', prd: '' };
let currentDocumentId = null;
const API_BASE = 'http://127.0.0.1:8000';

document.getElementById('archFile').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) await handleFileUpload(file, 'Architecture', 'architecture');
});

document.getElementById('prdFile').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) await handleFileUpload(file, 'PRD', 'prd');
});

async function handleFileUpload(file, fileType, contentKey) {
    const status = document.getElementById('status');
    status.innerHTML = `<div style="color: #007bff;">Processing ${fileType} file...</div>`;
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE}/upload-file`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            fileContents[contentKey] = data.extracted_text;
            status.innerHTML = `<div class="success">✅ ${fileType} file processed!</div>`;
        } else {
            throw new Error(data.error || 'File processing failed');
        }
    } catch (error) {
        fileContents[contentKey] = '';
        status.innerHTML = `<div class="error">❌ ${fileType} Error: ${error.message}</div>`;
    }
}

document.getElementById('analysisForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const repoUrl = document.getElementById('repoUrl').value;
    const analyzeBtn = document.getElementById('analyzeBtn');
    const status = document.getElementById('status');
    const results = document.getElementById('results');
    
    if (!fileContents.architecture) {
        status.innerHTML = '<div class="error">❌ Please upload an Architecture document!</div>';
        return;
    }
    
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '🔄 Analyzing...';
    status.innerHTML = '<div style="color: #007bff;">🔄 Analyzing repository...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                repo_url: repoUrl,
                architecture_content: fileContents.architecture,
                prd_content: fileContents.prd
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentDocumentId = data.document_id;
            document.getElementById('analysisContent').innerHTML = 
                `<pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; white-space: pre-wrap;">${data.analysis}</pre>`;
            results.style.display = 'block';
            status.innerHTML = '<div class="success">✅ Analysis completed!</div>';
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        status.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = '🔍 Analyze Project';
    }
});

async function downloadDocument(docType, format) {
    if (!currentDocumentId) {
        alert('No documents available');
        return;
    }
    
    try {
        const url = `${API_BASE}/download/${currentDocumentId}/${docType}/${format}`;
        const response = await fetch(url);
        
        if (!response.ok) throw new Error('Download failed');
        
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `${docType}_${currentDocumentId}.${format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
        alert(`Download failed: ${error.message}`);
    }
}