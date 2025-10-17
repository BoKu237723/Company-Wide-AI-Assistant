# Company-Wide-AI-Assistant (Version-2.0.0)

<p align="center">
  <img src="https://raw.githubusercontent.com/BoKu237723/Repo-Images/main/Company-Wide-AI-Assistant.png" alt="Company Wide AI Assistant" width="500"/>
</p>

## Description

A sophisticated AI assistant that provides department-specific information by automatically scanning and processing company documents from Google Drive. Uses Ollama's local language models with direct integration to your organization's document management system.

## ✨ New Features — Version 2.0

### 🔗 Google Drive Integration  
Seamlessly connect your workspace with Google Drive.  
- **Automatic authentication** with OAuth 2.0  
- **Secure token persistence** using encrypted pickle files  
- **Direct access** to company documents without manual downloads  

---

### 📁 Multi-Format Document Support  
Enhanced compatibility for diverse document types:  
- **Google Docs** 
- **Microsoft Word (.docx)** 
- **PDF Documents** 
- **Plain Text Files**

---

### 🔍 Smart Document Discovery  
Smarter automation for report and directory management:  
- **Automatic folder scanning** for `"Company Reports"` directory  
- **Department-specific detection** (Finance, Marketing, IT)  
- **Weekly report recognition** via `Week-XX` pattern  
- **File type filtering** and validation for clean processing  

---

### 💾 In-Memory Processing  
Optimized for performance and security:  
- **No local file storage** — all operations handled in memory  
- **Memory-efficient** processing for large documents  
- **Secure handling** without temporary files  

---

🚀 *Version 2.0 marks a major step toward full automation, smarter document discovery, and enhanced data security.*


## Requirements

- Python 3.6+
- Ollama installed locally
- Llama 3.1:8B model (or compatible model)
- Google Cloud Project with Drive API enabled
- Google OAuth 2.0 credentials

## Installation

### 1. Install Ollama

```bash
# Visit https://ollama.ai/download and follow installation instructions for your OS
# Or use one of these commands:

# macOS
# curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download

# Linux
# curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull the Required Model
```bash
ollama pull llama3.1:8b
```
### 3. Install Python Dependencies
```python 
pip install ollama-python
pip install ollama-python google-auth-oauthlib google-auth-httplib2 google-api-python-client
```
## File Structure
All these files must be in same folder
```
index.py                # main program file
ai_prompt.txt           # AI prompt template
credentials.json        # Google OAuth 2.0 credentials (from Google Cloud)
drive_token.pickle      # Auto-generated authentication token
```
## 🗂️ Required Google Drive Structure
For optimal performance, organize your Google Drive as follows

```
Google Drive/
└── Company Reports/
    ├── finance/
    │   ├── Week-1.docx
    │   ├── Week-2.pdf
    │   └── Week-3.txt
    ├── marketing/
    │   ├── Week-1.gdoc
    │   └── Campaign-Report.txt
    └── IT/
        ├── Week-1.docx
        └── System-Status.pdf
```

## Possible Next Features Updates
These following are ideas for features that could be added in future versions.

### Confident scoring
- Automatic department classification with confidence scoring, allowing user directly ask question without needing to clarify department.
