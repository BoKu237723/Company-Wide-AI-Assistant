# Company-Wide-AI-Assistant (Version-2.0.0)

<p align="center">
  <img src="https://raw.githubusercontent.com/BoKu237723/Repo-Images/main/Company-Wide-AI-Assistant.png" alt="Company Wide AI Assistant" width="500"/>
</p>

## Description

A sophisticated AI assistant that provides department-specific information by automatically scanning and processing company documents from Google Drive. Uses Ollama's local language models with direct integration to your organization's document management system.

## ✨ What's New in version 2.1

> ### 🧠 RNN-Powered Department Classification  
> Intelligent department detection using Recurrent Neural Networks:
> - **Mathematical RNN equations**: `h_t = tanh(Whx*x_t + Whh*h_{t-1} + b_h)`
> - **Enhanced classification**: Combines RNN speed with LLM semantic understanding
> - **Confidence scoring**: Automatic threshold-based department selection
> - **Continuous learning**: Improves from user interactions over time

> ### 🔄 Smart Multi-Department Querying  
> Ask questions once, get answers from all relevant departments:
> - **Automatic department detection**: No need to specify department manually
> - **Confidence-based filtering**: Only queries departments above threshold
> - **Combined insights**: Presents answers from multiple relevant departments
> - **Visual confidence indicators**: Color-coded confidence levels (🟢🟡🔴)

> ### 💾 Persistent Model Training  
> Model improvements are saved and reused:
> - **Auto-save functionality**: Best models automatically saved during training
> - **Training history**: Track performance improvements over time
> - **Vocabulary persistence**: Custom vocabulary built from training data
> - **Model status tracking**: View training date, accuracy, and sample count

> ### 📚 Continuous Learning System  
> The system learns and improves from usage:
> - **Interaction logging**: All queries and answers stored for training
> - **Automatic retraining**: Triggers after every 5 interactions
> - **Incremental learning**: Adds new unique questions to training set
> - **Performance monitoring**: Accuracy tracking and model validation

## 🎯 Existing Features from Previous Versions

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

🚀 *Version 2.1 marks a major step toward custom RNN, smarter multiple department query, and enhanced fall backs.*


## Requirements

- Python 3.6+
- PyTorch (for RNN functionality)
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
pip install torch torchvision torchaudio
```
## File Structure
All these files must be in same folder
```
index.py                # main program file
rnn.py                  # NEW: RNN classification and training system
ai_prompt.txt           # AI prompt template
credentials.json        # Google OAuth 2.0 credentials (from Google Cloud)
drive_token.pickle      # Auto-generated authentication token
rnn_models/             # NEW: Auto-created directory for model persistence
    ├── department_classifier_rnn.pth
    ├── vocab.pth
    ├── training_data.json
    ├── model_info.json
    └── performance_history.json
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

## 📝 Technical Notes

### GPU/CPU Compatibility
> **Important**: The current implementation assumes GPU availability for PyTorch operations. If you're running on CPU-only systems, you'll need to modify the model loading in `rnn.py`:
> 
> ```python
> # Change this line in setup_rnn_model():
> self.rnn_model = torch.load(paths['model'], map_location='cpu', weights_only=False)
> ```
> 
> The code currently uses `torch.load()` without `map_location='cpu'` for optimal GPU performance.

## 💌 Possible Next Features Updates
These following are ideas for features that could be added in future versions.

### Enhanced Training Data & MongoDB Integration
- Expended Initial Training Data
- MongoDB Atlas Integration
- Custom LSTM Architecture
