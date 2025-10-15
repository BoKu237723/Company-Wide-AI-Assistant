# Company-Wide-AI-Assistant (Version-1.0.0 Beta)

<p align="center">
  <img src="https://raw.githubusercontent.com/BoKu237723/Repo-Images/main/Company-Wide-AI-Assistant.png" alt="Company Wide AI Assistant" width="500"/>
</p>
<br>
<p align="center">
  <img src="https://github.com/BoKu237723/Repo-Images/blob/main/python-logo.png" alt="Python Logo" width="200"/>
  <img src="https://github.com/BoKu237723/Repo-Images/blob/main/ollama-1.png" alt="Ollama Logo" width="180"/>
</p>
<br>

## Description

A Python-based AI assistant that provides department-specific information using Ollama's local language models. The system can answer questions about different company departments by leveraging department-specific data files and customizable prompts.

## Features

- **Multi-Department Support**: Currently supports Finance, Marketing, and IT departments
- **Local AI Processing**: Uses Ollama with the Llama 3.1 8B model for privacy and offline capability
- **Customizable Data**: Each department has its own data file for specialized knowledge
- **Flexible Prompt System**: Uses template-based prompts that can be easily modified

## Requirements

- Python 3.6+
- Ollama installed locally
- Llama 3.1:8B model (or compatible model)

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
pip install ollama-python

## File Structure
All these files must be in same folder
```
main.py          # main program file
ai_prompt.txt    # AI prompt template
finance.txt      # Finance department data
marketing.txt    # Marketing department data
IT.txt           # IT department data
```

## Possible Next Features Updates
These following are ideas for features that could be added in future versions.

### API UPDATE
- Google Docs and Drive API Connection
- OAuth 2.0
- Token persistence with pickle files
- Removed local text file dependency
