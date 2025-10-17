# Credits

## Official Documents
This project includes or adapts the following documents.

### [Google API Integration](https://developers.google.com/workspace/drive/api/reference/rest/v3)
- **Description:** The Google Drive API allows clients to access resources from Google Drive.

### [Google Drive Export API](https://developers.google.com/drive/api/v3/manage-downloads#exporting_google_documents)
- **Description:** Implemented to convert Google Docs into plain text format for AI processing while maintaining document structure.

### [Google Drive Files List API](https://developers.google.com/drive/api/v3/reference/files/list)
- **Description:** Utilized to search and discover department folders and weekly reports within the Company Reports directory structure.

### [Multi-format File Support](https://developers.google.com/drive/api/v3/ref-export-formats)
- **Description:** Extended to process various file types (Google Docs, DOCX, PDF, Text) ensuring comprehensive department report coverage.

### [Google Credentials Management](https://google-auth.readthedocs.io/en/latest/reference/google.oauth2.credentials.html)
- **Description:** Designed to handle token refresh and storage for seamless re-authentication across multiple department data sessions.

### [Google Drive Export API](https://developers.google.com/drive/api/v3/manage-downloads#exporting_google_documents)
- **Description:** Implemented to convert Google Docs into plain text format for AI processing while maintaining document structure.


## Open Source Projects

This project includes or adapts the following open source projects. I am grateful for the authors and their contributions to the open source community.

### [SWE Agent](https://github.com/ollama/ollama-python)
- **License:** MIT License
- **Description:** Adapted for use in the CWAI project to generate AI responses using LLaMA.

### [Ollama Chat API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion)
- **License:** MIT License
- **Description:** Adapted for use in the DepartmentAI project to generate AI-powered department insights using the LLaMA 3.1 model

### [Google Drive API Authentication](https://github.com/googleworkspace/python-samples/blob/main/docs/quickstart/quickstart.py)
- **License:** Apache-2.0 license 
- **Description:** Adapted from Google's Python Quickstart to handle department-specific folder access with persistent token management.

### [File Export & Media Download](https://github.com/googleworkspace/python-samples/blob/main/drive/snippets/drive-v3/file_snippet/download_file.py)
- **License:** Apache-2.0 license 
- **Description:** Enhanced from simple downloads to multi-format department document processing with in-memory handling.