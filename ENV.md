# Environment Variables

Set these environment variables before running the analysis:

## Option 1: Create a .env file
Create a `.env` file in the project root with:
```
GEMINI_API=your_gemini_api_key_here
GPT_API=your_gpt_api_key_here
PROJECT_ID=your_google_cloud_project_id
LOCATION=us-central1
```

## Option 2: Set environment variables directly
```bash
export GEMINI_API=your_gemini_api_key_here
export GPT_API=your_gpt_api_key_here
export PROJECT_ID=your_google_cloud_project_id
export LOCATION=us-central1
```

## Required Variables
- `GEMINI_API`: Your Google Gemini API key for embeddings
- `GPT_API`: Your OpenAI API key (if using GPT models)
- `PROJECT_ID`: Your Google Cloud Project ID (for Gemini Vertex AI)
- `LOCATION`: Google Cloud region (optional, defaults to us-central1)
