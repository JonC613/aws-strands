# Python Workspace

This is a minimal Python workspace for running strands agents.

## Setup

1. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies (if not already installed):
   ```powershell
   pip install -r requirements.txt
   ```

## Running with LM Studio

To use LM Studio's OpenAI-compatible endpoint instead of AWS Bedrock:

1. Start LM Studio and load a model
2. Enable the local server (default: `http://localhost:1234`)
3. Set environment variables:
   ```powershell
   $env:USE_LM_STUDIO="true"
   $env:LM_STUDIO_URL="http://localhost:1234/v1"
   $env:LM_STUDIO_MODEL="local-model"  # Use the actual model name from LM Studio
   ```
4. Run the agent:
   ```powershell
   python src/main.py
   ```

Alternatively, create a `.env` file (copy from `.env.example`) and use `python-dotenv`:
```powershell
pip install python-dotenv
```

## Running with AWS Bedrock (default)

Ensure AWS credentials are configured:
```powershell
$env:AWS_REGION="us-east-1"
$env:AWS_ACCESS_KEY_ID="your_access_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key"
```

Then run:
```powershell
python src/main.py
```

## Testing

Run tests with pytest:
```powershell
python -m pytest -q
```
