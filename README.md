# leetcode-llm-tutor

This project includes: 
- Fine-tuning a language model specialized in coding interview questions 
- Generating new LeetCode style problems covering a variety of topics
- Providing step by step explanations and solutions to coding problems
- Deploying an interactive interface for users to learn and practice coding problems

## Running the code

This project uses the Hugging Face Hub to access models for fine-tuning. 
To run the code, you will need to have a hugging face account and generate an 
access token. 

1. Sign in or create an account at [Hugging Face](https://huggingface.co/).
2. Navigate to the access token page under the settings page
3. Create and save a new read token 
4. Set the token as environment variable named HUGGINGFACE_TOKEN 
- On macOS/Linux: ```bash export HUGGINGFACE_TOKEN="your-token-here"```
- On Windows: `setx HUGGINGFACE_TOKEN "your-token-here"`
 