import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def answer_query(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an AI assistant helping with insurance policy documents.
Based on the following content, answer the user's question.
Content:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Could not answer due to error: {e}"
