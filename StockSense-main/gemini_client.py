# gemini_client.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- Load environment variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path, override=True)

# --- Configure Gemini ---
_API_KEY = os.getenv("GOOGLE_API_KEY")
if not _API_KEY:
    print("⚠️ GOOGLE_API_KEY missing! Please add it to your .env file.")
else:
    genai.configure(api_key=_API_KEY)
    print("✅ Google API key loaded successfully.")

# ✅ Correct, working model name for v1beta API and google-generativeai 0.8.5
MODEL_NAME = "gemini-flash-latest"

def get_model():
    """Safely get a Gemini model compatible with your installed SDK."""
    try:
        return genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"⚠️ Fallback model used: {e}")
        return genai.GenerativeModel("gemini-pro-latest")

def is_ready() -> bool:
    """Return True if GOOGLE_API_KEY looks valid."""
    return bool(_API_KEY and len(_API_KEY) > 10)

def ask_gemini(messages: list[dict], system_prompt: str = "") -> str:
    """
    messages: list of {"role": "user"|"assistant", "content": "text"}
    Returns assistant text.
    """
    if not is_ready():
        return "⚠️ Gemini API key is missing. Please add GOOGLE_API_KEY to your .env file."

    try:
        # Combine system + messages into a single prompt string
        lines = []
        if system_prompt:
            lines.append(f"[System]\n{system_prompt}\n")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"[{role.capitalize()}]\n{content}\n")
        prompt = "\n".join(lines).strip()

        # ✅ Generate a response from Gemini
        model = get_model()
        resp = model.generate_content(prompt)

        # Return text response cleanly
        if hasattr(resp, "text"):
            return resp.text.strip()
        return "⚠️ No valid response received from Gemini."

    except Exception as e:
        return f"⚠️ Gemini error: {e}"
