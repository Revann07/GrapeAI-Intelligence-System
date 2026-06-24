import google.generativeai as genai
from PIL import Image

# This securely loads the key from your hidden .env file
genai.configure(api_key="GEMINI_API_KEY")

model = genai.GenerativeModel("gemini-2.5-flash")

def analyze_grape_image(image_path):
    image = Image.open(image_path)

    prompt = """
    Analyze this grape image. You are Canopy AI, an expert agricultural diagnostic system. Do not mention Google, Gemini, or being a large language model.

    Return exactly in this text format:
    Disease Name: [Name of the disease or "Healthy Plant"]
    Confidence: [Percentage, e.g., 95%]
    Severity: [Low/Medium/High] ([Exact Percentage]%)
    Symptoms: [Brief description of visual symptoms]
    Treatment: [Recommended actions]
    Prevention: [How to prevent this in the future]
    """

    response = model.generate_content(
        [prompt, image]
    )

    return response.text