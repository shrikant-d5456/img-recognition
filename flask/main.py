from flask import Flask, request, jsonify
import os
import base64
import io
from PIL import Image
from dotenv import load_dotenv

from langchain.schema.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Ensure OpenAI API Key is set
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise EnvironmentError("❌ OPENAI_API_KEY not set in .env file")

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=300,
    api_key=openai_key
)

@app.route("/identify", methods=["POST"])
def identify_plant():
    try:
        print("🟢 Request received")

        # Check if image is present
        if 'image' not in request.files:
            return jsonify({"error": "Missing image"}), 400

        file = request.files['image']
        print("🟡 Image received:", file.filename)

        # Convert image to base64
        image = Image.open(io.BytesIO(file.read()))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Create message
        message = HumanMessage(content=[
            {"type": "text", "text": (
                "Identify the herb in the image and respond in this JSON format:\n"
                "{\n"
                "  \"common_name\": \"\",\n"
                "  \"scientific_name\": \"\",\n"
                "  \"description\": \"\",\n"
                "  \"uses\": [\"use1\", \"use2\"],\n"
                "  \"cultivation\": \"\",\n"
                "  \"cautions\": [\"caution1\", \"caution2\"]\n"
                "}\n"
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ])

        print("🔵 Sending message to OpenAI...")
        result = llm.invoke([message])
        print("✅ Response received")

        return jsonify({"status": "success", "result": result.content})

    except Exception as e:
        import traceback
        print("❌ Full error traceback:")
        traceback.print_exc()
        return jsonify({"status": "failure", "error": str(e)}), 500


# ✅ Move this outside the function
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets this env variable
    app.run(host='0.0.0.0', port=port)
