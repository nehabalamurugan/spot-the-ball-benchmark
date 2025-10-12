import os
import csv
import re
import base64
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import google.auth.transport.requests
from google.auth import default
import openai

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Install it with: pip install python-dotenv")
    print("Or set GEMINI_API and PROJECT_ID environment variables directly.")

# ==========================================================
# Configuration
# ==========================================================

# Load configuration from environment variables
PROJECT_ID = os.getenv("PROJECT_ID")
if not PROJECT_ID:
    raise ValueError("PROJECT_ID environment variable not set. Please set it in your environment or .env file.")

LOCATION = os.getenv("LOCATION", "us-central1")  # Default to us-central1 if not set
NUM_PREDICTIONS = {
    "level0": 50,
    "level1": 50,
    "level2": 20,
}

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "images"
RESULTS_DIR = BASE_DIR / "results" / "gemini"

SPORT_FOLDERS = {
    "soccer": DATA_DIR / "soccer",
    "basketball": DATA_DIR / "basketball",
    "volleyball": DATA_DIR / "volleyball",
}

OUTPUT_PATHS = {
    level: {s: RESULTS_DIR / f"{s}_{level}.csv" for s in SPORT_FOLDERS}
    for level in ["level0", "level1", "level2"]
}


# ==========================================================
# Authentication
# ==========================================================

def authenticate_gemini():
    """Authenticate to Vertex AI Gemini endpoint."""
    try:
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
        client = openai.OpenAI(
            base_url=f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi",
            api_key=credentials.token,
        )
        print("Authentication successful")
        return client
    except Exception as e:
        raise RuntimeError(f"Authentication failed: {e}")


# ==========================================================
# Helper Functions
# ==========================================================

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_response(response: str):
    reasoning_match = re.search(r"Reasoning:\s*(.*?)\s*Cell:", response, re.DOTALL | re.IGNORECASE)
    cell_match = re.search(r"Cell:\s*([A-Fa-f][0-9]+)", response)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    cell = cell_match.group(1).upper() if cell_match else ""
    return reasoning, cell


# ==========================================================
# Prompt Builders
# ==========================================================

def build_prompt(level: str, sport: str, context: str = "") -> str:
    base = f"The ball has been removed from this {sport} image. Your task is to infer the most likely location of the ball."
    if level == "level1":
        context = "The location of the players, where they are looking, and their positions can help you infer the ball’s location."
    return f"{base}\n{context}\n\nRespond in this format:\nReasoning: <Explain where the ball is likely located and why.>\nCell: <Grid cell label like F4.>"


# ==========================================================
# Model Interaction
# ==========================================================

def get_prediction(client, image_path, prompt):
    encoded = encode_image_base64(image_path)
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ],
            }],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error on {image_path}: {e}")
        return "ERROR"


def ask_question(client, image_path, question):
    return get_prediction(client, image_path, question)


# ==========================================================
# Image Processing
# ==========================================================

def process_image(client, image_path: Path, sport: str, level: str, n: int):
    rows = []

    if level in ["level0", "level1"]:
        prompt = build_prompt(level, sport)
        for i in range(n):
            resp = get_prediction(client, image_path, prompt)
            reasoning, cell = parse_response(resp)
            rows.append([image_path.name, i + 1, reasoning, cell])

    elif level == "level2":
        questions = [
            "Where are the players located?",
            "Where are the players looking?",
            "How are the players positioned?",
        ]
        for i in range(n):
            answers = [ask_question(client, image_path, q) for q in questions]
            context = "\n".join([f"{q} {a}" for q, a in zip(questions, answers)])
            final_prompt = build_prompt(level, sport, context)
            resp = get_prediction(client, image_path, final_prompt)
            reasoning, cell = parse_response(resp)
            rows.append([image_path.name, i + 1, *answers, reasoning, cell, resp])
    return rows


# ==========================================================
# Sport-Level Processing
# ==========================================================

def process_sport(client, sport: str, level: str, image_dir: Path, output_file: Path):
    os.makedirs(output_file.parent, exist_ok=True)
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    if not image_files:
        print(f"No images found for {sport}")
        return

    print(f"\nProcessing {sport} ({len(image_files)} images) - {level}")
    num_predictions = NUM_PREDICTIONS[level]

    header = (
        ["image", "iteration", "reasoning", "cell"]
        if level != "level2"
        else ["image", "iteration", "q1_answer", "q2_answer", "q3_answer", "reasoning", "cell", "final_response"]
    )

    processed = set()
    if output_file.exists():
        try:
            processed_df = pd.read_csv(output_file)
            processed = set(processed_df["image"].unique())
        except Exception:
            pass

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not output_file.exists():
            writer.writerow(header)

        for img in tqdm(image_files, desc=f"{sport} {level}", unit="image"):
            if img.name in processed:
                continue
            try:
                rows = process_image(client, img, sport, level, num_predictions)
                writer.writerows(rows)
            except Exception as e:
                print(f"Error processing {img}: {e}")
                continue

    print(f"Completed {sport}-{level}. Results saved to {output_file}")


# ==========================================================
# Main
# ==========================================================

def main():
    client = authenticate_gemini()
    levels = ["level0", "level1", "level2"]

    for sport, folder in SPORT_FOLDERS.items():
        if not folder.exists():
            print(f"Folder not found for {sport}: {folder}")
            continue
        for level in levels:
            process_sport(client, sport, level, folder, OUTPUT_PATHS[level][sport])

    print("All processing completed successfully.")


if __name__ == "__main__":
    main()
