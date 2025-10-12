import os
import csv
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from PIL import Image


# ==========================================================
# Configuration
# ==========================================================

NUM_PREDICTIONS = {
    "level0": 50,
    "level1": 50,
    "level2": 20,
}

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "images"
RESULTS_DIR = BASE_DIR / "results" / "llama"

SPORT_FOLDERS = {
    "soccer": DATA_DIR / "soccer",
    "basketball": DATA_DIR / "basketball",
    "volleyball": DATA_DIR / "volleyball",
}

OUTPUT_PATHS = {
    level: {s: RESULTS_DIR / f"{s}_{level}.csv" for s in SPORT_FOLDERS}
    for level in ["level0", "level1", "level2"]
}

MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision"


# ==========================================================
# Model Initialization
# ==========================================================

def load_model():
    """Load llama/LLaMA-VL model."""
    try:
        print(f"Loading model: {MODEL_NAME}")
        pipe = pipeline("image-text-to-text", model=MODEL_NAME, device_map="auto")
        print("Model loaded successfully")
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


# ==========================================================
# Helper Functions
# ==========================================================

def parse_response(response: str):
    """Extract reasoning and predicted cell."""
    if not isinstance(response, str):
        return "", ""
    reasoning_match = re.search(r"Reasoning:\s*(.*?)\s*Cell:", response, re.DOTALL | re.IGNORECASE)
    cell_match = re.search(r"Cell:\s*([A-Fa-f][0-9]+)", response)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    cell = cell_match.group(1).upper() if cell_match else ""
    return reasoning, cell


def safe_clean(text):
    """Remove newlines and clean text."""
    if not isinstance(text, str):
        return ""
    return text.replace("\n", " ").replace("\r", " ").strip()


def extract_llama_response(response):
    """Extract the generated assistant text from llama output format."""
    if (
        isinstance(response, list)
        and len(response) > 0
        and isinstance(response[0], dict)
        and "generated_text" in response[0]
    ):
        for msg in response[0]["generated_text"]:
            if msg.get("role") == "assistant":
                return msg.get("content", "ERROR")
    return "ERROR"


# ==========================================================
# Prompt Builders
# ==========================================================

def build_prompt(level: str, sport: str, context: str = "") -> str:
    """Build the text prompt for each reasoning level."""
    base_prompt = f"The ball has been removed from this {sport} image. Your task is to infer the most likely location of the ball."

    if level == "level1":
        context = "The location of the players, where they are looking, and their positions can help you infer the ball’s location."

    return f"{base_prompt}\n{context}\n\nRespond in this format:\nReasoning: <Explain where the ball is likely located and why.>\nCell: <Grid cell label like F4.>"


# ==========================================================
# Model Interaction
# ==========================================================

def query_model(pipe, image_path: str, text_prompt: str, max_tokens: int = 300):
    """Query the model with an image and prompt."""
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": text_prompt},
            ]}
        ]
        response = pipe(text=messages, max_new_tokens=max_tokens)
        return extract_llama_response(response)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "ERROR"


def ask_question(pipe, image_path: str, question: str):
    """Ask an intermediate question (for Level 2)."""
    return query_model(pipe, image_path, question, max_tokens=200)


# ==========================================================
# Image Processing
# ==========================================================

def process_image(pipe, image_path: Path, sport: str, level: str, n: int):
    """Generate predictions for a single image."""
    rows = []

    if level in ["level0", "level1"]:
        prompt = build_prompt(level, sport)
        for i in range(n):
            resp = query_model(pipe, image_path, prompt)
            reasoning, cell = parse_response(resp)
            rows.append([image_path.name, i + 1, safe_clean(reasoning), safe_clean(cell)])

    elif level == "level2":
        questions = [
            "Where are the players located?",
            "Where are the players looking?",
            "How are the players positioned?"
        ]
        for i in range(n):
            answers = [ask_question(pipe, image_path, q) for q in questions]
            context = "\n".join([f"{q} {a}" for q, a in zip(questions, answers)])
            final_prompt = build_prompt(level, sport, context)
            resp = query_model(pipe, image_path, final_prompt)
            reasoning, cell = parse_response(resp)
            rows.append([
                image_path.name, i + 1,
                *map(safe_clean, answers),
                safe_clean(reasoning), safe_clean(cell),
                safe_clean(resp)
            ])
    return rows


# ==========================================================
# Sport-Level Processing
# ==========================================================

def process_sport(pipe, sport: str, level: str, image_dir: Path, output_file: Path):
    """Run model predictions for one sport and level."""
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
                rows = process_image(pipe, img, sport, level, num_predictions)
                writer.writerows(rows)
            except Exception as e:
                print(f"Error processing {img}: {e}")
                continue

    print(f"Completed {sport}-{level}. Results saved to {output_file}")


# ==========================================================
# Main
# ==========================================================

def main():
    """Run full pipeline across sports and reasoning levels."""
    pipe = load_model()
    levels = ["level0","level1","level2"] 

    for sport, folder in SPORT_FOLDERS.items():
        if not folder.exists():
            print(f"Folder not found for {sport}: {folder}")
            continue
        for level in levels:
            process_sport(pipe, sport, level, folder, OUTPUT_PATHS[level][sport])

    print("All processing completed successfully.")


if __name__ == "__main__":
    main()
