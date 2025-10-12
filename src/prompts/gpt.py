import os
import re
import base64
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Install it with: pip install python-dotenv")
    print("Or set GPT_API environment variable directly.")

# ==========================================================
# Configuration
# ==========================================================

# Load API key from environment variable
API_KEY = os.getenv("GPT_API")
if not API_KEY:
    raise ValueError("GPT_API environment variable not set. Please set it in your environment or .env file.")

os.environ["OPENAI_API_KEY"] = API_KEY

NUM_PREDICTIONS = {
    "level0": 50,
    "level1": 50,
    "level2": 20,
}

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "images"
RESULTS_DIR = BASE_DIR / "results" / "gpt"

SPORT_FOLDERS = {
    "soccer": DATA_DIR / "soccer",
    "basketball": DATA_DIR / "basketball",
    "volleyball": DATA_DIR / "volleyball",
}

OUTPUT_PATHS = {
    level: {s: RESULTS_DIR / f"{s}_{level}.csv" for s in SPORT_FOLDERS}
    for level in ["level0", "level1", "level2"]
}

MODEL_NAME = "gpt-4.1-mini"


# ==========================================================
# Helper Functions
# ==========================================================

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_response(response: str):
    """Extract reasoning and predicted cell."""
    reasoning_match = re.search(r"Reasoning:\s*(.*?)\s*Cell:", response, re.DOTALL | re.IGNORECASE)
    cell_match = re.search(r"Cell:\s*([A-Fa-f][0-9]+)", response)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    cell = cell_match.group(1).upper() if cell_match else ""
    return reasoning, cell


def safe_clean(text):
    if not isinstance(text, str):
        return ""
    return text.replace("\n", " ").replace("\r", " ").strip()


# ==========================================================
# Prompt Builders
# ==========================================================

def build_prompt(level: str, sport: str, context: str = "") -> str:
    """Build prompt text for each reasoning level."""
    if level == "level0":
        return (
            f"The ball has been removed from this {sport} image. "
            "Your task is to infer the most likely location of the ball.\n\n"
            "Respond in the following format:\n"
            "Reasoning: <Explain where the ball is likely located and why.>\n"
            "Cell: <Grid cell label like F4.>"
        )

    elif level == "level1":
        return (
            f"The ball has been removed from this {sport} image. "
            "Your task is to infer the most likely location of the ball.\n"
            "The location of the players, where they are looking, and their positions can help you infer the ball’s location.\n\n"
            "Respond in the following format:\n"
            "Reasoning: <Explain where the ball is likely located and why.>\n"
            "Cell: <Grid cell label like F4.>"
        )

    elif level == "level2":
        return (
            f"The ball has been removed from this {sport} image. Here are some observations:\n"
            f"{context}\n\n"
            "Using the above information, infer the ball’s location.\n\n"
            "Respond in the following format:\n"
            "Reasoning: <Explain where the ball is likely located and why.>\n"
            "Cell: <Grid cell label like F4.>"
        )


# ==========================================================
# Model Interaction
# ==========================================================

def query_gpt(image_path: str, text_prompt: str) -> str:
    """Send image + text to GPT model."""
    image_base64 = encode_image(image_path)
    model = ChatOpenAI(temperature=0.6, model=MODEL_NAME, max_tokens=1024)
    try:
        response = model.invoke([
            HumanMessage(
                content=[
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ]
            )
        ])
        return response.content.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "ERROR"


# ==========================================================
# Level-Specific Logic
# ==========================================================

def process_image(pipe, image_path: Path, sport: str, level: str, n: int):
    """Run inference for one image."""
    rows = []

    if level in ["level0", "level1"]:
        prompt = build_prompt(level, sport)
        for i in range(n):
            resp = query_gpt(str(image_path), prompt)
            reasoning, cell = parse_response(resp)
            rows.append([image_path.name, i + 1, safe_clean(reasoning), safe_clean(cell)])

    elif level == "level2":
        questions = [
            "Where are the players located?",
            "Where are the players looking?",
            "How are the players positioned?"
        ]
        for i in range(n):
            answers = [query_gpt(str(image_path), q) for q in questions]
            context = "\n".join([f"{q} {a}" for q, a in zip(questions, answers)])
            prompt = build_prompt(level, sport, context)
            resp = query_gpt(str(image_path), prompt)
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

def process_sport(sport: str, level: str, image_dir: Path, output_file: Path):
    """Run predictions for one sport and level."""
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
        writer = pd.DataFrame(columns=header)
        if not output_file.exists():
            writer.to_csv(output_file, index=False)

    for img in tqdm(image_files, desc=f"{sport} {level}", unit="image"):
        if img.name in processed:
            continue
        try:
            rows = process_image(None, img, sport, level, num_predictions)
            df = pd.DataFrame(rows, columns=header)
            df.to_csv(output_file, mode="a", header=False, index=False)
        except Exception as e:
            print(f"Error processing {img}: {e}")
            continue

    print(f"Completed {sport}-{level}. Results saved to {output_file}")


# ==========================================================
# Main
# ==========================================================

def main():
    """Run the full pipeline across all sports and levels."""
    levels = ["level0", "level1", "level2"]

    for sport, folder in SPORT_FOLDERS.items():
        if not folder.exists():
            print(f"Folder not found for {sport}: {folder}")
            continue
        for level in levels:
            process_sport(sport, level, folder, OUTPUT_PATHS[level][sport])

    print("All processing completed successfully.")


if __name__ == "__main__":
    main()
