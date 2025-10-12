import os
import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from google.auth import default
import google.auth.transport.requests
import openai
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Install it with: pip install python-dotenv")
    print("Or set GEMINI_API environment variable directly.")

# ==========================================================
# Configuration
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
VISUALS_DIR = DATA_DIR / "analysis" / "visuals"

# Sport name mapping (folder names to analysis names)
SPORT_MAPPING = {
    "soccer": "soccer",
    "basketball": "bb", 
    "volleyball": "vball"
}

# Model configurations
MODELS = ['gpt', 'gemini']
LEVELS = [0, 1, 2]



# Load API key from environment variable
API_KEY = os.getenv("GEMINI_API")
if not API_KEY:
    raise ValueError("GEMINI_API environment variable not set. Please set it in your environment or .env file.")
genai.configure(api_key=API_KEY)

# --- Utility: Save and Load Embeddings ---
def save_embeddings(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model_embeddings(model, sport, level):
    """Load embeddings for a specific model-sport-level combination"""
    embedding_path = RESULTS_DIR / model / "embeddings" / f"{sport}_level{level}_emb.pkl"
    if embedding_path.exists():
        return pd.read_pickle(embedding_path)
    else:
        print(f"Embedding file not found: {embedding_path}")
        return None

def load_all_embeddings():
    """Load all available embeddings and combine them"""
    all_embeddings = []
    sports = list(SPORT_MAPPING.values())
    
    for model in MODELS:
        for sport in sports:
            for level in LEVELS:
                df = load_model_embeddings(model, sport, level)
                if df is not None:
                    all_embeddings.append(df)
    
    if all_embeddings:
        return pd.concat(all_embeddings, ignore_index=True)
    else:
        return None

def find_csv_files():
    """Find all CSV files in the results directory following the naming convention"""
    csv_files = {}
    
    for model in MODELS:
        model_dir = RESULTS_DIR / model
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            continue
            
        csv_files[model] = {}
        
        # Look for CSV files with pattern: {sport}_{level}.csv
        for csv_file in model_dir.glob("*.csv"):
            filename = csv_file.stem  # Remove .csv extension
            
            # Parse filename to extract sport and level
            # Expected format: {sport}_level{level} (e.g., soccer_level0, bb_level1)
            if "_level" in filename:
                parts = filename.split("_level")
                if len(parts) == 2:
                    sport = parts[0]
                    try:
                        level = int(parts[1])
                        if sport not in csv_files[model]:
                            csv_files[model][sport] = {}
                        csv_files[model][sport][level] = csv_file
                    except ValueError:
                        print(f"Could not parse level from filename: {csv_file}")
            else:
                print(f"Unexpected CSV filename format: {csv_file}")
    
    return csv_files

def balance_dataset_by_model(df):
    """Balance the dataset so each model has equal samples across all levels"""
    # Find the minimum count per level across all models
    level_counts = df.groupby('level').size()
    min_per_level = level_counts.min()
    
    print(f"Minimum samples per level: {min_per_level}")
    print("Sample counts by level:")
    print(level_counts)
    
    # For each model, sample equal amounts from each level
    balanced_samples = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        model_samples = []
        
        for level in df['level'].unique():
            level_data = model_data[model_data['level'] == level]
            if len(level_data) >= min_per_level:
                # Randomly sample min_per_level samples from this level
                sampled = level_data.sample(n=min_per_level, random_state=42)
                model_samples.append(sampled)
            else:
                # If we don't have enough samples, take all available
                model_samples.append(level_data)
        
        if model_samples:
            balanced_samples.append(pd.concat(model_samples, ignore_index=True))
    
    return pd.concat(balanced_samples, ignore_index=True)

# --- Get embedding from OpenAI ---
def get_embedding(text):
    try:
        response = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")
        return response["embedding"]
    except Exception as e:
        print(f"Error embedding text: {text[:50]}... {e}")
        return None
    
# --- Templates ---
pose_templates = [
    "The player’s torso is angled toward the lower left side of the field.",
    "The defender's body is rotated toward the approaching opponent.",
    "The legs are aligned forward, indicating a sprint motion.",
    "Both players are physically oriented toward the center of the scene.",
    "The player is mid-stride, suggesting a rush toward the ball.",
    "One player is lunging while another is bracing, indicating contact.",
    "The attacker has one leg raised in a kicking posture.",
    "The defender is sliding aggressively to intercept.",
    "The player is balancing on one foot after a sudden motion.",
    "The posture implies defensive pressure or a physical block.",
    "The extended foot is approaching the ball’s likely location.",
    "The convergence of two players suggests the ball is between them.",
    "The player’s planted foot is directly adjacent to the ball area.",
    "The upper body leans into a contested zone near the ball.",
    "Torso and leg movement are directed toward a central target on the field.",
]

gaze_templates = [
    "The player's head is turned to the right.",
    "The eyes are directed diagonally downward.",
    "The gaze is angled toward the lower central part of the field.",
    "The head is tilted back slightly, indicating an upward glance.",
    "The player appears to be tracking an object in motion.",
    "Eyes are fixated on a moving target, likely the ball.",
    "The player is visually scanning the space ahead.",
    "The head tilt suggests reactive attention to unfolding action.",
    "Both players are looking at the same spot on the field.",
    "The gaze lines of multiple players intersect in the same region.",
    "The player’s attention is centered near the ball’s estimated position.",
    "The eyes are aimed directly at where the ball is likely to appear.",
    "Gaze and body orientation both converge toward the ball zone.",
]

# --- Load or compute template embeddings ---
def get_or_compute_template_embeddings():
    if os.path.exists("pose_embeddings.pkl") and os.path.exists("gaze_embeddings.pkl"):
        print("Loading template embeddings from disk...")
        pose_embeddings = load_embeddings("pose_embeddings.pkl")
        gaze_embeddings = load_embeddings("gaze_embeddings.pkl")
    else:
        print("Generating and saving template embeddings...")
        pose_embeddings = [get_embedding(t) for t in tqdm(pose_templates, desc="Pose templates")]
        gaze_embeddings = [get_embedding(t) for t in tqdm(gaze_templates, desc="Gaze templates")]
        save_embeddings(pose_embeddings, "pose_embeddings.pkl")
        save_embeddings(gaze_embeddings, "gaze_embeddings.pkl")
    return pose_embeddings, gaze_embeddings

# --- Main: Compare reasonings to pose/gaze ---
def compare_reasonings_to_templates(csv_path):
    pose_embeddings, gaze_embeddings = get_or_compute_template_embeddings()

    df = pd.read_csv(csv_path)
    if "reasoning" not in df.columns:
        raise ValueError("CSV must contain a 'reasoning' column")

    # Drop rows with NaN values in the reasoning column
    df = df.dropna(subset=['reasoning'])
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reasonings"):
        reasoning = row['reasoning']
        if pd.isna(reasoning) or not isinstance(reasoning, str):
            continue
            
        emb = get_embedding(reasoning)
        if emb is None:
            continue

        sim_pose = cosine_similarity([emb], pose_embeddings).flatten()
        sim_gaze = cosine_similarity([emb], gaze_embeddings).flatten()

        results.append({
            'image_id': row['image'],
            'reasoning': reasoning,
            'mean_pose': np.mean(sim_pose),
            'max_pose': np.max(sim_pose),
            'mean_gaze': np.mean(sim_gaze),
            'max_gaze': np.max(sim_gaze),
            'closer_to': 'pose' if np.mean(sim_pose) > np.mean(sim_gaze) else 'gaze',
            'embedding': emb  # Store this for later use
        })

    result_df = pd.DataFrame(results)
    result_df.to_pickle("level2_emb.pkl")
    print(f"Processed {len(results)} valid reasonings out of {len(df)} total rows")
    return result_df


def sem_space(file):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_pickle(file)

    sns.scatterplot(data=df, x='mean_pose', y='mean_gaze', hue='closer_to', palette={'pose': 'tomato', 'gaze': 'dodgerblue'})
    plt.plot([0, 1], [0, 1], 'k--', label="Equal Similarity")
    plt.title("Semantic Space of Reasonings")
    plt.xlabel("Mean Similarity to Pose Templates")
    plt.ylabel("Mean Similarity to Gaze Templates")
    plt.legend()
    plt.tight_layout()
    plt.show()

def bar_chart(file):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_pickle(file)

    df['closer_to'].value_counts().plot(kind='bar', color=['tomato', 'dodgerblue'])
    plt.title("Model Explanations: Pose vs. Gaze Alignment")
    plt.ylabel("Count")
    plt.xlabel("Closer To")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def sne_clustering(file):
    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    df = pd.read_pickle(file)

    embeddings = np.vstack(df['embedding'].to_numpy())
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_coords = tsne.fit_transform(embeddings)

    df['tsne_x'] = tsne_coords[:, 0]
    df['tsne_y'] = tsne_coords[:, 1]

    sns.scatterplot(data=df, x='tsne_x', y='tsne_y', hue='closer_to', palette={'pose': 'tomato', 'gaze': 'dodgerblue'})
    plt.title("t-SNE of Reasoning Embeddings")
    plt.xlabel("t-SNE X")
    plt.ylabel("t-SNE Y")
    plt.tight_layout()
    plt.show()


# --- Run on CSV ---
if __name__ == "__main__":
    # Check if combined analysis file already exists
    combined_file = DATA_DIR / "analysis" / "pose_gaze_analysis_by_level.pkl"
    
    if os.path.exists(combined_file):
        print(f"Found existing analysis file: {combined_file}")
        print("Loading existing data and regenerating graphs...")
        
        # Load existing data
        combined_df = pd.read_pickle(combined_file)
        print(f"Loaded {len(combined_df)} existing reasonings")
        
        # Skip to visualization part
        skip_to_visualization = True
    else:
        print("No existing analysis file found. Processing all data...")
        skip_to_visualization = False
    

        # Find all CSV files dynamically
        print("Discovering CSV files...")
        csv_files = find_csv_files()
        
        if not csv_files:
            print("No CSV files found in results directory!")
            print(f"Expected structure: {RESULTS_DIR}/{{model}}/{{sport}}_level{{level}}.csv")
            exit()
    
    all_results = []
    
    # Create embedding directories for each model
    for model in MODELS:
        embedding_dir = RESULTS_DIR / model / "embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each model-sport-level combination
    for model, sports in csv_files.items():
        for sport, levels in sports.items():
            for level, csv_path in levels.items():
                # Check if embedding file already exists
                embedding_path = RESULTS_DIR / model / "embeddings" / f"{sport}_level{level}_emb.pkl"
                
                if embedding_path.exists():
                    print(f"\nLoading existing embeddings for {model.upper()} {sport.upper()} Level {level}...")
                    # Load existing embeddings
                    df_result = pd.read_pickle(embedding_path)
                    all_results.append(df_result)
                    print(f"  Loaded {len(df_result)} existing embeddings for {model} {sport} level {level}")
                else:
                    print(f"\nProcessing {model.upper()} {sport.upper()} Level {level}...")
                    print(f"  Reading from: {csv_path}")
                    df_result = compare_reasonings_to_templates(str(csv_path))
                    if df_result is not None and not df_result.empty:
                        # Add model, sport, and level info
                        df_result['model'] = model
                        df_result['sport'] = sport
                        df_result['level'] = level
                        all_results.append(df_result)
                        
                        # Save individual embedding file
                        df_result.to_pickle(embedding_path)
                        print(f"  Saved embeddings to: {embedding_path}")
                        print(f"  Processed {len(df_result)} reasonings for {model} {sport} level {level}")
                    else:
                        print(f"  No valid data found in {csv_path}")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = DATA_DIR / "analysis" / "pose_gaze_analysis_by_level.pkl"
        combined_file.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_pickle(combined_file)
        print(f"\nCombined results saved: {len(combined_df)} total reasonings")
        print(f"Saved to: {combined_file}")
    else:
        print("No data processed!")
        exit()
    
    # Create visualizations (runs whether data was loaded or processed)
    print("\nCreating visualizations...")
    
    # Create visuals directory
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Visuals will be saved to: {VISUALS_DIR}")
    
    # Balance the dataset so each model has equal samples across levels
    print("Balancing dataset for equal samples per model across levels...")
    balanced_df = balance_dataset_by_model(combined_df)
    print(f"Original dataset: {len(combined_df)} samples")
    print(f"Balanced dataset: {len(balanced_df)} samples")
    
    # 1. Pose vs Gaze counts by model and level
    plt.figure(figsize=(12, 6))
    pose_gaze_counts = balanced_df.groupby(['model', 'level', 'closer_to']).size().unstack(fill_value=0)
    pose_gaze_counts.plot(kind='bar', color=['tomato', 'dodgerblue'])
    plt.title('Pose vs Gaze Alignment by Model and Level', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    plt.xlabel('Model - Level')
    plt.xticks(rotation=45)
    plt.legend(title='Closer To')
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'pose_gaze_by_model_level.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Pose vs Gaze counts by sport
    plt.figure(figsize=(12, 6))
    sport_counts = balanced_df.groupby(['sport', 'closer_to']).size().unstack(fill_value=0)
    sport_counts.plot(kind='bar', color=['tomato', 'dodgerblue'])
    plt.title('Pose vs Gaze Alignment by Sport', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    plt.xlabel('Sport')
    plt.xticks(rotation=0)
    plt.legend(title='Closer To', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'pose_gaze_by_sport.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Mean pose-gaze difference by level
    plt.figure(figsize=(10, 6))
    balanced_df['pose_gaze_diff'] = balanced_df['mean_pose'] - balanced_df['mean_gaze']
    mean_diff = balanced_df.groupby(['model', 'level'])['pose_gaze_diff'].mean().unstack()
    mean_diff.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Mean Pose-Gaze Difference by Level', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Mean Pose-Gaze Difference')
    plt.legend(title='Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # Ensure y-axis shows negative values if they exist
    y_min, y_max = plt.ylim()
    if y_min < 0:
        plt.ylim(bottom=y_min * 1.1)  # Add some padding below
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'pose_gaze_diff_by_level.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Mean pose-gaze difference by sport and level
    plt.figure(figsize=(12, 6))
    sport_level_diff = balanced_df.groupby(['sport', 'level'])['pose_gaze_diff'].mean().unstack()
    sport_level_diff.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Mean Pose-Gaze Difference by Sport and Level', fontsize=14, fontweight='bold')
    plt.xlabel('Sport')
    plt.ylabel('Mean Pose-Gaze Difference')
    plt.legend(title='Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # Ensure y-axis shows negative values if they exist
    y_min, y_max = plt.ylim()
    if y_min < 0:
        plt.ylim(bottom=y_min * 1.1)  # Add some padding below
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'pose_gaze_diff_by_sport_level.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. t-SNE plots
    print("\nCreating t-SNE visualizations...")
    
    # Use the balanced data for t-SNE
    df_tsne = balanced_df.copy()
    
    # Prepare embeddings
    embeddings = np.vstack(df_tsne['embedding'].to_numpy())
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_coords = tsne.fit_transform(embeddings)
    
    df_tsne['tsne_x'] = tsne_coords[:, 0]
    df_tsne['tsne_y'] = tsne_coords[:, 1]
    
    # t-SNE colored by pose/gaze alignment
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='tsne_x', y='tsne_y', hue='closer_to', 
                   palette={'pose': 'tomato', 'gaze': 'dodgerblue'}, s=50, alpha=0.7)
    plt.title('t-SNE: Colored by Pose/Gaze Alignment', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE X')
    plt.ylabel('t-SNE Y')
    plt.legend(title='Closer To')
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'tsne_pose_gaze.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # t-SNE colored by level
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='tsne_x', y='tsne_y', hue='level', 
                   palette='viridis', s=50, alpha=0.7)
    plt.title('t-SNE: Colored by Level', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE X')
    plt.ylabel('t-SNE Y')
    plt.legend(title='Level')
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'tsne_by_level.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # t-SNE colored by sport
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='tsne_x', y='tsne_y', hue='sport', 
                   palette='Set1', s=50, alpha=0.7)
    plt.title('t-SNE: Colored by Sport', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE X')
    plt.ylabel('t-SNE Y')
    plt.legend(title='Sport')
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'tsne_by_sport.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # t-SNE colored by model
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='tsne_x', y='tsne_y', hue='model', 
                   palette='tab10', s=50, alpha=0.7)
    plt.title('t-SNE: Colored by Model', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE X')
    plt.ylabel('t-SNE Y')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'tsne_by_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n=== SUMMARY ===")
    print("1. Pose vs Gaze counts by model and level:")
    print(combined_df.groupby(['model', 'level', 'closer_to']).size().unstack(fill_value=0))
    
    print("\n2. Pose vs Gaze counts by sport:")
    print(combined_df.groupby(['sport', 'closer_to']).size().unstack(fill_value=0))
    
    print("\n3. Mean pose-gaze difference by model and level (positive = more pose-like):")
    print(combined_df.groupby(['model', 'level'])['pose_gaze_diff'].agg(['mean', 'std']).round(4))
    
    print("\n4. Mean pose-gaze difference by sport and level:")
    print(combined_df.groupby(['sport', 'level'])['pose_gaze_diff'].agg(['mean', 'std']).round(4))
    
    print("\n5. Overall statistics:")
    print(f"Total reasonings analyzed: {len(combined_df)}")
    print(f"Models: {', '.join(combined_df['model'].unique())}")
    print(f"Sports: {', '.join(combined_df['sport'].unique())}")
    print(f"Levels: {', '.join(map(str, sorted(combined_df['level'].unique())))}")
    
    print("\n6. Files saved:")
    print("Individual embedding files saved to:")
    for model in MODELS:
        embedding_dir = RESULTS_DIR / model / "embeddings"
        print(f"  - {embedding_dir}")
    print("Format: {sport}_level{level}_emb.pkl")
    print(f"Combined analysis saved to: {DATA_DIR / 'analysis' / 'pose_gaze_analysis_by_level.pkl'}")
    print(f"Visualizations saved to: {VISUALS_DIR}")
    print("Generated visualizations:")
    print("  - pose_gaze_by_model_level.png")
    print("  - pose_gaze_by_sport.png")
    print("  - pose_gaze_diff_by_level.png")
    print("  - pose_gaze_diff_by_sport_level.png")
    print("  - tsne_pose_gaze.png")
    print("  - tsne_by_level.png")
    print("  - tsne_by_sport.png")
    print("  - tsne_by_model.png")
