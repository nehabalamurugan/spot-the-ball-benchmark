# Spot the Ball: A Benchmark for Visual Social Inference in Sports Scenes

This repository contains the code and data for our paper  
**“Spot the Ball: A Benchmark for Visual Social Inference in Sports Scenes” (AAAI 2025 submission)**.  

Humans can infer hidden elements of a scene—like a missing ball—from subtle behavioral cues such as gaze, pose, and body orientation.  
This benchmark evaluates whether **vision–language models (VLMs)** exhibit similar social reasoning skills.

---

## 🧩 Overview

The **Spot the Ball** benchmark tests models’ ability to locate an occluded sports ball across **soccer, basketball, and volleyball** images.

### Key features
- **150 curated evaluation images** with human baseline annotations  
- **3,000+ procedurally generated scenes** via an automated pipeline  
- **Four multimodal models** evaluated (Gemini, GPT-4, LLaMA-3.2, Qwen-2.5)  
- **Three prompting levels** (direct, social-cue, and chain-of-thought)  
- Comprehensive **metrics**: accuracy, Euclidean error, Wasserstein distance, player proximity, and entropy

---

## 📂 Repository Structure



## Setup

Clone the repository:

git clone https://github.com/nehabalamurugan/spot-the-ball-benchmark.git
cd spot-the-ball-benchmark


## Install dependencies:

```pip install -r requirements.txt```


## Run evaluation:

```python src/evaluate_models.py --model gpt4v --dataset data/eval_set```


## Visualize results:

```python src/analysis/plot_results.py```

## Results Summary

Humans outperform all tested models by a large margin (19–34% vs. ≤17% accuracy).
Models show strong center and near-player biases and fail to fully use gaze cues.
Richer prompting improves textual reasoning but not predictive accuracy, revealing a persistent gap in visual social inference.

Citation

If you use this benchmark or code, please cite:

@article{balamurugan2025spotball,
  title={Spot the Ball: A Benchmark for Visual Social Inference in Sports Scenes},
  author={},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}