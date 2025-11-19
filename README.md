# AI Algorithms – Option A  
## Model Compression for LLMs using Genetic Algorithms (Pruning Optimization)

This repository contains the implementation and report for the **AI Algorithms Final Project 2025**,  
**Option A: A Survey of AI Techniques for LLM Optimization**, following the guidelines of the  
**AI Algorithms Project Statement 2025**.

Our team chose to focus on:

> **Model Compression for Large Language Models (LLMs)** using **Pruning**,  
> with a practical implementation based on a **Genetic Algorithm (GA)** that searches for an optimal pruning rate on a small neural network.

---

## 🚀 Project Overview

Large Language Models (LLMs) often contain billions of parameters, resulting in:
- high inference latency  
- large GPU memory requirements  
- expensive deployment costs  

Model compression techniques (pruning, quantization) aim to reduce these costs while preserving accuracy.

In this project, we:

1. **Survey 6 key academic papers** on pruning and quantization techniques for LLMs.  
2. **Analyze and compare** their methods, limitations, and trade-offs.  
3. **Implement a Genetic Algorithm (GA)** to find an effective pruning rate on a small MLP proxy model.  
4. **Evaluate the GA** against fixed pruning baselines (0.1, 0.3, 0.5, 0.7).  
5. Provide a **critical analysis** and connect observations to state-of-the-art work.

This repository contains both the **Python code** and the **LaTeX report**.

---

## 🧪 Experimental Setup

### Baseline Model
We train a small **2-layer MLP** on MNIST (subset of 10,000 samples).  
Although not an LLM, it allows:
- fast execution  
- clear pruning analysis  
- conceptual mapping to pruning strategies used in LLMs  

### Optimization Technique
We use a **Genetic Algorithm**, consistent with the course content:
- Tournament selection  
- Arithmetic crossover  
- Gaussian mutation  
- Elitism  
- Fitness = validation accuracy  

The GA searches for a **single global pruning rate** in `[0, 0.9]`.

### Comparisons Included
We compare the GA against:
- The **baseline model** (no pruning)  
- Four **fixed pruning baselines**: 0.10, 0.30, 0.50, 0.70  
- The **best GA solution**

This provides strong material for the *Empirical Evaluation* required in the project.

---

## 📁 Repository Structure
.
├── main.py                  # Full implementation (MLP + Pruning + GA)
├── README.md                # You are here
├── requirements.txt         # Dependencies for the project
├── report/
│   ├── main.tex             # LaTeX report (Option A structure)
│   ├── references.bib       # Bibliography for the report
│   └── figures/             # Plots and figures 


