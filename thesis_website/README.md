# Thesis Demo Website - Credit Card Fraud Detection

This directory contains a premium, interactive dashboard website designed for presenting your final thesis. It visually illustrates the Hybrid Attention-Autoencoder Framework, displays the empirical results, and contains an interactive sandbox transaction simulator.

## Features

1. **Thesis Abstract & Hero Header**: Highlighting the core contributions of your research.
2. **Interactive Block Diagram**: Shows the modular structure of the supervised/unsupervised dual-branch architecture. Clicking on any block opens a mathematical and design description.
3. **Interactive Results Viewer**: Tables listing the cross-seed average metrics and novelty detection rates, alongside a tabbed gallery for your 12 publication figures (supporting high-resolution image zoom Lightbox).
4. **Live Transaction Simulator**: Sliders to mock card charges and watch the Hybrid scoring engine run in real-time, plotting an SVG circular gauge, verdict alerts, and custom SHAP feature attributions!

---

## How to Run Locally

Because the webpage loads figures and files from local folders, web browsers may restrict certain functions if opened directly as a file (`file://`). It is **highly recommended** to run a simple local web server:

### Option 1: Run with Python (Standard & Recommended)
Open your terminal and run:

```bash
# Navigate to the website directory
cd thesis_website

# Start a simple HTTP server
python3 -m http.server 8080
```

Once running, open your web browser and navigate to:
👉 **[http://localhost:8080](http://localhost:8080)**

### Option 2: Run with VS Code Live Server
If you use VS Code, you can install the **"Live Server"** extension, open this folder in VS Code, and click the **"Go Live"** button in the status bar.
