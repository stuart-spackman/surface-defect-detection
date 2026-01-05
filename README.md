Surface Defect Detection Using Classical Machine Learning

Author: Stuart Spackman
Advisor: Dr. Jeffrey Butler (Grand Canyon University)

This project explores whether lightweight classical machine learning models, combined with careful feature engineering, can approach the performance of more resource-intensive deep learning approaches for industrial surface defect classification.

The system classifies defects such as scratches, inclusions, patches, and rolled-in scale using logistic regression with engineered features, and includes an interactive GUI for visualization and inference.

🎯 Motivation & Research Question

Manufacturing industries (e.g., semiconductor, aerospace, steel) face significant costs when surface defects are missed or misclassified. While deep learning dominates the literature, it often requires:

Large labeled datasets

High computational resources

Complex deployment pipelines

Research Question:

Can classical machine learning algorithms, paired with strategic feature engineering, achieve competitive performance while remaining lightweight and deployable?

🧠 Approach Overview
Feature Engineering

Histogram of Oriented Gradients (HOG) — captures shape and texture

Edge Detection — highlights discontinuities and defect boundaries

Principal Component Analysis (PCA) — reduces dimensionality and noise

Model

Logistic Regression (scikit-learn)

Evaluated using precision, recall, and F1 score

Tools & Technologies

Python 3.12

scikit-learn

OpenCV

Streamlit (GUI)

Jupyter Notebooks (experimentation & analysis)

📊 Results

The best-performing pipeline was:

HOG + PCA + Logistic Regression

Macro F1 Score: 0.75

Strong performance across multiple defect classes

Demonstrates that classical ML remains viable when paired with good feature design

Confusion Matrix (Best Model: HOG + PCA)

(Add this image to /assets/confusion_matrix_hog_pca.png and keep the line below)

🖥️ Application & Demonstration

The project includes a GUI that allows users to:

Upload surface images

Visualize defect-relevant features

Run trained classifiers

View predicted labels and confidence

🎥 Demo Videos

Project Presentation

Part 1 – Poster Discussion
https://www.loom.com/share/808fda2a03304674aff723203c3e7e6f

Part 2 – Additional Project Motivations
https://www.loom.com/share/c9f5aa1cc00d4c959bfac4e4f8c19047

Technical Walkthrough

Part 3 – Jupyter Notebooks & Backend
https://www.loom.com/share/23e8001f8ea640659f8dfdfb1dd9c064

Part 4 – Front End & Live Demonstration
https://www.loom.com/share/b054471052c0425ba77daed5823da0a7

🚀 Quick Start
# Clone the repository
git clone https://github.com/stuart-spackman/surface-defect-detection.git
cd surface-defect-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app/app.py


Trained model artifacts are not stored directly in the repository.
See the README or Release notes for reproduction or download instructions.

🔁 Reproducibility

Code used for published results is frozen under GitHub Release v1.0-paper

Dependencies are fully specified in requirements.txt

Feature extraction and model training scripts are included

Results can be regenerated from the original dataset

�� Paper-aligned code:
https://github.com/stuart-spackman/surface-defect-detection/releases/tag/v1.0-paper

📂 Dataset

NEU Surface Defect Database

Contains six defect classes commonly studied in industrial inspection

Publicly available and widely cited in defect-detection literature

📈 Academic & Industrial Impact
Industrial

Reduces human subjectivity in inspection

Lowers deployment and compute costs

Suitable for constrained or edge environments

Academic

Demonstrates continued relevance of classical ML

Shows how feature combinations can rival more complex architectures

Encourages principled feature engineering over brute-force modeling

🧩 Limitations & Future Work

Performance still trails state-of-the-art deep learning in some classes

Dataset size limits generalization

Future work:

Hybrid classical + shallow neural models

Automated feature selection

Real-time industrial deployment testing

📄 Poster

A digital poster summarizing this work is included in the repository:

CST-590-RS-DigitalPoster.pptx 

CST-590-RS-DigitalPoster

📚 References

Yang et al. (2020) — Survey of deep learning for defect detection

Mohammed & Melhum (2020) — HOG feature extraction with tuned parameters

Gaja & Liou (2018) — Logistic regression and ANN methods for metal defect classification

📬 Contact

Stuart Spackman
GitHub: https://github.com/stuart-spackman

LinkedIn / Email: linkedin.com/stuart-spackman
