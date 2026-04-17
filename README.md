# MovieLens 10M Recommender System

A professional-grade movie recommendation engine built using the **MovieLens 10M dataset**. This project implements a full machine learning pipeline—from memory-optimized data ingestion and statistical genre analysis to a custom-built Biased Matrix Factorization model optimized via Stochastic Gradient Descent.

##  Project Overview

The system is designed with a modular, **Object-Oriented Programming (OOP)** architecture, ensuring scalability and clean separation of concerns. It addresses the core challenges of recommendation systems, including the "Cold Start" problem and temporal validation.

### Key Features
* **Memory-Efficient Data Loading:** Custom loader handling 10 million ratings with optimized data types.
* **Genre Trend Analysis:** Identification of shifts in genre popularity over time using **Bayesian Weighting** to correct for sample size bias in early data years.
* **Biased Matrix Factorization:** A latent factor model implemented from scratch using **Stochastic Gradient Descent (SGD)**.
* **Temporal Validation:** A realistic train/test split (Training on data pre-2008, testing on 2008-2009).
* **Dual-Engine Recommendation:**
    * **Popularity-Based:** For "Cold Start" users with no history.
    * **Item-Item Similarity:** Using latent factors and **Cosine Similarity** to provide recommendations based on specific user interests (e.g., "Iron Man", "300").

---

## 📂 Project Structure

```text
ml10m-recommender/
├── data/               # Project data (ignored by git)
├── notebooks/          # Interactive execution
│   └── main.ipynb      # Main project notebook
├── src/                # Core logic (OOP Modules)
│   ├── __init__.py
│   ├── data_loader.py  # Data ingestion and preprocessing
│   ├── analyzer.py     # Genre and trend analysis logic
│   ├── recommender.py  # MF Model and Recommender Engine
├── requirements.txt    # Project dependencies
├── .gitignore          # Data and environment exclusions
└── README.md           # Project documentation
```

## 🛠️ Installation & Setup

Clone the repository:

```commandline
git clone [https://github.com/seferlisk/ml-10m-recommender.git](https://github.com/seferlisk/ml-10m-recommender.git)
cd recommender-ml10m
```
Create a virtual environment:
```commandline
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:
```commandline
pip install -r requirements.txt
```
## 📊 Methodology

1. Data Processing
* Data is loaded from movies.dat and ratings.dat. We utilize int32 and float32 types to minimize the RAM footprint, allowing 10M rows to be processed on standard hardware.

2. Genre Decline Analysis
   * We analyze the average annual ratings per genre. To prevent "noisy" early-year data from skewing results, we apply a **Bayesian Average** adjustment, pulling averages with low rating counts toward the global mean.

3. Recommendation Engine
   * **Model**: The model learns User and Movie latent vectors. We incorporate Biases (User and Movie specific) to account for different rating scales and inherent movie quality.

   * **Optimization**: Stochastic Gradient Descent (SGD) with L2 Regularization to prevent overfitting and gradient explosion.

   * **Similarity**: For context-aware recommendations, we generate a "user profile" by averaging latent vectors of liked movies and calculating the Cosine Similarity against the rest of the catalog.

## 📈 Results
The model's performance is evaluated using Mean Squared Error (MSE) on a temporal test set (ratings from 2008-2009).

## 📜 License
#### Distributed under the  Apache-2.0 license.

---

## 👤 Author
***Konstantinos Seferlis***

AI Research & Development

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/konstantinos-seferlis-b16bb7155/)

