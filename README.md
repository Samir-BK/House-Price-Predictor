# 🏠 House Price Predictor

A supervised machine learning project that predicts residential house sale prices using Linear Regression. Built as part of a Supervised ML assignment for **HomeVista Properties** — a real estate company that wants to automate its house pricing process.

---

## 📌 Project Overview

This project uses historical property data to train a Linear Regression model that can predict the market price of a house based on its physical features, location, and condition.

**Tools & Technologies:**
- Python 3
- JupyterLab
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn

---

## 📂 Project Structure

```
House-Price-Predictor/
│
├── Predictor_ML.ipynb          # Main Jupyter notebook
└── README.md                   # Project documentation
```

---

## 📊 Dataset Description

Each row represents one residential house with its physical, location, and construction details.

| Feature | Description |
|---|---|
| `Id` | Unique identification number |
| `MSSubClass` | Type of dwelling (20 = 1-Story, 60 = 2-Story, etc.) |
| `MSZoning` | General zoning classification (RL, RM, FV, etc.) |
| `LotArea` | Lot size in square feet |
| `LotConfig` | Lot configuration (Inside, Corner, CulDSac, etc.) |
| `BldgType` | Type of dwelling (1Fam, Duplex, Townhouse, etc.) |
| `OverallCond` | Overall condition rating (scale 1–10) |
| `YearBuilt` | Original construction year |
| `YearRemodAdd` | Year the house was remodeled |
| `Exterior1st` | Exterior covering (VinylSd, MetalSd, HdBoard, etc.) |
| `BsmtFinSF2` | Type 2 finished basement square feet |
| `TotalBsmtSF` | Total basement area in square feet |
| `SalePrice` | **Target variable** — Final selling price of the house |

---

## ⚙️ Methodology

### 1. Data Loading & Exploration
- Loaded dataset using `pandas`
- Explored data shape, column types, and distributions
- Visualized `SalePrice` distribution and feature relationships using Seaborn scatterplots

### 2. Feature Engineering
- **Dropped irrelevant columns:** `Id`, `BldgType`, `BsmtFinSF2`
- **One-Hot Encoded** categorical columns: `MSZoning`, `LotConfig`, `Exterior1st`
- **Encoded `MSSubClass`** as a category (converted to string before encoding since it represents house types, not real numeric values)
- **Created new features:**
  - `HouseAge = 2026 - YearBuilt`
  - `YearsSinceRemodel = 2026 - YearRemodAdd`
- **Dropped original year columns** after creating age features
- **Handled missing values** using median imputation for `TotalBsmtSF`
- **Removed outliers** — houses priced above $400,000 excluded to reduce skew impact
- **Feature Scaling** using `StandardScaler` for normalized numeric features

### 3. Model Training
- Split data into training and test sets using `train_test_split` (80/20 split)
- Trained a `LinearRegression` model from scikit-learn

### 4. Model Evaluation
- Evaluated using **Mean Absolute Error (MAE)** and **R² Score**

---

## 📈 Results

| Metric | Score |
|---|---|
| Mean Absolute Error | ~$28,628 |
| R² Score | 0.2769 |

The model provides a working baseline. The relatively low R² reflects the limitations of Linear Regression on complex, non-linear real estate data. Improvements can be made by using more features or switching to a more powerful model like Random Forest.

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/Samir-BK/House-Price-Predictor.git
cd House-Price-Predictor
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab
```

3. Launch JupyterLab:
```bash
jupyter lab
```

4. Open `Predictor_ML.ipynb` and run all cells.

---

## 🔮 Future Improvements

- Add more features (bedrooms, bathrooms, neighbourhood, garage size)
- Try Random Forest or Gradient Boosting for better accuracy
- Apply log transformation to `SalePrice` to reduce skewness
- Use cross-validation for more reliable evaluation

---

## 👤 Author

**Samir B K**  
Bachelor of Business in IT — Software Development & AI  
Haaga-Helia University of Applied Sciences, Helsinki  
GitHub: [@Samir-BK](https://github.com/Samir-BK)
