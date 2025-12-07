# 📊 METABRIC Breast Cancer Dashboard
Interactive genomic & clinical analytics built with **Dash**, **Plotly**, **Pandas**, and deployed using **Docker** for production environments.

## 🚀 Overview
This dashboard visualizes the **METABRIC breast cancer dataset**, providing tools for exploring:

- Clinical features  
- Mutation frequencies  
- mRNA expression patterns  
- PCA + clustering  
- Co-mutation & co-expression networks  
- Survival analysis (Kaplan-Meier curves)

## 🧬 Features
### 🔹 Clinical Exploration
- Age distribution  
- Tumor stage  
- Surgery type  
- Nottingham Prognostic Index  

### 🔹 Mutation Analysis
- Top mutated genes  
- Subtype-specific mutation rates  
- ER-status comparisons  
- Co-mutation heatmaps & network graph  
- Treatment correlations  
- Survival by mutation status  

### 🔹 mRNA Expression
- Gene expression boxplots  
- Correlation analysis  
- PCA projections  
- K-Means clustering  
- Co-expression heatmaps & networks  

### 🔹 Survival Analysis
- Cancer-specific vs overall survival  
- KM curves with confidence intervals  
- Comparison between mutation groups  

## 🧪 Local Development
```
pip install -r requirements.txt
python app.py
```

## 📁 Dataset
METABRIC dataset required in:
```
data/METABRIC_RNA_Mutation.csv
```

## 📄 License
Licensed under the **MIT License**.

## ⭐ Acknowledgements
- METABRIC Study  
- UConn CSE 5520
