# 🚔 PatrolIQ – Smart Safety Analytics Platform

## Crime Hotspot & Pattern Analysis using Machine Learning

---

# 📌 Project Overview

PatrolIQ is a data-driven crime intelligence and smart safety analytics platform developed using the Chicago Crime Dataset. The platform analyzes large-scale crime data to identify geographic crime hotspots, temporal crime patterns, and actionable patrol strategies using Machine Learning and interactive visual analytics.

The project combines clustering algorithms, dimensionality reduction techniques, and dashboard visualization to support smarter patrol planning and public safety decision-making.

---

# 🎯 Objectives

* Identify geographic crime hotspots
* Analyze temporal crime patterns
* Detect high-risk zones and time periods
* Provide actionable patrol strategies
* Build a production-ready interactive dashboard

---

# 📂 Dataset Information

* Dataset: Chicago Crime Dataset
* Total Records: ~7.8 Million
* Dataset Size: ~1.7 GB
* Working Sample: 500,000 recent records

---

# 🔄 Project Workflow

# Step 1: Data Acquisition & Preprocessing

Downloaded and processed the Chicago Crime Dataset for large-scale crime analytics.

### Tasks Performed

* Missing value handling
* Data cleaning and validation
* Datetime conversion
* Feature extraction from timestamps

### Temporal Features Extracted

* Hour
* Day of Week
* Month

---

# 📊 Step 2: Exploratory Data Analysis (EDA)

Performed extensive exploratory analysis to understand crime distribution and patterns.

### Analysis Performed

* Distribution across 30+ crime types
* Geographic pattern analysis using latitude and longitude
* Temporal trend analysis:

  * Hourly trends
  * Daily trends
  * Seasonal trends
* Arrest rate analysis
* Domestic crime analysis
* Statistical summaries and insights

---

# 🛠️ Step 3: Feature Engineering

Developed meaningful spatial and temporal features for clustering and analysis.

### Temporal Features

* Hour
* Day
* Month
* Weekend Indicator
* Season

### Geographic Features

* Coordinate binning
* Normalized latitude and longitude

### Additional Engineering

* Crime Severity Score
* Frequency encoding for Location Description

---

# 🤖 Step 4: Clustering Analysis

## 📍 Geographic Clustering

### K-Means Clustering (Final Model)

Identified 7 distinct crime hotspots using K-Means clustering.

### Model Selection Techniques

* Elbow Method
* Silhouette Score

### DBSCAN

* Detected high-density crime regions
* Identified noise and outlier zones
* Applied on sampled data for computational efficiency

### Hierarchical Clustering

* Used Ward linkage
* Generated dendrograms for cluster hierarchy analysis

---

## ⏱️ Temporal Clustering

Applied K-Means clustering on:

* Hour
* Day
* Month
* Season

### Identified Temporal Crime Patterns

* Late-night high-risk crime activity
* Weekend crime spikes
* Seasonal patterns (Winter/Fall)
* Daytime activity clusters

---

# 📈 Model Evaluation

### Evaluation Metrics

* Silhouette Score
* Elbow Method
* Cluster consistency comparison

### Final Model Selection

K-Means was selected as the final model because of:

* Scalability
* Interpretability
* Consistent clustering performance

---

# 📉 Step 5: Dimensionality Reduction

## PCA (Principal Component Analysis)

* First 2 components explained only ~38% variance
* Required ~6 components to capture 70% variance
* Indicated high intrinsic dimensionality

## t-SNE

* Captured local cluster structures
* Revealed non-linear patterns

## UMAP (Best Technique)

* Preserved both local and global structure
* Produced clear cluster separation
* Delivered the best visualization of crime patterns

---

# 🔥 Key Insight

The dataset exhibited complex non-linear relationships, making PCA insufficient for low-dimensional representation.

Non-linear dimensionality reduction techniques like UMAP provided the most meaningful and interpretable visualization of crime patterns.

---

# 📈 Step 6: MLflow Integration

Integrated MLflow for experiment tracking and model management.

### Tracked Parameters

* Number of clusters (K)
* DBSCAN parameters (eps, min_samples)
* Model metrics
* Clustering performance

### Benefits

* Experiment reproducibility
* Model comparison
* Performance tracking

---

# 🌐 Step 7: Streamlit Dashboard Application

Built a production-ready interactive dashboard using Streamlit.

## Dashboard Features

### 📍 Geographic Crime Maps

* Crime hotspot visualization
* Cluster maps
* Heatmaps

### ⏱️ Temporal Crime Analysis

* Hourly crime trends
* Daily patterns
* Seasonal analysis

### 📊 Dimensionality Reduction Visualizations

* PCA plots
* UMAP visualizations

### 🧠 Cluster Interpretation Panels

* Crime hotspot understanding
* Cluster-based insights

### 🚔 Patrol Strategy Recommendations

* Smart patrol planning suggestions
* High-risk zone recommendations

### 📈 MLflow Monitoring

* Model tracking visualization
* Experiment comparison

---

# ☁️ Step 8: Deployment

Deployed the application using Streamlit Cloud with GitHub integration.

### Deployment Features

* Responsive UI
* Error handling
* Smooth user experience
* Cloud accessibility

---

# 🧠 Key Insights

## 📍 Spatial Insights

Identified 7 major crime hotspots including:

* Retail theft zones
* Residential violence areas
* Night-time high-risk zones

## ⏱️ Temporal Insights

Detected 5 major crime patterns:

* Late-night crime peaks
* Weekend activity spikes
* Seasonal variations
* Daytime activity clusters

## 📊 Dimensionality Insights

* Crime data exhibited strong non-linear behavior
* UMAP outperformed PCA and t-SNE for visualization quality

---

# 🚔 Patrol Strategy Recommendations

* Increase patrol deployment in violent crime clusters
* Strengthen late-night patrol coverage
* Deploy additional forces during weekends
* Focus theft prevention in high-traffic commercial areas
* Adjust patrol strategies for seasonal crime variations

---

# 🛠️ Tech Stack

## Programming Language

* Python

## Libraries & Frameworks

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* UMAP
* MLflow

---

# 📁 Project Structure

```bash
PatrolIQ-Smart-Safety-Analytics-Platform/
│
├── streamlit_app.py
├── Model_Evaluation.ipynb
├── Data_aquisition_&_sampling.ipynb
├── data_cleaning.ipynb
├── cleaned_chicago_crime_dataset.ipynb
├── feature_engieering.ipynb
├── dimentionality_reduction.ipynb
├── requirements.txt
├── README.md
```

# 👨‍💻 Author
## Mohanraj D.
