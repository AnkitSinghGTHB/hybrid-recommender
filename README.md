# Hybrid Recommender System

## Overview

This project is a **Hybrid Recommender System** that can work with multiple types of datasets such as books, movies, or products.

It uses a **Data Adapter** to automatically understand and convert any dataset into a standard format, making the system flexible and scalable.

---

## Features

*  Upload any dataset (CSV)
*  Automatic column detection using Data Adapter
*  Hybrid recommendation system
*  Supports:

  * Content-Based Filtering
  * Collaborative Filtering
*  Interactive UI using Streamlit
*  Works across multiple domains (Books, Movies, etc.)

---

## System Architecture

User Upload
⬇
Data Adapter
⬇
Content-Based Model
⬇
Collaborative Model (if available)
⬇
Hybrid Model
⬇
Recommendations

---

##  Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

---

##  Project Structure

```
app.py                  # Streamlit UI
data_adapter.py         # Data preprocessing & column mapping
content_model.py        # Content-based filtering
collaborative_model.py  # Collaborative filtering
hybrid_model.py         # Hybrid recommendation logic
raw_data.py             # Dataset preparation
datasets/               # Dataset files
requirements.txt        # Dependencies
```

---

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the application:

```
streamlit run app.py
```

---

## Dataset

This project uses datasets such as:

* Goodbooks-10k (from Kaggle)

You can also upload your own dataset.


---

##  Key Concept

The system uses cosine similarity for content-based filtering:

* Finds similarity between items based on features
* Recommends items with highest similarity

It also uses collaborative filtering:

* Recommends items liked by similar users

---

##  Future Improvements

* Better hybrid scoring (weighted model)
* Deep learning-based recommendations
* Improved UI with images
* Smart dataset type detection

---

⭐ If you like this project, consider giving it a star!
