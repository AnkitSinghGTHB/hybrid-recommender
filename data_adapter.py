import pandas as pd
import numpy as np


def detect_column(columns, keywords):
    """Detect a column by matching against a list of keywords (case-insensitive)."""
    for col in columns:
        for key in keywords:
            if key in col.lower():
                return col
    return None


def validate_dataframe(df):
    """Check that a DataFrame has the minimum required data for recommendation."""
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if len(df.columns) < 2:
        raise ValueError("DataFrame must have at least 2 columns.")
    return True


def adapt_data(df):
    """
    Adapt any CSV DataFrame into the unified schema used by the recommender.
    
    Detected columns:
      - title / name         → 'title'
      - description / summary → 'description'
      - user / user_id       → 'user_id'
      - rating / score       → 'rating'
      - review / text        → 'review_text'
      - category / genre     → 'category'
      - item_id / product_id → 'item_id'
      - clicks / views       → 'views'
      - purchases / orders   → 'purchases'
    
    Returns: (adapted_df, meta_dict)
    """
    validate_dataframe(df)
    columns = df.columns

    # --- Detect columns ---
    title_col    = detect_column(columns, ['title', 'name', 'product_name', 'item_name'])
    desc_col     = detect_column(columns, ['desc', 'summary', 'overview', 'about'])
    user_col     = detect_column(columns, ['user_id', 'user', 'reviewer', 'customer'])
    rating_col   = detect_column(columns, ['rating', 'score', 'stars'])
    review_col   = detect_column(columns, ['review', 'text', 'comment', 'feedback', 'review_text'])
    category_col = detect_column(columns, ['category', 'genre', 'tags', 'type', 'department'])
    item_id_col  = detect_column(columns, ['item_id', 'product_id', 'asin', 'isbn', 'book_id', 'movie_id'])
    views_col    = detect_column(columns, ['views', 'clicks', 'impressions'])
    purchase_col = detect_column(columns, ['purchases', 'orders', 'bought', 'transactions'])

    df = df.copy()

    # --- Rename detected columns ---
    rename_map = {}
    if title_col:    rename_map[title_col]    = 'title'
    if desc_col:     rename_map[desc_col]     = 'description'
    if user_col:     rename_map[user_col]     = 'user_id'
    if rating_col:   rename_map[rating_col]   = 'rating'
    if review_col:   rename_map[review_col]   = 'review_text'
    if category_col: rename_map[category_col] = 'category'
    if item_id_col:  rename_map[item_id_col]  = 'item_id'
    if views_col:    rename_map[views_col]    = 'views'
    if purchase_col: rename_map[purchase_col] = 'purchases'

    df = df.rename(columns=rename_map)

    # --- Fill missing safely ---
    # Title
    if 'title' in df.columns:
        df['title'] = df['title'].fillna('Unknown')
    else:
        df['title'] = df.iloc[:, 0].astype(str)

    # Description
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('')
    else:
        df['description'] = ''

    # Category
    if 'category' not in df.columns:
        df['category'] = ''
    else:
        df['category'] = df['category'].fillna('')

    # Item ID
    if 'item_id' not in df.columns:
        df['item_id'] = range(len(df))

    # Review text
    if 'review_text' in df.columns:
        df['review_text'] = df['review_text'].fillna('')
    else:
        df['review_text'] = ''

    # Rating
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    else:
        df['rating'] = 0.0

    # User
    if 'user_id' not in df.columns:
        df['user_id'] = 'anonymous'

    # Behavior
    if 'views' not in df.columns:
        df['views'] = 0
    if 'purchases' not in df.columns:
        df['purchases'] = 0

    # --- Combined text feature for content-based model ---
    df['combined'] = (
        df['title'].astype(str) + ' ' +
        df['description'].astype(str) + ' ' +
        df['category'].astype(str)
    )

    meta = {
        'title_col':    title_col,
        'desc_col':     desc_col,
        'user_col':     user_col,
        'rating_col':   rating_col,
        'review_col':   review_col,
        'category_col': category_col,
        'item_id_col':  item_id_col,
        'views_col':    views_col,
        'purchase_col': purchase_col,
        'has_user_data':  user_col is not None and rating_col is not None,
        'has_reviews':    review_col is not None,
        'has_behavior':   views_col is not None or purchase_col is not None,
        'total_rows':     len(df),
        'total_columns':  len(df.columns),
    }

    return df, meta