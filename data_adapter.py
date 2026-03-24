import pandas as pd

def detect_column(columns, keywords):
    for col in columns:
        for key in keywords:
            if key in col.lower():
                return col
    return None


def adapt_data(df):
    columns = df.columns

    title_col = detect_column(columns, ['title', 'name'])
    desc_col = detect_column(columns, ['desc', 'summary', 'overview'])
    user_col = detect_column(columns, ['user'])
    rating_col = detect_column(columns, ['rating', 'score', 'stars'])

    df = df.copy()

    rename_map = {}
    if title_col: rename_map[title_col] = 'title'
    if desc_col: rename_map[desc_col] = 'description'
    if user_col: rename_map[user_col] = 'user_id'
    if rating_col: rename_map[rating_col] = 'rating'

    df = df.rename(columns=rename_map)

    # Fill missing safely
    # Handle title
    if 'title' in df.columns:
        df['title'] = df['title'].fillna('')
    else:
        df['title'] = df.iloc[:, -1].astype(str)   # fallback (use last column like label)

    # Handle description
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('')
    else:
        df['description'] = df.astype(str).agg(' '.join, axis=1)
        
    # Combine features
    df['combined'] = df['title'] + " " + df['description']

    meta = {
        "title_col": title_col,
        "desc_col": desc_col,
        "user_col": user_col,
        "rating_col": rating_col,
        "has_user_data": user_col is not None and rating_col is not None
    }

    return df, meta