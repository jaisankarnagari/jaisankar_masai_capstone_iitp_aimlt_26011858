import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import sqlite3
from pathlib import Path
from tabulate import tabulate

# -------------------------------
# Task 1: Scrape books across categories
# -------------------------------
def scrape_books(categories_to_scrape):
    base_url = "https://books.toscrape.com/catalogue/category/books/{}/index.html"
    books = []
    for cat in categories_to_scrape:
        url = base_url.format(cat)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        category_name = soup.select_one("div.page-header.action h1").text.strip()
        for article in soup.select("article.product_pod"):
            title = article.h3.a["title"]
            price_raw = article.select_one("p.price_color").text.strip()
            star_rating = article.select_one("p.star-rating")["class"][1]
            availability = article.select_one("p.availability").text.strip()
            books.append({
                "title": title,
                "price_gbp": price_raw,
                "star_rating": star_rating,
                "availability": availability,
                "category": category_name
            })
    df = pd.DataFrame(books)
    print("Raw scraped data:")
    print(tabulate(df.head(10), headers="keys", tablefmt="grid", showindex=False))
    return df

# -------------------------------
# Task 2: Clean fields
# -------------------------------
def clean_books(df):
    def parse_price(val):
        try:
            cleaned = re.sub(r"[^\d.]", "", val)
            return float(cleaned) if cleaned else None
        except:
            return None
    rating_map = {"One":1, "Two":2, "Three":3, "Four":4, "Five":5}
    def parse_availability(text):
        if isinstance(text, str) and "In stock" in text:
            return True
        elif isinstance(text, str):
            return False
        else:
            return None
    df["price_gbp"] = df["price_gbp"].apply(parse_price)
    df["rating"] = df["star_rating"].map(rating_map)
    df["in_stock"] = df["availability"].apply(parse_availability)
    for col in ["price_gbp", "rating"]:
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=["in_stock", "category"]).reset_index(drop=True)
    print("Cleaned data:")
    print(tabulate(df.head(10)[["title","price_gbp","rating","in_stock","category"]],
                   headers="keys", tablefmt="grid", showindex=False))
    return df

# -------------------------------
# Task 3: Currency conversion
# -------------------------------
def convert_currency(df, rate=105.50):
    df["price_inr"] = df["price_gbp"] * rate
    print("With INR conversion:")
    print(tabulate(df.head(10)[["title","price_gbp","price_inr","rating","in_stock","category"]],
                   headers="keys", tablefmt="grid", showindex=False))
    return df

# -------------------------------
# Task 4: Normalized SQLite schema
# -------------------------------
def get_db_path(db_name="books_catalog.db"):
    if Path(db_name).is_absolute():
        return Path(db_name)
    return Path(__file__).resolve().parent / db_name


def create_schema_and_insert(df, db_name="books_catalog.db"):
    db_path = get_db_path(db_name)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS books")
    cur.execute("DROP TABLE IF EXISTS categories")
    cur.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE
    )
    """)
    cur.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        price_gbp REAL,
        price_inr REAL,
        rating INTEGER,
        in_stock INTEGER,
        category_id INTEGER,
        FOREIGN KEY(category_id) REFERENCES categories(category_id)
    )
    """)
    categories = df["category"].unique()
    for cat in categories:
        cur.execute("INSERT INTO categories (category_name) VALUES (?)", (cat,))
    conn.commit()
    cat_map = {row[1]: row[0] for row in cur.execute("SELECT * FROM categories")}
    for _, row in df.iterrows():
        cur.execute("""
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (row["title"], row["price_gbp"], row["price_inr"], int(row["rating"]), int(row["in_stock"]), cat_map[row["category"]]))
    conn.commit()
    return conn

# -------------------------------
# Task 5: SQL Queries with tabulate
# -------------------------------
def run_sql_queries(conn):
    q1 = pd.read_sql("SELECT title, price_inr FROM books WHERE rating=5", conn)
    print("Books with rating=5:")
    print(tabulate(q1, headers="keys", tablefmt="grid", showindex=False), "\n")
    q2 = pd.read_sql("SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 5", conn)
    print("Top 5 expensive books:")
    print(tabulate(q2, headers="keys", tablefmt="grid", showindex=False), "\n")
    q3 = pd.read_sql("SELECT DISTINCT rating FROM books", conn)
    print("Distinct ratings:")
    print(tabulate(q3, headers="keys", tablefmt="grid", showindex=False), "\n")
    q4 = pd.read_sql("SELECT title, rating FROM books WHERE rating IN (3,4)", conn)
    print("Books with rating IN (3,4):")
    print(tabulate(q4, headers="keys", tablefmt="grid", showindex=False), "\n")
    q5 = pd.read_sql("""
    SELECT b.title, b.rating, c.category_name
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    ORDER BY c.category_name, b.rating DESC
    LIMIT 10
    """, conn)
    print("Join: 10 highest-rated books per category:")
    print(tabulate(q5, headers="keys", tablefmt="grid", showindex=False))

# -------------------------------
# Task 6: Pandas equivalents
# -------------------------------
def pandas_equivalents(conn):
    q1_df = pd.read_sql("SELECT title, price_inr FROM books WHERE rating=5", conn)
    q2_df = pd.read_sql("SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 5", conn)
    print("Pandas DataFrame for rating=5:")
    print(tabulate(q1_df, headers="keys", tablefmt="grid", showindex=False), "\n")
    print("Pandas DataFrame for Top 5 expensive:")
    print(tabulate(q2_df, headers="keys", tablefmt="grid", showindex=False), "\n")
    books_df = pd.read_sql("SELECT * FROM books", conn)
    cats_df = pd.read_sql("SELECT * FROM categories", conn)
    join_df = pd.merge(books_df, cats_df, on="category_id")
    join_sorted = join_df.sort_values(["category_name","rating"], ascending=[True,False]).head(10)
    print("Pandas merge equivalent of JOIN:")
    print(tabulate(join_sorted[["title","rating","category_name"]],
                   headers="keys", tablefmt="grid", showindex=False))

# -------------------------------
# Example usage (end-to-end run)
# -------------------------------
if __name__ == "__main__":
    df_raw = scrape_books(["science_22","travel_2","mystery_3"])
    df_clean = clean_books(df_raw)
    df_conv = convert_currency(df_clean)
    conn = create_schema_and_insert(df_conv)
    run_sql_queries(conn)
    pandas_equivalents(conn)
