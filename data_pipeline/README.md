Data Pipeline — Books Catalog
Overview
This module implements an end‑to‑end ETL pipeline for the Books to Scrape dataset.
It scrapes ≥ 3 categories (≥ 60 books), cleans and converts fields, loads them into a normalized SQLite schema, and demonstrates SQL + pandas queries with tabular output.

nstall & Run
Requirements
Python ≥ 3.9

Libraries:
pip install requests beautifulsoup4 pandas tabulate

Run the pipeline
python books_data_pipeline.py

This will do following tasks:

1. Scrape books across 3 categories (Science, Travel, Mystery).

2. Clean fields into proper types.

3. Convert GBP → INR using the fixed baseline rate.

4. Create a normalized SQLite schema (books_catalog.db).

5. Execute ≥ 5 SQL queries with tabular output.

6. Show pandas equivalents of the join query.


Cleaning Decisions
Price (GBP): stripped £ symbol, converted to float.

Star rating: mapped text (One…Five) → integer (1–5).

Availability: parsed into boolean in_stock (True if “In stock”).

Error handling:

Numeric fields (price_gbp, rating) → median imputation.

Categorical/boolean failures (availability, category) → row dropped.

Currency conversion:

Fixed baseline: 1 GBP = 105.50 INR.

This constant is project‑defined and stated here; no API lookup required.

Database Schema
Two‑table normalized schema with PK/FK:

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY(category_id) REFERENCES categories(category_id)
);


Queries Demonstrated
SELECT + WHERE → books with rating = 5

ORDER BY + LIMIT → top 5 expensive books

DISTINCT → distinct ratings present

IN → books with rating IN (3,4)

JOIN → 10 highest‑rated books per category

All queries print results in tabular format using tabulate.

Pandas Equivalents
pd.read_sql used to fetch query results into DataFrames.

pd.merge reproduces the JOIN query directly in memory.

Outputs shown side‑by‑side to confirm equivalence.

Repository Notes
Module lives at /data_pipeline.

Includes:

books_data_pipeline.py (script)

books_catalog.db (SQLite database, or regeneratable script)

Executed SQL queries with tabular output

This README documenting install/run steps and design decisions

Submission Checklist
[x] ≥ 60 books scraped across ≥ 3 categories

[x] Cleaned fields: price_gbp, rating, in_stock, price_inr

[x] Fixed baseline conversion rate (1 GBP = 105.50 INR) stated here

[x] Normalized SQLite schema with PK/FK

[x] ≥ 5 SQL queries covering required clauses + JOIN

[x] Pandas equivalents shown and matching

[x] README with install/run steps and design decisions

[x] Feature branch created, committed ≥ 2 times, merged back into main

