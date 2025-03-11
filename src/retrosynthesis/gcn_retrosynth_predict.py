# import os
# import sqlite3
# import pandas as pd

# # Define the data directory and database path.
# DATA_DIR = "data"
# db_path = os.path.join(DATA_DIR, "chembl_35.db")

# # Connect to the ChEMBL database.
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# # Query to extract canonical SMILES from compound_structures.
# # (Adjust the table/column names if your schema differs.)
# query = """
# SELECT canonical_smiles
# FROM compound_structures
# WHERE canonical_smiles IS NOT NULL
# LIMIT 10000
# """

# df = pd.read_sql_query(query, conn)
# conn.close()

# # Save the extracted SMILES to a CSV file.
# csv_path = os.path.join(DATA_DIR, "molecule_smiles.csv")
# df.to_csv(csv_path, index=False)
# print(f"Extracted {len(df)} SMILES and saved to {csv_path}")

import sqlite3

conn = sqlite3.connect("data/chembl_35.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)
conn.close()
