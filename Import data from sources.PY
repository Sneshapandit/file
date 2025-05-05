"""Import Data from different Sources such as (Excel, Sql Server, Oracle etc.) and load in
targeted system. """
✅ Step 1: Install required libraries
Run this in your terminal:

pip install pandas sqlalchemy openpyxl pyodbc cx_Oracle
Also install drivers:

sudo apt-get install unixodbc-dev
sudo apt-get install libaio1
✅ Step 2: Python Code to Import and Load Data


import pandas as pd
from sqlalchemy import create_engine

# ---------- Read from Excel ----------
excel_df = pd.read_excel("data.xlsx")  # path to your Excel file

# ---------- Read from SQL Server ----------
sql_server_conn_str = "mssql+pyodbc://username:password@server_name/database_name?driver=ODBC+Driver+17+for+SQL+Server"
sql_server_engine = create_engine(sql_server_conn_str)
sql_df = pd.read_sql("SELECT * FROM your_table", sql_server_engine)

# ---------- Read from Oracle ----------
oracle_conn_str = "oracle+cx_oracle://username:password@host:port/?service_name=your_service"
oracle_engine = create_engine(oracle_conn_str)
oracle_df = pd.read_sql("SELECT * FROM your_table", oracle_engine)

# ---------- Target system (example: save all to CSV) ----------
excel_df.to_csv("output_excel.csv", index=False)
sql_df.to_csv("output_sql.csv", index=False)
oracle_df.to_csv("output_oracle.csv", index=False)
🔁 Alternative Target: Load into PostgreSQL


pg_engine = create_engine("postgresql://user:password@localhost:5432/target_db")

excel_df.to_sql("excel_table", pg_engine, if_exists="replace", index=False)
sql_df.to_sql("sql_table", pg_engine, if_exists="replace", index=False)
oracle_df.to_sql("oracle_table", pg_engine, if_exists="replace", index=False)