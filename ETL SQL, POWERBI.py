"""Perform the Extraction Transformation and Loading (ETL) process to construct the
database in the Sql server / Power BI."""
# Import libraries
import pandas as pd

# ----------------------------------------
# Step 1: Extract Data
# ----------------------------------------

# Extract Sales data
sales_df = pd.read_excel("sales_data.xlsx", sheet_name="Sales Data")
print("Raw Sales Data:")
print(sales_df.head())

# Extract Customer data
customers_df = pd.read_csv("customers.csv")
print("\nRaw Customer Data:")
print(customers_df.head())

# Extract Product data
products_df = pd.read_excel("products.xlsx", sheet_name="Sheet1")
print("\nRaw Product Data:")
print(products_df.head())

# ----------------------------------------
# Step 2: Transform Data
# ----------------------------------------

# Merge Sales with Product Data based on SalesMan and Sales_Rep_Name
final_df = pd.merge(sales_df, products_df, left_on="SalesMan", right_on="Sales_Rep_Name", how="left")

# Calculate Total Revenue
final_df["TotalRevenue"] = final_df["Units"] * final_df["Unit_price"]

# Drop missing values
final_df.dropna(inplace=True)

# Show transformed data
print("\nTransformed Data:")
print(final_df.head())

# ----------------------------------------
# Step 3: Save to File for Power BI
# ----------------------------------------

# Save the final DataFrame to CSV (for Power BI import)
final_df.to_csv("Transformed_SalesReport.csv", index=False)
print("\nTransformed data saved to 'Transformed_SalesReport.csv'!")
