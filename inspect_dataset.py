
import pandas as pd
import os

file_path = "hackathon_preli_dataset.xlsx"

# Check whether the dataset exists
if os.path.exists(file_path):

    # Load the Excel dataset
    df = pd.read_excel(file_path)

    print("\n========== DATASET OVERVIEW ==========")

    print("\n1. Dataset Shape:")
    print(df.shape)

    print("\n2. Column Names:")
    print(df.columns.tolist())

    print("\n3. First 5 Rows:")
    print(df.head())

    print("\n4. Dataset Information:")
    df.info()

    print("\n5. Missing Values:")
    print(df.isnull().sum())

    print("\n6. Duplicate Rows:")
    print(df.duplicated().sum())

    print("\n7. Statistical Summary:")
    print(df.describe(include="all").T)

else:
    print("Dataset file not found!")
