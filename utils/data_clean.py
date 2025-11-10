import pandas as pd
import re

def clean_import_file_with_groups(file_path, month_label):
    df = pd.read_excel(file_path)

    # Drop header junk rows
    df = df.drop(index=[0,1,2]).reset_index(drop=True)

    df.columns = [
        "SL_NO", "Commodity", "Unit", "Quantity_2025", "Rupees_2025",
        "Dollar_2025", "Quantity_2024", "Rupees_2024", "Dollars_2024"
    ]

    df = df[df["Commodity"].notna()]  # Remove blank commodity rows
    
    # Detect group rows (ALL CAPS or contains 'GROUP')
    def is_group(row):
        text = str(row).strip()
        if "GROUP" in text.upper(): return True
        if bool(re.match(r'^[A-Z ]+$', text)) and len(text) > 3:
            return True
        return False

    current_group = None
    groups = []

    for item in df["Commodity"]:
        if is_group(item):
            current_group = item.title().replace("Group", "").strip() + " Group"
            groups.append(None)  # placeholder for group row
        else:
            groups.append(current_group)

    df["Group"] = groups

    # Remove rows that are group titles
    df = df[df["Group"].notna()]

    # Remove total rows if exist
    df = df[df["Commodity"].str.contains("TOTAL") == False]

    # Add Month column
    df["Month"] = month_label

    # Clean formatting
    df["Commodity"] = df["Commodity"].str.title().str.strip()
    df["Group"] = df["Group"].str.strip()

    df = df[
        [
            "month", "group", "commodity", "unit",
            "quantity_2025", "rupees_2025", "dollar_2025",
            "quantity_2024", "rupees_2024", "dollar_2024"
        ]
    ]

    return df

# df = pd.read_csv("imports.csv")
# # june_df = clean_import_file_with_groups("Export_September_2025.xlsx", "2025-09")
# # normalize commodity name for fuzzy matching
# # Filter the DataFrame to keep rows where 'Commodity', after being split, has a length of 1
# df = df[df['Commodity'].str.upper().str.split(" ").str.len() == 1]
# # print(df.head(),(len(df['Commodity'].str.split().str[0]) == 1))
# df.to_csv("importss.csv", index=False)

# import pandas as pd

# df = pd.read_csv("trade_data.csv")
# df["commodity"] = df["commodity"].apply(lambda x: x.lower().strip())
# cols = ["quantity_2025","quantity_2024","rupees_2025","rupees_2024","dollar_2025","dollar_2024"]

# # for c in cols:
# #     import_data[c] = pd.to_numeric(import_data[c], errors='coerce')  # keep decimal
# print(df[df['commodity'].str.contains('milkcream milk food for infants', case=False, na=False)][['month','rupees_2025']].sort_values('month'))
# # print(import_data[import_data['Group'] == 'Food Group']['Rupees_2025'].max())
# df.to_csv("trade_data.csv", index=False)

import pandas as pd
from pathlib import Path

# ---- LOAD DATA ----
# Load CSV from project root (works regardless of where script is run from)
project_root = Path(__file__).parent.parent
csv_file = project_root / "trade_data.csv"
df = pd.read_csv(csv_file)

df = df.sort_values(by='month')
df.to_csv("trade_data.csv", index=False)
