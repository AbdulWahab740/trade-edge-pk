import pandas as pd

def load_trade_data(import_path, export_path):
    df_import = pd.read_csv(import_path)
    df_export = pd.read_csv(export_path)

    # Add trade type
    df_import["TradeType"] = "Import"
    df_export["TradeType"] = "Export"

    # Normalize column names
    for df in [df_import, df_export]:
        df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    # Merge
    df = pd.concat([df_import, df_export], ignore_index=True)
    return df

def normalize_trade_data(df):
    # Convert numeric columns safely
    for col in ["quantity_2025", "rupees_2025", "dollar_2025", 
                "quantity_2024", "rupees_2024", "dollar_2024"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derive unified columns (most recent available)
    df["quantity_2025"] = df["quantity_2025"].where(df["quantity_2025"] > 0, df["quantity_2024"])
    df["rupees_2025"] = df["rupees_2025"].where(df["rupees_2025"] > 0, df["rupees_2024"])
    df["dollar_2025"] = df["dollar_2025"].where(df["dollar_2025"] > 0, df["dollar_2024"])

    # Handle missing metadata
    df["group"] = df["group"].fillna("Unknown")
    df["commodity"] = df["commodity"].fillna("Unknown")
    df["month"] = df["month"].fillna("Unknown")

    return df


if __name__ == "__main__":
    import_path = "imports.csv"
    export_path = "exports.csv"
    df = load_trade_data(import_path, export_path)
    df = normalize_trade_data(df)
    # import csv
    # df.to_csv("trade_data.csv", index=False)

    cleaned_lines = []
    with open("trade_data.csv", 'r', encoding='utf-8') as f:
        for line in f:
            # Strip newline, then remove any trailing commas or whitespace
            cleaned_line = line.rstrip().rstrip(',')
            cleaned_lines.append(cleaned_line)

    # Write back cleaned lines
    with open("trade_data.csv", 'w', encoding='utf-8', newline='') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

    print("✅ Trailing commas removed successfully!")

    df.to_csv("trade_data.csv", index=False)