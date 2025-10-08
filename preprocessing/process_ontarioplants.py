import pandas as pd
df = pd.read_csv("OntarioPlants.csv")


### FILTERING DATA TO KEEP ONLY COMMON TYPES OF VASCULAR PLANTS

# Clean up
df.columns = df.columns.str.strip().str.upper()
df["EXOTIC_STATUS"] = df["EXOTIC_STATUS"].astype(str).str.strip().str.upper()
df["PROVINCIALLY_TRACKED"] = df["PROVINCIALLY_TRACKED"].astype(str).str.strip().str.upper()
df["RESTRICTED_SPECIES"] = df["RESTRICTED_SPECIES"].astype(str).str.strip().fillna("N")
df["S_RANK"] = df["S_RANK"].astype(str).str.strip().str.upper()
df["COEFF_CONSERVATISM"] = pd.to_numeric(df["COEFF_CONSERVATISM"], errors="coerce")

print("Original:", len(df))

# Native = rows where EXOTIC_STATUS is blank or NaN
##df_native = df[df["EXOTIC_STATUS"].isin(["", "NAN"])]

# Keep secure/common species
df_native = df[df["S_RANK"].str.contains("S4|S5|SNA", na=False)]

# # Keep vascular plants
df_native = df_native[df_native["NHIC_CLASS"].str.contains("Vascular", na=False)]

# Select columns
keep_cols = [
    "SCIENTIFIC_NAME", "ENGLISH_COMMON_NAME", "FAMILY", "GENUS",
    "S_RANK", "COEFF_CONSERVATISM", "S_RANK_REASONS"
]
ontario_native = df_native[keep_cols].drop_duplicates()

print("Remaining species:", len(ontario_native))

## Result: 3208
ontario_native.to_csv("ontario_native_filtered.csv", index=False)
print("Exported succesfully!")