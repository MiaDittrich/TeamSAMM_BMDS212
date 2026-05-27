import pandas as pd
import numpy as np

# Include a file path to the .csv of the BRCA1 variants to sample from
file_path = "BRCA1_last300.csv"

df = pd.read_csv(file_path)
print(df.head())
df['hgvs_pro'] = df['hgvs_pro'].str[2:]
df['hgvs_pro'] = df['hgvs_pro'].str.replace(r'\d+', '', regex=True)
seed = 42

selected = df.sample(n=40, random_state = seed, replace = False).sort_values(by = 'uniprot_position')
residues = selected[['uniprot_position', 'hgvs_pro']]
residues.reset_index(drop=True)
residues['variant'] = (
    residues['hgvs_pro'].str[:3] +
    residues['uniprot_position'].astype(str) +
    residues['hgvs_pro'].str[-3:]
)

reference_variants = [
    "Ser1841Arg", "Leu1839Ser", "Val1838Glu", "Ala1708Glu",
    "Arg1699Gln", "Met1775Arg", "Met1775Lys", "Tyr1703Ser",
    "Trp1718Leu", "Trp1718Ser", "Gly1770Val", "Cys1697Arg",
    "Ser1715Arg", "Met1552Ile", "Met1652Ile", "Val1665Met",
    "Asp1692Asn", "Asp1733Gly", "Met1775Ile"
]

overlapping = residues[residues['variant'].isin(reference_variants)]

if overlapping.empty:
    print("No overlapping variants found.")
else:
    print(f"Found {len(overlapping)} overlapping variant(s):")
    print(overlapping.to_string(index=False))

print(residues['variant'].to_string(index = False))