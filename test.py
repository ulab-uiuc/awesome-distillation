import pandas as pd

file_path = "/root/math/data/sft_qwen3_4b.parquet"
df = pd.read_parquet(file_path)

for i, row in df.head(10).iterrows():
    print("=" * 80)
    print(f"Sample {i}")
    print(row.to_dict())