import pandas as pd
from pathlib import Path

data_dir = Path("./WikiTableQuestions/csv/200-csv")
csv_files = sorted([(f for f in data_dir.glob("*.csv")), ])
dfs = []
for csv_file in csv_files:
    print(f"processing {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    except Exception as e:
        print(f"Error parsing {csv_file}: {str(e)}")