import pandas as pd
import os


input_dir = 'sequences'
output_dir = 'sliced_sequences'
for split in os.listdir(input_dir):
    for label in os.listdir(f'{input_dir}/{split}'):
        for filename in os.listdir(f'{input_dir}/{split}/{label}'):
            pattern = [0, 1, 2, 3, 4] * 12
            df = pd.read_csv(f'{input_dir}/{split}/{label}/{filename}')
            df = df.apply(lambda row: row.fillna(df.ffill().mean() + df.bfill().mean()) / 2, axis=1)
            if df.isna().any().any():
                print(f'{split}/{label}/{filename}')

            # Split the df into 5
            for i in range(5):
                sub_df = df.iloc[pattern.index(i)::5, :]
                sub_df.to_csv(f'{output_dir}/{split}/{label}/{filename[:-4]}_{i+1}.csv', index=False)
