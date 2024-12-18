import pandas as pd

df = pd.read_parquet('/Users/elaineyys/Desktop/autogen_graphRAG/output/20241107-151951/artifacts/create_final_relationships.parquet', engine='pyarrow')
print(df.columns, df)