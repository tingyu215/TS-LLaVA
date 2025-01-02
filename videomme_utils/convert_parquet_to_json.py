import duckdb
import json

# Define the path to your Parquet file
parquet_file = './ckpt/videomme/test-00000-of-00001.parquet'

# Connect to an in-memory DuckDB instance
con = duckdb.connect()

# Read the Parquet file into a DuckDB DataFrame
df = con.execute(f"SELECT * FROM read_parquet('{parquet_file}')").fetchdf()

# Convert the DataFrame to JSON
json_output = df.to_json(orient='records', lines=True)

# Print or save the JSON output
with open('./ckpt/videomme/temp.json', 'w') as f:
    f.write(json_output)

json_objects = []
with open('./ckpt/videomme/temp.json', 'r') as f:
    for line in f:
        json_objects.append(json.loads(line))

print(json_objects[0])


with open('./ckpt/videomme/test-00000-of-00001.json', 'w') as f:
    json.dump(json_objects, f)


with open('./ckpt/videomme/test-00000-of-00001.json', 'r') as f:
    gt_dict = json.load(f)

print(len(gt_dict))
print(gt_dict[0])